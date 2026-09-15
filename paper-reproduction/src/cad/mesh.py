"""
CAD mesh loading and lightweight software rendering of poses over an image.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

# Distinct overlay colors per category (BGR for OpenCV).
CATEGORY_COLORS = [
    (66, 133, 244),
    (52, 168, 83),
    (238, 178, 0),
    (188, 64, 188),
    (0, 188, 251),
]

# Vertices with a camera-space depth below this (in meters) are treated as
# behind/at the camera and skipped during projection.
MIN_DEPTH_M = 0.05


def load_obj(path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read an OBJ file into (vertices [V,3], triangle faces [F,3])."""
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # OBJ face tokens are "idx", "idx/uv" or "idx/uv/normal".
                idx = [int(token.split("/")[0]) for token in line.split()[1:]]
                # OBJ indices are 1-based; negatives count from the end.
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                # Fan-triangulate the (possibly n-gon) face.
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return (
        torch.tensor(vertices, dtype=torch.float32),
        torch.tensor(faces, dtype=torch.int64),
    )


def bbox_center(vertices: torch.Tensor) -> torch.Tensor:
    """Centre of the axis-aligned bounding box of the vertices."""
    return (vertices.amin(dim=0) + vertices.amax(dim=0)) / 2.0


def normalize_class_name(name: str) -> str:
    """Canonical form for matching class names to OBJ filenames.

    Lower-cases and drops every non-alphanumeric character so that e.g. the
    class ``ma_simple_object_higher`` matches the file ``MA Simple Object
    higher.obj`` regardless of spacing, casing or separators.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def index_cad_directory(cad_dir: str) -> Dict[str, Path]:
    """Map ``normalize_class_name(stem) -> path`` for every .obj in ``cad_dir``."""
    return {
        normalize_class_name(path.stem): path
        for path in sorted(Path(cad_dir).glob("*.obj"))
    }


def build_categories(labels: List[str], num_classes: int) -> List[Dict]:
    """Map class names to model label ids, accounting for the background offset.

    torchvision reserves label 0 for background, so a checkpoint trained with the
    +1 id shift (see RocaDataset) has ``num_classes == len(labels) + 1`` and
    predicts labels 1..N; a legacy checkpoint without the shift has
    ``num_classes == len(labels)`` and predicts labels 0..N-1. Returns
    ``[{"id", "name"}]`` with ids in the model's label space, so a predicted
    label maps directly to its class name either way. The result is what
    :class:`CadModelLibrary` expects as ``categories``.
    """
    offset = num_classes - len(labels)
    if offset not in (0, 1):
        raise ValueError(
            f"Checkpoint has {num_classes} classes but the class list has {len(labels)} "
            f"— expected an equal count or exactly one more (background)."
        )
    return [{"id": i + offset, "name": name} for i, name in enumerate(labels)]


class CadModelLibrary:
    """Loads the per-category CAD meshes once and serves them by category id.

    OBJ files are resolved by matching each category name to a file in
    ``cad_dir`` via :func:`normalize_class_name`, so no hard-coded filename
    table is needed.
    """

    def __init__(self, cad_dir: str, categories: List[Dict]):
        self.id_to_name = {c["id"]: c["name"] for c in categories}
        index = index_cad_directory(cad_dir)
        self._meshes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for cat_id, name in self.id_to_name.items():
            path = index.get(normalize_class_name(name))
            if path is None:
                raise KeyError(f"no OBJ file in '{cad_dir}' matches category '{name}'")
            self._meshes[cat_id] = load_obj(path)

    def mesh(self, category_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._meshes[category_id]

    def centroid(self, category_id: int) -> torch.Tensor:
        return bbox_center(self._meshes[category_id][0])

    def name(self, category_id: int) -> str:
        return self.id_to_name.get(category_id, f"category_{category_id}")


def pose_instances(predictions: Dict, library: "CadModelLibrary") -> List[Dict]:
    """Build :func:`render_instances` instance dicts from model predictions.

    ``predictions`` is the dict from ``extract_predictions`` (CPU tensors:
    boxes [N,4], scores [N], labels [N], rotations [N,3,3], translations [N,3]).
    Each predicted pose places the known CAD mesh of its category in the camera
    frame via ``verts_cam = verts @ R.T + t``.
    """
    instances: List[Dict] = []
    for i in range(predictions["boxes"].shape[0]):
        label = int(predictions["labels"][i])
        verts, faces = library.mesh(label)
        verts_cam = verts @ predictions["rotations"][i].T + predictions["translations"][i]
        instances.append({
            "verts_cam": verts_cam,
            "faces": faces,
            "color": CATEGORY_COLORS[label % len(CATEGORY_COLORS)],
            "caption": f"{library.name(label)} {float(predictions['scores'][i]):.2f}",
        })
    return instances


def render_pose_overlay(
    image_rgb: np.ndarray,
    predictions: Dict,
    library: "CadModelLibrary",
    intrinsics,
    alpha: float = 0.6,
) -> np.ndarray:
    """Render predicted CAD poses over an RGB image; returns an RGB image.

    Convenience wrapper for callers working in RGB (the inference server / ROS
    node). Internally converts to BGR for the OpenCV rasteriser (CATEGORY_COLORS
    are BGR) and back. ``intrinsics`` may be a 3x3 numpy array or torch tensor and
    must match ``image_rgb``'s resolution.
    """
    image_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    k = intrinsics.numpy() if torch.is_tensor(intrinsics) else np.asarray(intrinsics, dtype=float)
    out_bgr = render_instances(image_bgr, pose_instances(predictions, library), k, alpha=alpha)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def render_instances(
    image: np.ndarray,
    instances: List[Dict],
    intrinsics: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Render camera-space meshes over `image` with a simple shaded
    painter's-algorithm rasteriser.

    Each instance dict must contain:
      - "verts_cam": [V,3] vertices in the camera frame (torch.Tensor)
      - "faces":     [F,3] triangle indices (torch.Tensor)
      - "color":     (b, g, r) base colour
      - "caption":   optional text label drawn near the top of the mesh
    """
    out = image.copy()
    for inst in instances:
        verts = inst["verts_cam"].numpy()
        faces = inst["faces"].numpy()
        color = np.array(inst["color"], dtype=np.float64)

        in_front = verts[:, 2] > MIN_DEPTH_M
        uv = np.zeros((verts.shape[0], 2), dtype=np.float64)
        proj = verts[in_front] @ intrinsics.T
        uv[in_front] = proj[:, :2] / proj[:, 2:3]

        # Keep only triangles whose vertices are all in front of the camera.
        face_valid = in_front[faces].all(axis=1)
        faces = faces[face_valid]
        if faces.shape[0] == 0:
            continue

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normals, axis=1)
        # Lambert-ish shading on |n_z|, with an ambient floor.
        shade = np.where(norm_len > 0, np.abs(normals[:, 2]) / np.maximum(norm_len, 1e-9), 1.0)
        shade = 0.4 + 0.6 * shade

        # Painter's algorithm: draw far triangles first.
        order = np.argsort(-(v0[:, 2] + v1[:, 2] + v2[:, 2]))
        overlay = out.copy()
        face_uv = np.round(uv[faces]).astype(np.int32)
        for f in order:
            cv2.fillPoly(overlay, [face_uv[f]], tuple(color * shade[f]))
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)

        caption = inst.get("caption")
        if caption:
            anchor_uv = uv[faces.reshape(-1)]
            x = int(anchor_uv[:, 0].mean())
            y = int(anchor_uv[:, 1].min()) - 4
            x = int(np.clip(x, 0, image.shape[1] - 1))
            y = int(np.clip(y, 10, image.shape[0] - 1))
            cv2.putText(
                out, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
    return out