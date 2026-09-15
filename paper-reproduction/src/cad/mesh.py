"""CAD mesh loading: parse OBJ files and index a directory of per-category meshes."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch


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
