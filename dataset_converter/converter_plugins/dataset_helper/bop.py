import glob
import json
import os

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from . import replicator, roca

SILHOUETTE_MAX_OVERSHOOT = 1.0  # image sizes beyond each border
SILHOUETTE_SUPERSAMPLE = 4
SILHOUETTE_MAX_SUBPIXELS = 4_000_000


def obj_paths_semantic_labels_and_obj_ids(obj_dir: str) -> list[dict]:
    """
    Retrieves paths to OBJ files, their semantic labels, and their BOP object IDs.

    Labels are sorted alphabetically and numbered from 1, so the IDs match ROCA's labelids.txt.

    :param obj_dir: Path to the directory containing the OBJ files.
    :return: A list of dictionaries with paths, semantic labels, and object IDs of the OBJ files.
    """

    obj_paths = glob.glob(os.path.join(obj_dir, "*.obj"))
    semantic_labels = [roca.obj_path_to_semantic_label(obj_path) for obj_path in obj_paths]
    files_and_labels = sorted(zip(obj_paths, semantic_labels), key=lambda x: x[1])

    return [{"obj_file": obj_file, "semantic_label": semantic_label, "obj_id": idx + 1} for
            idx, (obj_file, semantic_label) in enumerate(files_and_labels)]


def camera_params_to_world_to_camera(camera_params: dict) -> np.ndarray:
    """
    Builds the world to OpenCV camera transformation from Replicator camera parameters.

    cameraViewTransform is column-major and in the OpenGL convention, hence the transpose and the
    diag(1, -1, -1, 1) flip to OpenCV.

    :param camera_params: Contents of a camera_params_[nr].json file.
    :return: The 4x4 world to OpenCV camera transformation matrix.
    """

    world_to_gl_camera = np.array(camera_params["cameraViewTransform"]).reshape(4, 4).transpose()
    gl_to_cv_camera = np.diag([1.0, -1.0, -1.0, 1.0])

    return gl_to_cv_camera @ world_to_gl_camera


def world_pose_to_camera_frame(position: dict, orientation: dict, world_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms an object pose from the world frame into the BOP camera frame.

    :param position: Position of the object in world coordinates in meters, with keys x, y and z.
    :param orientation: Orientation of the object in world coordinates, with keys w, x, y and z.
    :param world_to_camera: 4x4 world to OpenCV camera transformation matrix.
    :return: (cam_R_m2c, cam_t_m2c) as a 3x3 rotation matrix and a translation in millimeters.
    """

    # Isaac Sim stores the quaternion scalar first, from_quat expects it last
    rotation_world = Rotation.from_quat(
        [orientation["x"], orientation["y"], orientation["z"], orientation["w"]]).as_matrix()
    translation_world = np.array([position["x"], position["y"], position["z"], 1.0])

    cam_R_m2c = world_to_camera[:3, :3] @ rotation_world
    cam_t_m2c = (world_to_camera @ translation_world)[:3] * 1000.0  # meters -> millimeters

    return cam_R_m2c, cam_t_m2c


def calc_2d_bbox(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    """
    Calculates a BOP bounding box from pixel coordinates.

    :param xs: Array of x coordinates.
    :param ys: Array of y coordinates.
    :return: The bounding box as [x, y, width, height].
    """

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def union_2d_bbox(first: list[int], second: list[int]) -> list[int]:
    """
    Calculates the smallest bounding box containing two bounding boxes.

    :param first: The first bounding box as [x, y, width, height].
    :param second: The second bounding box as [x, y, width, height].
    :return: The enclosing bounding box as [x, y, width, height].
    """

    x = min(first[0], second[0])
    y = min(first[1], second[1])
    width = max(first[0] + first[2], second[0] + second[2]) - x
    height = max(first[1] + first[3], second[1] + second[3]) - y

    return [x, y, width, height]


def rasterize_silhouette(vertices: np.ndarray, faces: np.ndarray, cam_R_m2c: np.ndarray,
                         cam_t_m2c: np.ndarray, cam_K: np.ndarray,
                         im_size: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """
    Rasterizes the amodal silhouette of a mesh in a given pose, into a window around the object.

    Fills one triangle at a time: cv2.fillPoly and cv2.drawContours apply an even-odd rule across all
    contours, so overlapping triangles would cancel out and leave holes.

    :param vertices: Mesh vertices as an (V, 3) array in the model frame in millimeters.
    :param faces: Mesh faces as an (F, 3) array of vertex indices.
    :param cam_R_m2c: 3x3 model to camera rotation matrix.
    :param cam_t_m2c: Model to camera translation in millimeters.
    :param cam_K: 3x3 camera matrix.
    :param im_size: Image size as (width, height).
    :return: (silhouette, area, x_origin, y_origin) with the silhouette as a boolean mask at image
             resolution whose top left pixel lies at image coordinate (x_origin, y_origin), and the
             subpixel accurate area of the silhouette in whole pixels.
    """

    width, height = im_size
    empty = (np.zeros((0, 0), dtype=bool), 0.0, 0, 0)

    # Triangles at or behind the image plane cannot be projected
    points = vertices @ cam_R_m2c.T + cam_t_m2c
    in_front = points[:, 2] > 1e-6
    faces = faces[in_front[faces].all(axis=1)]
    if len(faces) == 0:
        return empty

    projected = points @ cam_K.T
    projected = projected[:, :2] / projected[:, 2:3]

    # Window around the object, capped at the allowed overshoot
    limit_x = SILHOUETTE_MAX_OVERSHOOT * width
    limit_y = SILHOUETTE_MAX_OVERSHOOT * height
    used = projected[np.unique(faces)]
    x_origin = int(np.floor(max(used[:, 0].min(), -limit_x)))
    y_origin = int(np.floor(max(used[:, 1].min(), -limit_y)))
    x_end = int(np.ceil(min(used[:, 0].max(), width + limit_x)))
    y_end = int(np.ceil(min(used[:, 1].max(), height + limit_y)))
    window_width = x_end - x_origin + 1
    window_height = y_end - y_origin + 1
    if window_width <= 0 or window_height <= 0:
        return empty

    # Rather than allocate an unbounded buffer for a huge silhouette
    supersample = SILHOUETTE_SUPERSAMPLE
    while supersample > 1 and window_width * window_height * supersample ** 2 > SILHOUETTE_MAX_SUBPIXELS:
        supersample -= 1

    canvas = np.zeros((window_height * supersample, window_width * supersample), dtype=np.uint8)
    projected = (projected - [x_origin, y_origin]) * supersample

    # Clamp before casting to int32; fillConvexPoly clips to the canvas anyway
    np.clip(projected, -canvas.shape[1], 2 * canvas.shape[1], out=projected)
    for triangle in np.round(projected[faces]).astype(np.int32):
        cv2.fillConvexPoly(canvas, triangle, 1)

    # Any coverage keeps the mask a superset. The area has to be subpixel accurate because
    # px_count_visib comes from Replicator's renderer while px_count_all comes from this rasterizer:
    # a binary fill counts every touched boundary pixel in full, which inflates px_count_all by
    # roughly a perimeter and drops visib_fract of a fully visible object to about 0.92.
    coverage = canvas.reshape(window_height, supersample, window_width, supersample).sum(axis=(1, 3))

    return coverage > 0, float(coverage.sum()) / supersample ** 2, x_origin, y_origin


def silhouette_in_image(silhouette: np.ndarray, x_origin: int, y_origin: int,
                        im_size: tuple[int, int]) -> np.ndarray:
    """
    Places the part of a silhouette that falls inside the image into an image sized mask.

    :param silhouette: Boolean silhouette mask as returned by rasterize_silhouette.
    :param x_origin: Image x coordinate of the top left pixel of the silhouette.
    :param y_origin: Image y coordinate of the top left pixel of the silhouette.
    :param im_size: Image size as (width, height).
    :return: The silhouette clipped to the image, as an image sized boolean mask.
    """

    width, height = im_size
    mask = np.zeros((height, width), dtype=bool)
    if silhouette.size == 0:
        return mask

    x_start, y_start = max(x_origin, 0), max(y_origin, 0)
    x_stop = min(x_origin + silhouette.shape[1], width)
    y_stop = min(y_origin + silhouette.shape[0], height)
    if x_start >= x_stop or y_start >= y_stop:
        return mask

    mask[y_start:y_stop, x_start:x_stop] = silhouette[
        y_start - y_origin:y_stop - y_origin, x_start - x_origin:x_stop - x_origin]

    return mask


def binary_mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encodes a binary mask as COCO's uncompressed run length encoding.

    :param mask: Boolean mask of the object.
    :return: The mask in COCO RLE format, with the counts and the mask shape.
    """

    flat = mask.ravel(order="F").astype(np.uint8)
    counts = [0] if flat[0] == 1 else []

    # Segment boundaries, plus the two ends, give the run lengths as their differences
    changes = np.where(np.concatenate(([True], flat[:-1] != flat[1:], [True])))[0]
    counts.extend(np.diff(changes).tolist())

    return {"counts": counts, "size": list(mask.shape)}


def coco_annotation(annotation_id: int, im_id: int, obj_id: int, mask_visib: np.ndarray,
                    bbox: list[int], ignore: bool) -> dict:
    """
    Builds the COCO annotation of a single object, following bop_toolkit create_annotation_info.

    :param annotation_id: Index of the annotation within its scene, counted from one.
    :param im_id: BOP image ID.
    :param obj_id: BOP object ID.
    :param mask_visib: Visible mask of the object.
    :param bbox: Amodal bounding box as [x, y, width, height].
    :param ignore: Whether the evaluation should skip this annotation.
    :return: The COCO annotation entry.
    """

    return {
        "id": annotation_id,
        "image_id": im_id,
        "category_id": obj_id,
        "iscrowd": 0,
        "area": int(mask_visib.sum()),
        "bbox": bbox,
        "segmentation": binary_mask_to_rle(mask_visib),
        "width": mask_visib.shape[1],
        "height": mask_visib.shape[0],
        "ignore": ignore,
    }


def calc_diameter(vertices: np.ndarray) -> float:
    """
    Calculates the diameter of a model, i.e. the largest distance between any two of its points.

    Only convex hull points can attain it, which keeps the pairwise distances tractable.

    :param vertices: Mesh vertices as an (V, 3) array.
    :return: The diameter in the unit of the vertices.
    """

    if len(vertices) > 3:
        hull_vertices = vertices[ConvexHull(vertices).vertices]
    else:
        hull_vertices = vertices

    differences = hull_vertices[:, None, :] - hull_vertices[None, :, :]

    return float(np.sqrt((differences ** 2).sum(axis=-1)).max())


def write_models(obj_paths_labels_ids: list[dict], models_dir: str) -> None:
    """
    Converts the CAD models to BOP model files and writes models_info.json.

    The vertices are only scaled from meters to millimeters. No rotation, because the OBJ vertex
    frame is already the USD prim frame, and no re-centering, because cam_t_m2c is the CAD origin.

    :param obj_paths_labels_ids: OBJ paths, semantic labels and object IDs of the models.
    :param models_dir: Directory the PLY files and models_info.json are written to.
    """

    os.makedirs(models_dir, exist_ok=True)

    models_info = {}
    for obj_path_label_id in obj_paths_labels_ids:
        obj_id = obj_path_label_id["obj_id"]
        semantic_label = obj_path_label_id["semantic_label"]

        mesh = trimesh.load(obj_path_label_id["obj_file"], force="mesh")
        mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) * 1000.0, faces=mesh.faces, process=False)
        mesh.export(os.path.join(models_dir, f"obj_{obj_id:06d}.ply"))

        minimum, maximum = mesh.bounds
        diameter = calc_diameter(np.asarray(mesh.vertices))
        model_info = {
            "diameter": diameter,
            "min_x": float(minimum[0]), "min_y": float(minimum[1]), "min_z": float(minimum[2]),
            "size_x": float(maximum[0] - minimum[0]),
            "size_y": float(maximum[1] - minimum[1]),
            "size_z": float(maximum[2] - minimum[2]),
        }

        models_info[str(obj_id)] = model_info

        print(f"  obj_{obj_id:06d} {semantic_label}: diameter {diameter:.1f} mm")

    with open(os.path.join(models_dir, "models_info.json"), "w") as models_info_file:
        json.dump(models_info, models_info_file, indent=2)
