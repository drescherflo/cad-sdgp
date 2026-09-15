"""CAD model loading and lightweight overlay rendering.

A shared, dependency-light utility (stdlib + cv2 + numpy + torch only) used by
both the evaluation pipeline (``src/evaluation``) and the deployment inference
server (``roca_inference_server.py``) to load per-category CAD meshes and render
predicted/ground-truth poses over an image.
"""

from src.cad.mesh import (
    CATEGORY_COLORS,
    CadModelLibrary,
    bbox_center,
    build_categories,
    index_cad_directory,
    load_obj,
    normalize_class_name,
    pose_instances,
    render_instances,
    render_pose_overlay,
)

__all__ = [
    "CATEGORY_COLORS",
    "CadModelLibrary",
    "bbox_center",
    "build_categories",
    "index_cad_directory",
    "load_obj",
    "normalize_class_name",
    "pose_instances",
    "render_instances",
    "render_pose_overlay",
]