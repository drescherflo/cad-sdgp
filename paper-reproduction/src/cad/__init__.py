"""CAD model loading, used to project ground-truth boxes and compute object centroids during evaluation."""

from src.cad.mesh import (
    bbox_center,
    index_cad_directory,
    load_obj,
    normalize_class_name,
)

__all__ = [
    "bbox_center",
    "index_cad_directory",
    "load_obj",
    "normalize_class_name",
]
