"""
Geometry and matching helpers for evaluation.
"""

import math
from typing import List, Tuple

import torch
from torchvision.ops import box_iou


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    """
    Greedy one-to-one matching of predicted boxes to ground-truth boxes.

    Predictions are processed in descending score order; each one is assigned
    to the unmatched GT box with the highest IoU, provided that IoU reaches
    iou_threshold. Returns a list of (pred_idx, gt_idx, iou) tuples.
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return []

    iou = box_iou(pred_boxes, gt_boxes)
    order = pred_scores.argsort(descending=True).tolist()

    matched_gt = set()
    matches: List[Tuple[int, int, float]] = []
    for pred_idx in order:
        ious = iou[pred_idx].clone()
        for g in matched_gt:
            ious[g] = -1.0
        best_iou, best_gt = ious.max(dim=0)
        best_iou = best_iou.item()
        best_gt = best_gt.item()
        if best_iou >= iou_threshold:
            matches.append((pred_idx, best_gt, best_iou))
            matched_gt.add(best_gt)
    return matches


def rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Angular distance between rotation matrices in degrees. Returns [M] tensor."""
    R_diff = R_pred @ R_gt.transpose(-1, -2)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    return torch.acos(cos_angle) * (180.0 / math.pi)


def camera_to_world_from_view_transform(view_transform) -> torch.Tensor:
    """
    Build a 4x4 camera-to-world matrix from an Omniverse Replicator
    ``cameraViewTransform`` (16 floats).

    The Replicator stores the world-to-camera view matrix row-major in USD
    row-vector convention (p_cam = p_world @ V) for an OpenGL-style camera
    (-Z forward, +Y up). The model/dataset use the computer-vision camera
    frame (+Z forward, +Y down), so the GL camera frame is flipped about
    its X axis before inverting.
    """
    V = torch.tensor(view_transform, dtype=torch.float64).reshape(4, 4).T
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
    return torch.linalg.inv(flip @ V).float()


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 rigid transform to [N, 3] points. Returns [N, 3]."""
    return points @ T[:3, :3].T + T[:3, 3]
