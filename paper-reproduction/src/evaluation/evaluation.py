"""
Evaluate ROCA raw-data predictions against the ReplicatorToRocaEval ground truth.

Reads the ``eval_raw_data.json`` files produced by
``generate_evaluation_raw_data.py`` and, per frame, matches detections to the
ground-truth objects, then aggregates pose/classification/detection metrics
across datasets, materials, object types and dataset types and renders the
corresponding plots, CSVs and a ``summary.json``.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from src.cad import bbox_center, index_cad_directory, load_obj, normalize_class_name
from src.evaluation.metrics import (
    camera_to_world_from_view_transform,
    match_predictions,
    rotation_error_deg,
)

MIN_DEPTH_M = 0.05


# ---------------------------------------------------------------------------
# Small geometry helpers (numpy)
# ---------------------------------------------------------------------------

def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z]."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def read_intrinsics(path: str) -> np.ndarray:
    with open(path, "r") as f:
        rows = [line.strip().split() for line in f if line.strip()]
    return np.array(rows, dtype=np.float64)


def project_to_box(
    verts_cam: np.ndarray, K: np.ndarray, img_w: int, img_h: int
) -> Optional[List[float]]:
    """Project camera-frame vertices and return their clipped 2D bbox [x1,y1,x2,y2].

    Returns None if fewer than three vertices are in front of the camera or the
    resulting box is degenerate after clipping to the image bounds.
    """
    front = verts_cam[:, 2] > MIN_DEPTH_M
    if int(front.sum()) < 3:
        return None
    proj = verts_cam[front] @ K.T
    uv = proj[:, :2] / proj[:, 2:3]
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    x1 = float(np.clip(x1, 0, img_w))
    x2 = float(np.clip(x2, 0, img_w))
    y1 = float(np.clip(y1, 0, img_h))
    y2 = float(np.clip(y2, 0, img_h))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]


def _nanmean(values) -> float:
    return float(np.nanmean(values)) if len(values) else float("nan")


def _nanstd(values) -> float:
    return float(np.nanstd(values)) if len(values) else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


# ---------------------------------------------------------------------------
# CAD meshes keyed by class name
# ---------------------------------------------------------------------------

class CadMeshes:
    """Loads CAD meshes once and serves vertices/centroid by class name.

    OBJ files are discovered in ``cad_dir`` and keyed by their normalized name,
    so any class name appearing in the data (GT or predictions) is resolved by
    matching against the available files (no hard-coded filename table).
    """

    def __init__(self, cad_dir: str):
        self._meshes: Dict[str, np.ndarray] = {}
        self._centroids: Dict[str, np.ndarray] = {}
        for key, path in index_cad_directory(cad_dir).items():
            verts, _faces = load_obj(path)
            self._meshes[key] = verts.numpy().astype(np.float64)
            self._centroids[key] = bbox_center(verts).numpy().astype(np.float64)

    def verts(self, name: str) -> np.ndarray:
        return self._meshes[normalize_class_name(name)]

    def centroid(self, name: str) -> np.ndarray:
        return self._centroids[normalize_class_name(name)]

    def has(self, name: str) -> bool:
        return normalize_class_name(name) in self._meshes


# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------

def evaluate_frame(
    instances: List[Dict],
    gt_objects: List[Dict],
    cam_to_world: np.ndarray,
    K: np.ndarray,
    meshes: CadMeshes,
    img_w: int,
    img_h: int,
    iou_threshold: float,
) -> Dict:
    """Match a frame's detections to GT objects and compute per-match errors.

    Detection boxes are always projected from each prediction's pose: ROCA raw
    data never carries a stored box.
    """
    world_to_cam = np.linalg.inv(cam_to_world)
    R_cw = cam_to_world[:3, :3]

    # --- Ground truth: 2D boxes (projected CAD) + world centroid + world rotation
    gt_boxes: List[List[float]] = []
    gt_centroids_world: List[np.ndarray] = []
    gt_quats_world: List[np.ndarray] = []
    gt_names: List[str] = []
    gt_occlusions: List[float] = []
    for g in gt_objects:
        name = g["semantic_labels"]["class"]
        if not meshes.has(name):
            continue
        pos = g["pose"]["position"]
        ori = g["pose"]["orientation"]
        t_wo = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
        q_wo = np.array([ori["w"], ori["x"], ori["y"], ori["z"]], dtype=np.float64)
        R_wo = quaternion_to_matrix(q_wo)
        verts_world = meshes.verts(name) @ R_wo.T + t_wo
        verts_cam = (world_to_cam[:3, :3] @ verts_world.T).T + world_to_cam[:3, 3]
        box = project_to_box(verts_cam, K, img_w, img_h)
        if box is None:
            continue
        gt_boxes.append(box)
        gt_centroids_world.append(R_wo @ meshes.centroid(name) + t_wo)
        gt_quats_world.append(q_wo)
        gt_names.append(name)
        gt_occlusions.append(float(g.get("occlusion_ratio", 0.0)))

    # --- Predictions: 2D boxes (stored) + world centroid + world rotation
    pred_boxes: List[List[float]] = []
    pred_scores: List[float] = []
    pred_centroids_world: List[np.ndarray] = []
    pred_quats_world: List[np.ndarray] = []
    pred_names: List[str] = []
    for inst in instances:
        name = inst["semantic_label"]
        if not meshes.has(name):
            continue
        q_co = np.array(inst["rotation"], dtype=np.float64)
        t_co = np.array(inst["translation"], dtype=np.float64)
        R_co = quaternion_to_matrix(q_co)
        scale = np.asarray(inst.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        centroid_cam = R_co @ (meshes.centroid(name) * scale) + t_co
        centroid_world = R_cw @ centroid_cam + cam_to_world[:3, 3]

        # Synthesise the box by projecting the predicted CAD mesh at its
        # predicted camera-frame pose. Note this is a pose-derived box, not
        # the model's real detection box.
        verts_cam = (R_co @ (meshes.verts(name) * scale).T).T + t_co
        box = project_to_box(verts_cam, K, img_w, img_h)
        if box is None:
            continue  # pose projects out of view -> cannot be matched
        pred_boxes.append(box)
        pred_scores.append(float(inst.get("score", 1.0)))
        pred_centroids_world.append(centroid_world)
        pred_quats_world.append(matrix_to_quaternion(R_cw @ R_co))
        pred_names.append(name)

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    matches = match_predictions(
        torch.tensor(pred_boxes, dtype=torch.float32) if n_pred else torch.zeros(0, 4),
        torch.tensor(pred_scores, dtype=torch.float32) if n_pred else torch.zeros(0),
        torch.tensor(gt_boxes, dtype=torch.float32) if n_gt else torch.zeros(0, 4),
        iou_threshold,
    )

    distance_errors: List[float] = []
    rotation_errors_deg: List[float] = []
    rotation_errors_huynh: List[float] = []
    scale_errors: List[float] = []
    correct = incorrect = 0
    matched_gt = set()
    for pred_idx, gt_idx, _iou in matches:
        matched_gt.add(gt_idx)
        distance_errors.append(float(np.linalg.norm(pred_centroids_world[pred_idx] - gt_centroids_world[gt_idx])))
        R_p = torch.tensor(quaternion_to_matrix(pred_quats_world[pred_idx]), dtype=torch.float32)
        R_g = torch.tensor(quaternion_to_matrix(gt_quats_world[gt_idx]), dtype=torch.float32)
        rotation_errors_deg.append(float(rotation_error_deg(R_p, R_g)))
        rotation_errors_huynh.append(1.0 - float(abs(np.dot(pred_quats_world[pred_idx], gt_quats_world[gt_idx]))))
        scale_errors.append(0.0)  # this model does not predict scale (scale == 1)
        if pred_names[pred_idx] == gt_names[gt_idx]:
            correct += 1
        else:
            incorrect += 1

    tp = len(matches)
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - len(matched_gt),
        "distance_errors": distance_errors,
        "rotation_errors_deg": rotation_errors_deg,
        "rotation_errors_huynh": rotation_errors_huynh,
        "scale_errors": scale_errors,
        "correct_classifications": correct,
        "incorrect_classifications": incorrect,
        "undetected_occlusions": [gt_occlusions[i] for i in range(n_gt) if i not in matched_gt],
    }


# ---------------------------------------------------------------------------
# Dataset walking + aggregation
# ---------------------------------------------------------------------------

# Object-level error metrics pooled across frames (lower is better).
OBJECT_METRICS = [
    ("distance_errors", "Distanzfehler [m]", False, 3),
    ("rotation_errors_deg", "Rotationsfehler [°]", False, 1),
    ("rotation_errors_huynh", "Rotationsfehler (1-|q·q|)", False, 3),
    ("scale_errors", "Skalierungsfehler", False, 4),
    ("undetected_occlusions", "Verdeckungsgrad unentdeckter Objekte", True, 2),
]
# Frame-level ratio metrics (in [0, 1] except found_ratio).
FRAME_METRICS = [
    ("found_ratio", "Anteil gefundener Objekte", True, 2),
    ("recall", "Detektionsrate (Recall)", True, 2),
    ("precision", "Präzision", True, 2),
    ("classification_ratio", "Anteil korrekter Klassifizierungen", True, 2),
]


def _empty_pool() -> Dict[str, list]:
    return {key: [] for key, *_ in OBJECT_METRICS + FRAME_METRICS}


def evaluate_run(eval_dataset_path: str, meshes: CadMeshes, iou_threshold: float,
                 exclude_suffixes: Optional[List[str]] = None) -> Dict:
    """Walk converted/{type}/{dataset}/ReplicatorToRocaEval/eval_raw_data/{net}.

    Dataset directories whose name ends with one of ``exclude_suffixes`` are
    skipped entirely (e.g. to drop known-uninteresting material variants).

    Returns nested results keyed by dataset_type -> dataset -> net.
    """
    results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for dataset_type in ("cluttered", "uncluttered"):
        type_root = os.path.join(eval_dataset_path, "converted", dataset_type)
        if not os.path.isdir(type_root):
            continue
        results[dataset_type] = {}
        for dataset_dir in sorted(glob.glob(os.path.join(type_root, "*"))):
            dataset = os.path.basename(dataset_dir)
            if exclude_suffixes and dataset.endswith(tuple(exclude_suffixes)):
                continue
            gt_dir = os.path.join(dataset_dir, "ReplicatorToRocaEval")
            net_dirs = sorted(glob.glob(os.path.join(gt_dir, "eval_raw_data", "*")))
            if not net_dirs:
                continue
            results[dataset_type][dataset] = {}
            for net_dir in net_dirs:
                net = os.path.basename(net_dir)
                raw_json = os.path.join(net_dir, "eval_raw_data.json")
                if not os.path.exists(raw_json):
                    continue
                print(f"Processing {dataset_type}/{dataset}/{net}")
                results[dataset_type][dataset][net] = _evaluate_net(
                    raw_json, gt_dir, meshes, iou_threshold
                )
    return results


def _evaluate_net(raw_json: str, gt_dir: str, meshes: CadMeshes, iou_threshold: float) -> dict:
    with open(raw_json) as f:
        raw = json.load(f)

    rows = []
    pool = _empty_pool()
    for frame in raw["per_frame_results"]:
        num = frame["image"].removesuffix(".jpg").removeprefix("image_")
        gt_path = os.path.join(gt_dir, f"world_pose_visible_objects_{num}.json")
        cam_path = os.path.join(gt_dir, f"camera_params_{num}.json")
        intr_path = os.path.join(gt_dir, f"image_{num}.txt")
        if not (os.path.exists(gt_path) and os.path.exists(cam_path) and os.path.exists(intr_path)):
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        with open(cam_path) as f:
            cam = json.load(f)
        cam_to_world = camera_to_world_from_view_transform(cam["cameraViewTransform"]).numpy().astype(np.float64)
        K = read_intrinsics(intr_path)
        img_w, img_h = int(cam["renderProductResolution"][0]), int(cam["renderProductResolution"][1])

        m = evaluate_frame(frame["instances"], gt, cam_to_world, K, meshes, img_w, img_h,
                           iou_threshold)
        inference_time = frame["inference_end_time"] - frame["inference_start_time"]

        found_ratio = _ratio(m["n_pred"], m["n_gt"])
        recall = _ratio(m["tp"], m["n_gt"])
        precision = _ratio(m["tp"], m["tp"] + m["fp"])
        classification_ratio = _ratio(m["correct_classifications"], m["correct_classifications"] + m["incorrect_classifications"])

        # Pool object-level errors (extend) and frame-level ratios.
        for key, *_ in OBJECT_METRICS:
            pool[key].extend(m[key])
        pool["found_ratio"].append(found_ratio)
        pool["recall"].append(recall)
        pool["precision"].append(precision)
        pool["classification_ratio"].append(classification_ratio)

        rows.append({
            "image": frame["image"],
            "gt_count": m["n_gt"],
            "pred_count": m["n_pred"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "inference_time_s": inference_time,
            "found_ratio": found_ratio,
            "recall": recall,
            "precision": precision,
            "classification_ratio": classification_ratio,
            "distance_mean": _nanmean(m["distance_errors"]),
            "distance_std": _nanstd(m["distance_errors"]),
            "distance_min": float(np.min(m["distance_errors"])) if m["distance_errors"] else float("nan"),
            "distance_max": float(np.max(m["distance_errors"])) if m["distance_errors"] else float("nan"),
            "rotation_deg_mean": _nanmean(m["rotation_errors_deg"]),
            "rotation_deg_std": _nanstd(m["rotation_errors_deg"]),
            "rotation_deg_min": float(np.min(m["rotation_errors_deg"])) if m["rotation_errors_deg"] else float("nan"),
            "rotation_deg_max": float(np.max(m["rotation_errors_deg"])) if m["rotation_errors_deg"] else float("nan"),
            "rotation_huynh_mean": _nanmean(m["rotation_errors_huynh"]),
            "occlusion_undetected_mean": _nanmean(m["undetected_occlusions"]),
            "correct_classifications": m["correct_classifications"],
            "incorrect_classifications": m["incorrect_classifications"],
        })

    return {"df": pd.DataFrame(rows), "pool": pool}


def _merge_pools(pools: List[Dict[str, list]]) -> Dict[str, list]:
    merged = _empty_pool()
    for pool in pools:
        for key in merged:
            merged[key].extend(pool[key])
    return merged


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _autolabel(rects, ax, percent: bool, round_decimals: int):
    for rect in rects:
        height = rect.get_height()
        if not np.isfinite(height):
            continue
        text = f"{height:.0%}" if percent else f"{height:.{round_decimals}f}"
        ax.annotate(text, xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6)


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def line_plot(path, title, ylabel, series: Dict[str, np.ndarray], bands: Optional[Dict] = None,
              percent: bool = False, ylim=None):
    fig, ax = plt.subplots(dpi=300)
    for label, y in series.items():
        y = np.asarray(y, dtype=float)
        x = np.arange(len(y))
        ax.plot(x, y, label=label)
        if bands and label in bands:
            mean, std = bands[label]
            mean = np.asarray(mean, dtype=float); std = np.asarray(std, dtype=float)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Frame-Nummer")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    if len(series) > 1:
        ax.legend()
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    if ylim:
        ax.set_ylim(ylim)
    _save(fig, path)


def grouped_bar(path, title, ylabel, group_labels, series: Dict[str, Tuple[list, list]],
                percent: bool = False, round_decimals: int = 2):
    fig, ax = plt.subplots(dpi=300)
    n = max(len(series), 1)
    width = 0.9 / n
    x = np.arange(len(group_labels))
    for i, (label, (means, stds)) in enumerate(series.items()):
        rects = ax.bar(x + i * width, means, width, label=label, yerr=stds,
                       error_kw=dict(ecolor="lightgray", lw=1, capsize=3))
        _autolabel(rects, ax, percent, round_decimals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(group_labels, rotation=90)
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.legend()
    _save(fig, path)


def single_bar(path, title, ylabel, labels, means, stds, percent: bool = False, round_decimals: int = 2):
    fig, ax = plt.subplots(dpi=300)
    rects = ax.bar(labels, means, yerr=stds, error_kw=dict(ecolor="lightgray", lw=1, capsize=3))
    _autolabel(rects, ax, percent, round_decimals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=90)
    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    _save(fig, path)


def radar_plot(path, title, axis_names, series: Dict[str, list]):
    n = len(axis_names)
    if n < 3:
        return
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(dpi=300, subplot_kw=dict(polar=True))
    for label, values in series.items():
        vals = list(values) + [values[0]]
        ax.plot(angles, vals, label=label)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    _save(fig, path)


def regression_plot(path, title, x_vals, y_vals):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(x, y, s=8, alpha=0.4)
    subtitle = ""
    if len(x) >= 2 and np.ptp(x) > 0:
        lr = stats.linregress(x, y)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, lr.intercept + lr.slope * xs, color="red")
        subtitle = f"\nr={lr.rvalue:.3f}, Steigung={lr.slope:.4f} s/Objekt"
    ax.set_title(title + subtitle)
    ax.set_xlabel("Anzahl detektierter Objekte")
    ax.set_ylabel("Inferenz-Zeit [s]")
    ax.grid(True)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Plotting orchestration
# ---------------------------------------------------------------------------

def _plot_per_net(out_dir, df: pd.DataFrame):
    """Per-frame line plots for a single (dataset, net)."""
    line_plot(os.path.join(out_dir, "inference_time.pdf"), "Inferenz-Zeit pro Frame",
              "Inferenz-Zeit [s]", {"Inferenz-Zeit": df["inference_time_s"].to_numpy()})
    line_plot(os.path.join(out_dir, "detected_objects.pdf"), "Objekte pro Frame", "Anzahl Objekte",
              {"Sichtbar (GT)": df["gt_count"].to_numpy(), "Detektiert": df["pred_count"].to_numpy()})
    line_plot(os.path.join(out_dir, "distance_error.pdf"), "Distanzfehler pro Frame", "Distanzfehler [m]",
              {"Mittel": df["distance_mean"].to_numpy(), "Max": df["distance_max"].to_numpy(),
               "Min": df["distance_min"].to_numpy()},
              bands={"Mittel": (df["distance_mean"].to_numpy(), df["distance_std"].to_numpy())})
    line_plot(os.path.join(out_dir, "rotation_error_deg.pdf"), "Rotationsfehler pro Frame", "Rotationsfehler [°]",
              {"Mittel": df["rotation_deg_mean"].to_numpy(), "Max": df["rotation_deg_max"].to_numpy(),
               "Min": df["rotation_deg_min"].to_numpy()},
              bands={"Mittel": (df["rotation_deg_mean"].to_numpy(), df["rotation_deg_std"].to_numpy())})
    line_plot(os.path.join(out_dir, "classification_ratio.pdf"), "Anteil korrekter Klassifizierungen pro Frame",
              "Anteil", {"Korrekt": df["classification_ratio"].to_numpy()}, percent=True, ylim=(-0.05, 1.2))
    line_plot(os.path.join(out_dir, "occlusion_undetected.pdf"), "Verdeckungsgrad unentdeckter Objekte pro Frame",
              "Verdeckungsgrad", {"Mittel": df["occlusion_undetected_mean"].to_numpy()}, percent=True, ylim=(-0.05, 1.2))


def _plot_per_dataset(out_dir, nets: Dict[str, dict]):
    """Compare nets within one dataset (per-frame lines)."""
    line_plot(os.path.join(out_dir, "inference_time.pdf"), "Inferenz-Zeit pro Frame", "Inferenz-Zeit [s]",
              {net: r["df"]["inference_time_s"].to_numpy() for net, r in nets.items()})
    for col, ylabel, fname in [("distance_mean", "Distanzfehler [m]", "distance_error_mean.pdf"),
                               ("rotation_deg_mean", "Rotationsfehler [°]", "rotation_error_deg_mean.pdf"),
                               ("recall", "Detektionsrate", "recall.pdf"),
                               ("precision", "Präzision", "precision.pdf")]:
        percent = col in ("recall", "precision")
        line_plot(os.path.join(out_dir, fname), ylabel + " pro Frame", ylabel,
                  {net: r["df"][col].to_numpy() for net, r in nets.items()}, percent=percent)


def _bar_levels(out_dir, prefix, group_labels, group_pools: Dict[str, Dict[str, Dict[str, list]]], nets: List[str]):
    """Grouped bars: one group per material/object-type, bars = nets, for every metric."""
    for key, ylabel, percent, dec in OBJECT_METRICS + FRAME_METRICS:
        series = {}
        for net in nets:
            means, stds = [], []
            for grp in group_labels:
                pool = group_pools[grp].get(net)
                if pool is None:
                    means.append(np.nan); stds.append(np.nan)
                else:
                    means.append(_nanmean(pool[key])); stds.append(_nanstd(pool[key]))
            series[net.removesuffix("_Augmentation")] = (means, stds)
        grouped_bar(os.path.join(out_dir, f"{prefix}-{key}.pdf"),
                    f"{ylabel} nach {prefix}", ylabel, group_labels, series, percent=percent, round_decimals=dec)


def _bars_over_nets(out_dir, fname_prefix, net_pools: Dict[str, Dict[str, list]]):
    """Single bars over nets (dataset-type / all levels) for every metric + radar + regression inputs."""
    nets = sorted(net_pools.keys())
    labels = [n.removesuffix("_Augmentation") for n in nets]
    radar_axes, radar_series = [], {n: [] for n in nets}
    for key, ylabel, percent, dec in OBJECT_METRICS + FRAME_METRICS:
        means = [_nanmean(net_pools[n][key]) for n in nets]
        stds = [_nanstd(net_pools[n][key]) for n in nets]
        single_bar(os.path.join(out_dir, f"{fname_prefix}-{key}.pdf"), ylabel, ylabel, labels, means, stds,
                   percent=percent, round_decimals=dec)
        # radar: normalise to [0,1] "goodness" (higher = better)
        vals = np.array(means, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            hi = finite.max() if finite.max() > 0 else 1.0
            norm = np.clip(vals / hi, 0, 1)
            goodness = norm if key in ("recall", "precision", "found_ratio", "classification_ratio") else 1 - norm
            radar_axes.append(ylabel)
            for n, g in zip(nets, goodness):
                radar_series[n].append(float(g) if np.isfinite(g) else 0.0)
    radar_plot(os.path.join(out_dir, f"{fname_prefix}-radar.pdf"), "Metrik-Übersicht (höher = besser)",
               radar_axes, {n.removesuffix("_Augmentation"): v for n, v in radar_series.items()})


def _material_of(dataset: str) -> str:
    return dataset.split("_")[-1]


def _object_type_of(dataset: str) -> str:
    return "_".join(dataset.split("_")[:-2])


def summarize_and_plot(results: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    summary: Dict[str, dict] = {}

    for dataset_type, datasets in results.items():
        type_dir = os.path.join(output_dir, dataset_type)
        os.makedirs(type_dir, exist_ok=True)

        all_nets = sorted({net for ds in datasets.values() for net in ds})

        # --- per-net and per-dataset ---
        for dataset, nets in datasets.items():
            for net, r in nets.items():
                net_dir = os.path.join(type_dir, dataset, net)
                os.makedirs(net_dir, exist_ok=True)
                r["df"].to_csv(os.path.join(net_dir, f"{dataset}-{net}.csv"), index=False, sep=";")
                _plot_per_net(net_dir, r["df"])
            _plot_per_dataset(os.path.join(type_dir, dataset), nets)

        # --- per material / per object type (grouped bars) ---
        for grouping, key_fn in (("material", _material_of), ("object-type", _object_type_of)):
            groups = sorted({key_fn(ds) for ds in datasets})
            group_pools: Dict[str, Dict[str, Dict[str, list]]] = {}
            for grp in groups:
                group_pools[grp] = {}
                for net in all_nets:
                    pools = [nets[net]["pool"] for ds, nets in datasets.items()
                             if key_fn(ds) == grp and net in nets]
                    if pools:
                        group_pools[grp][net] = _merge_pools(pools)
            _bar_levels(type_dir, grouping, groups, group_pools, all_nets)

        # --- per dataset type (bars over nets + radar) ---
        type_pools = {net: _merge_pools([nets[net]["pool"] for nets in datasets.values() if net in nets])
                      for net in all_nets}
        _bars_over_nets(type_dir, "dataset-type", type_pools)

        # inference regression (pooled over all frames of this type)
        x = [c for nets in datasets.values() for r in nets.values() for c in r["df"]["pred_count"]]
        y = [t for nets in datasets.values() for r in nets.values() for t in r["df"]["inference_time_s"]]
        regression_plot(os.path.join(type_dir, "inference_regression.pdf"),
                        "Inferenz-Zeit vs. Objektanzahl", x, y)

        # summary per net for this dataset type
        for net, pool in type_pools.items():
            summary.setdefault(net, {})[dataset_type] = _net_summary(pool)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote results and summary.json to {output_dir}")


def _net_summary(pool: Dict[str, list]) -> dict:
    def stat(key):
        vals = [v for v in pool[key] if np.isfinite(v)]
        if not vals:
            return {"mean": None, "median": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "std": float(np.std(vals)), "n": len(vals)}
    return {key: stat(key) for key, *_ in OBJECT_METRICS + FRAME_METRICS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dataset-path", required=True, help="Root with converted/{cluttered,uncluttered}/...")
    parser.add_argument("--cad-dir", required=True, help="Directory with the per-category OBJ files")
    parser.add_argument("--output-dir", default="eval_output", help="Where to write CSVs, plots and summary.json")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--exclude-suffix", action="append", default=[], metavar="SUFFIX",
                        help="Skip dataset directories whose name ends with SUFFIX (repeatable)")
    return parser.parse_args()


def main():
    args = parse_args()
    meshes = CadMeshes(args.cad_dir)
    results = evaluate_run(args.eval_dataset_path, meshes, args.iou_threshold, args.exclude_suffix)
    summarize_and_plot(results, args.output_dir)


if __name__ == "__main__":
    main()
