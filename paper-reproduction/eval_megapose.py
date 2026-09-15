#!/usr/bin/env python
"""Run MegaPose over a BOP-format dataset and write BOP result CSVs.

The evaluation entrypoint of the public release does not run, so this drives the pose estimator
directly. Detections are taken from the ground truth, since MegaPose ships no detector for these
objects. Run it from the MegaPose checkout, with its conda environment:

    python eval_megapose.py --method-name megaposebase --out-dir results
    python eval_megapose.py --method-name megaposeft --out-dir results --refiner-run-id ft-refiner

One pass writes both the RGB-only result (refiner-final) and the depth-refined one
(depth-refiner).
"""

# Standard Library
import argparse
import functools
import time
from pathlib import Path

# MegaPose, imported first: it sets the CUDA and EGL environment and pulls in cv2, which has to
# be loaded before torch.
import megapose  # noqa: F401

# Third Party
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# MegaPose
from megapose.config import EXP_DIR, LOCAL_DATA_DIR
from megapose.datasets.datasets_cfg import make_object_dataset, make_scene_dataset
from megapose.datasets.scene_dataset import SceneObservation
from megapose.inference import icp_refiner as icp_refiner_module
from megapose.inference.icp_refiner import ICPRefiner
from megapose.inference.pose_estimator import PoseEstimator
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import load_pose_models
from megapose.utils.load_model import NAMED_MODELS
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

# Third Party
from bop_toolkit_lib import inout  # noqa: E402  (megapose puts it on sys.path)

MODEL = "megapose-1.0-RGB-multi-hypothesis-icp"
SPLIT, SPLIT_TYPE = "test", "pbr"
N_WORKERS, BSZ_IMAGES, BSZ_OBJECTS = 8, 128, 16
FLUSH_EVERY = 50
# ICPRefiner hardcodes 1000, at which ICP silently does nothing for objects this small.
ICP_N_MIN_POINTS = 100

_orig_icp_refinement = icp_refiner_module.icp_refinement


@functools.wraps(_orig_icp_refinement)
def _icp_refinement(*args, **kwargs):
    kwargs["n_min_points"] = ICP_N_MIN_POINTS
    return _orig_icp_refinement(*args, **kwargs)


icp_refiner_module.icp_refinement = _icp_refinement


def build_pose_estimator(object_dataset, refiner_run_id):
    model = NAMED_MODELS[MODEL]
    coarse_model, refiner_model, mesh_db = load_pose_models(
        coarse_run_id=model["coarse_run_id"],
        refiner_run_id=refiner_run_id or model["refiner_run_id"],
        object_dataset=object_dataset,
        force_panda3d_renderer=True,
        renderer_kwargs={"preload_cache": False, "split_objects": False, "n_workers": N_WORKERS},
        # Finetuned checkpoints live in experiments, the released ones in megapose-models.
        models_root=EXP_DIR if refiner_run_id else LOCAL_DATA_DIR / "megapose-models",
    )
    return PoseEstimator(
        refiner_model=refiner_model,
        coarse_model=coarse_model,
        detector_model=None,
        depth_refiner=ICPRefiner(mesh_db, refiner_model.renderer),
        bsz_objects=8,
        bsz_images=BSZ_IMAGES,
    )


def collect_rows(preds, scene_id, view_id, elapsed):
    infos = preds.infos
    poses = preds.poses.cpu().numpy()
    has_score = "pose_score" in infos.columns
    return [
        dict(
            scene_id=scene_id,
            im_id=view_id,
            obj_id=int(str(infos.iloc[n]["label"]).split("_")[-1]),
            score=float(infos.iloc[n]["pose_score"]) if has_score else 1.0,
            R=poses[n][:3, :3],
            t=poses[n][:3, 3] * 1e3,  # m -> mm
            time=elapsed,
        )
        for n in range(len(infos))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-name", required=True, help="csv prefix, no '_' and no '-'")
    parser.add_argument("--out-dir", required=True, help="where to write the result csv files")
    parser.add_argument("--ds-name", default="sodah6dof.bop19")
    parser.add_argument(
        "--refiner-run-id",
        default=None,
        help="finetuned refiner under local_data/experiments. Omit for the released weights",
    )
    args = parser.parse_args()

    set_logging_level("info")
    inference_params = dict(NAMED_MODELS[MODEL]["inference_parameters"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_root = args.ds_name.split(".")[0]

    scene_ds = make_scene_dataset(args.ds_name, load_depth=True)
    object_ds = make_object_dataset(ds_root)
    logger.info(f"{len(scene_ds.frame_index)} frames, objects {[o.label for o in object_ds.objects]}")

    pose_estimator = build_pose_estimator(object_ds, args.refiner_run_id)
    dataloader = DataLoader(
        scene_ds,
        batch_size=1,
        num_workers=N_WORKERS,
        shuffle=False,
        collate_fn=SceneObservation.collate_fn,
    )

    rows = {"refiner-final": [], "depth-refiner": []}

    def save():
        for key, pred_rows in rows.items():
            if pred_rows:
                # bop_toolkit parses "{method}_{dataset}-{split}-{split_type}.csv"
                name = f"{args.method_name}-{key}_{ds_root}-{SPLIT}-{SPLIT_TYPE}.csv"
                inout.save_bop_results(str(out_dir / name), pred_rows)

    t_start = time.time()
    for n, data in enumerate(tqdm(dataloader, desc=args.ds_name)):
        gt_detections = data["gt_detections"].cuda()
        if len(gt_detections.infos) == 0:
            continue
        scene_id = int(np.unique(gt_detections.infos["scene_id"]).item())
        view_id = int(np.unique(gt_detections.infos["view_id"]).item())
        obs = ObservationTensor.from_torch_batched(
            data["rgb"], data["depth"], data["cameras"].K
        ).cuda()

        with torch.no_grad():
            _, extra = pose_estimator.run_inference_pipeline(
                obs,
                detections=gt_detections,
                run_detector=False,
                bsz_objects=BSZ_OBJECTS,
                **inference_params,
            )

        elapsed = float(extra["time"])
        rows["refiner-final"] += collect_rows(extra["refiner"]["preds"], scene_id, view_id, elapsed)
        rows["depth-refiner"] += collect_rows(
            extra["depth_refiner"]["preds"], scene_id, view_id, elapsed
        )

        if (n + 1) % FLUSH_EVERY == 0:
            save()

    save()
    logger.info(f"Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
