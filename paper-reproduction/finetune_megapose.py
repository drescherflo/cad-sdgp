#!/usr/bin/env python
"""Finetune the MegaPose pose refiner on a BOP-format dataset.

MegaPose ships no finetuning entrypoint, so this script assembles the configuration and calls
train_megapose() directly. The defaults are the settings the reported results were produced
with. Run it from the MegaPose checkout, with its conda environment:

    python finetune_megapose.py

Checkpoints, the resolved configuration and a per-epoch log are written to
local_data/experiments/<run-id>/.
"""

# Standard Library
import argparse
import functools

# Third Party
from omegaconf import OmegaConf

# MegaPose
import megapose.training.train_megapose as tm
from megapose.config import EXP_DIR
from megapose.datasets.object_dataset import RigidObjectDataset
from megapose.scripts.run_megapose_training import make_refiner_cfg
from megapose.training.training_config import DatasetConfig, HardwareConfig, TrainingConfig
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

PRETRAINED_REFINER = "refiner-rgb-653307694"

_forward_loss = tm.megapose_forward_loss


@functools.wraps(_forward_loss)
def _forward_loss_with_debug_dict(*args, debug_dict=None, **kwargs):
    """Give debug_dict a default, which the validation loop does not pass."""
    return _forward_loss(*args, debug_dict={} if debug_dict is None else debug_dict, **kwargs)


def _concat_object_datasets_dedup(datasets):
    """Concatenate object datasets, keeping every label only once.

    train_datasets and val_datasets point at the same dataset, which would otherwise produce
    duplicate labels and make RigidObjectDataset raise.
    """
    seen, objects = set(), []
    for dataset in datasets:
        for obj in dataset.list_objects:
            if obj.label not in seen:
                seen.add(obj.label)
                objects.append(obj)
    return RigidObjectDataset(objects)


tm.megapose_forward_loss = _forward_loss_with_debug_dict
tm.concat_object_datasets = _concat_object_datasets_dedup


def build_cfg(args: argparse.Namespace) -> TrainingConfig:
    cfg = OmegaConf.structured(TrainingConfig)
    # check_update_config() assigns is_coarse_compat, which TrainingConfig does not declare.
    OmegaConf.set_struct(cfg, False)

    cfg.hardware = HardwareConfig(n_cpus=10, n_gpus=1)
    cfg = make_refiner_cfg(cfg)
    cfg.run_id_pretrain = PRETRAINED_REFINER

    objects = dict(
        mesh_obj_ds_name=args.ds_name,
        renderer_obj_ds_name=f"{args.ds_name}.panda3d",
    )
    cfg.train_datasets = [DatasetConfig(ds_name=f"{args.ds_name}.pbr", **objects)]
    cfg.val_datasets = [DatasetConfig(ds_name=f"{args.ds_name}.val", **objects)]

    cfg.input_resize = (360, 480)  # (h, w), the native render resolution
    cfg.n_symmetries_batch = 1
    cfg.background_augmentation = False  # would need VOC2012

    cfg.epoch_size = 6000
    cfg.val_size = 800
    cfg.n_epochs_warmup = 2
    cfg.lr_epoch_decay = 30
    cfg.val_epoch_interval = 1
    cfg.save_epoch_interval = 5
    cfg.n_dataloader_workers = 8
    cfg.n_rendering_workers = 8
    cfg.add_iteration_epoch_interval = 1

    cfg.lr = args.lr
    cfg.n_epochs = args.n_epochs
    cfg.batch_size = args.batch_size
    cfg.init_trans_std = tuple(args.init_trans_std)

    cfg.run_id = args.run_id
    cfg.save_dir = str(EXP_DIR / args.run_id)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="ft-refiner", help="name of the run")
    parser.add_argument("--ds-name", default="sodah6dof", help="registered dataset name")
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="the MegaPose default of 3e-4 scaled linearly to the batch size below",
    )
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--init-trans-std",
        type=float,
        nargs=3,
        default=(0.02, 0.02, 0.20),
        metavar=("X", "Y", "Z"),
        help="translation noise of the start pose in metres, matched to the depth error scale "
        "of the scenes. The MegaPose default is 0.01 0.01 0.05",
    )
    args = parser.parse_args()

    set_logging_level("info")
    cfg = build_cfg(args)
    logger.info(
        f"Training {cfg.run_id} on {args.ds_name}: lr={cfg.lr} batch={cfg.batch_size} "
        f"epochs={cfg.n_epochs} init_trans_std={list(cfg.init_trans_std)}"
    )
    tm.train_megapose(cfg)


if __name__ == "__main__":
    main()
