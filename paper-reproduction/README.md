# Reproducing the Results of "Synthetic Data Generation Pipeline for CAD-Based Object Reconstruction and Pose Estimation"

All required files can be downloaded here: https://drive.proton.me/urls/27FPHA51MM#iAkDQYk01pAZ

---

## Training ROCA

### Creating the Training Dataset

1. Create the directory `~/custom_dataset/cad_models`
2. Place the three directories `OBJ`, `OBJ_converted`, and `STEP` from the `cad-models` folder of the downloaded files into `~/custom_dataset/cad_models`
3. Copy the contents of the `training` directory from the downloaded files into `~/custom_dataset/`
4. Set up the SDGP environment as described in the root `README.md` (`pixi install`)
5. Run `create_train_dataset.sh` - depending on your hardware, this may take one to two days
6. Clone [ROCA](https://github.com/drescherflo/ROCA) into your home directory
7. `cd` into `ROCA` and run `setup.sh` to set up the environment
8. `cd` into `renderer` and open `env.sh`
9. Set `BASE_DIR` to the path of the generated dataset (default: `conveyor_belt`)
10. Run `run.sh` to generate the missing metadata required for training

### Training

1. `cd` into `ROCA/network`
2. Open `env.sh` and set `BASE_DIR` to the path of the generated dataset (default: `conveyor_belt`)
3. Run `run.sh` to start training

> **Note:** The `terminal_log.txt` files in the `trained-models` subdirectories were created by redirecting stdout via `tee`.

---

## Evaluating ROCA

### Creating the Evaluation Dataset

1. Create the directory `~/custom_dataset/converted/conveyor_belt/ReplicatorToRoca/Dataset`
2. Copy the contents of `evaluation/Dataset` from the downloaded files into that directory
3. Create the directory `~/eval_dataset/cad_models`
4. Place the three directories `OBJ`, `OBJ_converted`, and `STEP` from the `cad-models` folder of the downloaded files into `~/eval_dataset/cad_models`
5. Copy the directory `evaluation/dataset_generator_configs` from the downloaded files into `~/eval_dataset/`
6. Set up the SDGP environment as described in the root `README.md` (`pixi install`)
7. Run `create_eval_dataset.sh` - depending on your hardware, this may take up to five days

### Generating Raw Evaluation Data

1. Clone [ROCA](https://github.com/drescherflo/ROCA) into your home directory
2. `cd` into `ROCA` and run `setup.sh` to set up the environment
3. Copy the contents of `trained-models` from the downloaded files into `~/custom_dataset`
4. Run `run_eval.sh`

### Running the Evaluation

1. Open `evaluation.py` and set the path to the evaluation dataset at the end of the file
2. Run it in the `paper-repro` environment: `pixi run -e paper-repro python evaluation.py`

---

## Finetuning MegaPose

### Setting Up MegaPose

1. Clone [MegaPose](https://github.com/megapose6d/megapose6d) into your home directory
2. Set up the conda environment as described in the MegaPose README
3. Download the pretrained weights: `python -m megapose.scripts.download --megapose_models`.
   They land in `local_data/megapose-models/`.
4. Symlink them into `local_data/experiments/`, because `run_id_pretrain` resolves against that
   directory and not against `megapose-models`:
   ```bash
   ln -s ../megapose-models/refiner-rgb-653307694 local_data/experiments/
   ln -s ../megapose-models/coarse-rgb-906902141  local_data/experiments/
   ```

### Registering the Dataset

`create_dataset.sh` already writes a BOP-format copy of each scenario through the
`ReplicatorToBop` converter, with `train_pbr`, `val_pbr` and `test_pbr` splits. Three steps make
it usable:

1. Symlink it into `local_data/bop_datasets/`, under a name containing **neither `_` nor `-`** -
   the BOP toolkit parses result filenames as `{method}_{dataset}-{split}-{split_type}.csv`.
2. Register that name in `src/megapose/datasets/datasets_cfg.py`. MegaPose has no dataset
   configuration. Datasets are hardcoded in three `if`/`elif` chains, one per factory function.
3. Register the dataset with the BOP toolkit as well, so the scoring stage can resolve object
   ids, symmetries and image size.

### Finetuning the Refiner

Start from the released RGB refiner weights and train on `<name>.pbr`,
validating on `<name>.val`. Only the refiner is finetuned. The coarse network keeps the released
weights.

| Setting | Value |
| --- | --- |
| Epochs | 10 |
| Samples per epoch | 6000 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | `5e-6` |
| Warmup epochs | 2 |
| `init_trans_std` | `[0.02, 0.02, 0.20]` |
| `input_resize` | `(360, 480)` |


### Evaluating

Evaluation runs in two stages. Detections are ground-truth boxes, since MegaPose ships no
detector for these objects.

1. Inference over `test_pbr`, writing BOP-format result CSVs. A single pass produces both the
   RGB-only result (`refiner-final`) and the ICP-refined one (`depth-refiner`), with ICP run at
   `n_min_points=100`.
2. Scoring with the BOP toolkit (`scripts/eval_bop19_pose.py`, `--renderer_type vispy`), using
   the target list matching the evaluated frames.

---

## Physics Calibration

The `physics-calibration` directory in the downloaded files contains all files needed to run `plot_physics_frequency.py` and regenerate the plot from the paper. The scene description JSON files are also included, so the experiments can be rerun by loading them with `../dataset_generator/conveyor_belt_dataset_generator.py`.