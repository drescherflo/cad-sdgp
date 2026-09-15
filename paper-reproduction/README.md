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

3. Register the dataset with the BOP toolkit as well, so the scoring stage can resolve it:
   * `bop_toolkit_lib/dataset_params.py`: add the dataset to `obj_ids`, add it to
     `symmetric_obj_ids` as an empty list, and add an `elif` case to `get_split_params` setting
     `scene_ids` and `im_size`.
   * `scripts/eval_bop19_pose.py`: add a `vsd_deltas` entry for the dataset, set to 15.
   * Symlink the dataset under the path the toolkit resolves as `BOP_PATH`.

### Finetuning the Refiner

Start from the released RGB refiner weights and train on `<name>.pbr`, validating on
`<name>.val`. Only the refiner is finetuned. The coarse network keeps the released weights.

MegaPose provides no finetuning entrypoint, so `finetune_megapose.py` in this directory
assembles the configuration and calls `train_megapose()`. Its defaults are the settings below,
so running it in the MegaPose environment reproduces the run:

```bash
python finetune_megapose.py --run-id ft-refiner
```


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

Training writes to `local_data/experiments/<run-id>/`: `config.yaml` with the fully resolved
configuration, `log.txt` with one JSON line per epoch holding training and validation loss,
learning rate and timings, and `checkpoint.pth.tar` alongside periodic
`checkpoint_epoch=*.pth.tar`.

### Evaluating

Evaluation runs in two stages. Detections are ground-truth boxes, since MegaPose ships no
detector for these objects.

1. **Inference** with `eval_megapose.py` from this directory, over `<name>.bop19`. Omitting
   `--refiner-run-id` evaluates the released weights, which produces the baseline:

   ```bash
   python eval_megapose.py --method-name megaposebase --out-dir <results directory>
   python eval_megapose.py --method-name megaposeft   --out-dir <results directory> \
       --refiner-run-id ft-refiner
   ```

   One pass writes two BOP result files, so the RGB-only and the depth-refined numbers come from
   the same inference:

   ```
   <method>-refiner-final_<dataset>-test-pbr.csv
   <method>-depth-refiner_<dataset>-test-pbr.csv
   ```

2. **Scoring** with the BOP toolkit, passing both result files:

   ```bash
   python scripts/eval_bop19_pose.py \
       --renderer_type vispy \
       --targets_filename test_targets_bop19.json \
       --results_path <directory holding the csv files> \
       --eval_path <output directory> \
       --result_filenames <csv>,<csv>
   ```

   For each result file the toolkit writes a directory containing `scores_bop19.json` with the
   average recall and its VSD, MSSD and MSPD components.

---

## Physics Calibration

The `physics-calibration` directory in the downloaded files contains all files needed to run `plot_physics_frequency.py` and regenerate the plot from the paper. The scene description JSON files are also included, so the experiments can be rerun by loading them with `../dataset_generator/conveyor_belt_dataset_generator.py`.