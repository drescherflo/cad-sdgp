#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set variables
CUSTOM_DATASET_DIR="$HOME/custom_dataset"
EVAL_DATASET_BASE_DIR="$HOME/eval_dataset"
ROCA_DIR="$HOME/ROCA"
ROCA_NETWORK_DIR="$ROCA_DIR/network"
ROCA_MODELS_AND_CONF_BASE="$CUSTOM_DATASET_DIR/ROCA_Outputs"
EVALUATION_OUTPUT_DIR="$EVAL_DATASET_BASE_DIR/evaluation_output"
EVAL_IOU_THRESHOLD="0.5"

# Create evaluation raw data
for model_dir in "$ROCA_MODELS_AND_CONF_BASE/"*; do
  model_dir_basename=$(basename "$model_dir")
  echo "Creating evaluation raw data with model $model_dir_basename"

  dataset_types=("cluttered" "uncluttered")
  for dataset_type in "${dataset_types[@]}"; do
    for eval_dataset in "$EVAL_DATASET_BASE_DIR/converted/$dataset_type"/*/ReplicatorToRocaEval; do
      echo "Processing $eval_dataset ..."
      conda run --no-capture-output -n roca --cwd "$ROCA_NETWORK_DIR" python "$ROCA_NETWORK_DIR/generate_evaluation_raw_data.py" \
        --input_dir "$eval_dataset" \
        --model "$model_dir/model_final.pth" \
        --model_config "$model_dir/config.yaml" \
        --data_dir "$CUSTOM_DATASET_DIR/converted/conveyor_belt/ReplicatorToRoca/Dataset" \
        --out_file "$eval_dataset/eval_raw_data/$model_dir_basename/eval_raw_data.json" \
        --image_out_dir "$eval_dataset/eval_images/$model_dir_basename"
    done
  done
done

# Run evaluation over all models/datasets found under EVAL_DATASET_BASE_DIR
echo "Running evaluation..."
(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pixi run -e paper-repro python -m src.evaluation.evaluation \
    --eval-dataset-path "$EVAL_DATASET_BASE_DIR" \
    --cad-dir "$EVAL_DATASET_BASE_DIR/cad_models/OBJ" \
    --output-dir "$EVALUATION_OUTPUT_DIR" \
    --iou-threshold "$EVAL_IOU_THRESHOLD" \
    --exclude-suffix glass \
    --exclude-suffix default
)
