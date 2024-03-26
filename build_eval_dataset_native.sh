#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set variables
CUSTOM_DATASET_DIR="$HOME/eval_dataset"
ISAAC_SIM_INSTALL_DIR="$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1"

# Run dataset generation
## STEP to OBJ
echo "Converting STEP files to OBJ"
conda run --no-capture-output -n step-to-obj python step_to_obj_conversion/step_to_obj.py \
  --input_dir "$CUSTOM_DATASET_DIR/cad_models/STEP/" \
  --output_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ/"

## OBJ to USD
echo "Converting OBJ to USD"
"$ISAAC_SIM_INSTALL_DIR"/python.sh "$ISAAC_SIM_INSTALL_DIR/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py" \
  --folders "$CUSTOM_DATASET_DIR/cad_models/OBJ"

## Generate eval dataset configs
usd_dir="$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/"
config_base="$CUSTOM_DATASET_DIR/dataset_generator_configs/"

echo "Generating configs for the eval datasets"
source custom_dataset/dataset_generation/config_generation/venv/bin/activate
python custom_dataset/dataset_generation/config_generation/eval_conveyor_config_generator.py \
   --usd_dir "$usd_dir" \
   --out_dir "$config_base" \
   --num_scenes 5 \
   --num_cluttered_scenes 5
deactivate

## Generate eval datasets
echo "Generating the evaluation datasets"
replicator_dataset_base="$CUSTOM_DATASET_DIR/replicator_dataset/"
config_dirs=("cluttered" "uncluttered")

for dir in "${config_dirs[@]}"; do
  # Find all json files in current dir
  config_files=$(find "${config_base}/${dir}" -type f -name "*.json")

  # For all json files
  for config_file in ${config_files}; do
    echo "Processing ${config_file}..."

    # Extract filename without file type for output dir
    filename=$(basename "${config_file}" .json)

    # Create output dir
    output_dir="${replicator_dataset_base}/${dir}/${filename}/"

    # Run dataset generation script
    "$ISAAC_SIM_INSTALL_DIR"/python.sh custom_dataset/dataset_generation/data_generation/conveyor_belt_dataset_generator.py \
      --headless \
      --output_dir "$output_dir" \
      --usd_dir "$usd_dir" \
      --config_file "$config_file" \
      --writer ResumableBasicWriter rgb=True camera_params=True \
      --writer WorldPoseWriter
  done
done

## Convert datasets
echo "Converting the generated datasets"
source custom_dataset/dataset_conversion/venv/bin/activate
for dir in "${config_dirs[@]}"; do
  dataset_type_dir="$CUSTOM_DATASET_DIR/replicator_dataset/$dir"
  for dataset_dir in "$dataset_type_dir"/*; do
    python custom_dataset/dataset_conversion/dataset_converter.py \
      --obj_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ/" \
      --replicator_data_dir "$dataset_dir" \
      --output_dir "$CUSTOM_DATASET_DIR/converted/$dir/" \
      --converter ReplicatorToRocaEval
  done
done
deactivate
