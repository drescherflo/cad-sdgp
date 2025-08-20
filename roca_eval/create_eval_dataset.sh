#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set variables
CUSTOM_DATASET_DIR="$HOME/eval_dataset"

# Run dataset generation
cd ..

## STEP to OBJ
echo "Converting STEP files to OBJ"
conda run --no-capture-output -n sodah-sdgp python asset_converter/step_to_obj.py \
  --input_dir "$CUSTOM_DATASET_DIR/cad_models/step/" \
  --output_dir "$CUSTOM_DATASET_DIR/cad_models/obj/"

## OBJ to USD
echo "Converting OBJ to USD"
conda run --no-capture-output -n sodah-sdgp python asset_converter/obj_to_usd.py \
  --input_dir "$CUSTOM_DATASET_DIR/cad_models/obj" \
  --output_dir "$CUSTOM_DATASET_DIR/cad_models/usd"

## Generate eval dataset configs
usd_dir="$CUSTOM_DATASET_DIR/cad_models/usd/"
config_base="$CUSTOM_DATASET_DIR/dataset_generator_configs/"

echo "Generating configs for the eval datasets"
conda run --no-capture-output -n sodah-sdgp python dataset_config_generator/eval_conveyor_config_generator.py \
   --usd_dir "$usd_dir" \
   --out_dir "$config_base" \
   --object_material_config_file "dataset_config_generator/config/eval_object_materials.json"

## Generate eval datasets
set +e # don't exit on exit code != 0 (this prevents aborting the script when isaac sim segfaults on exit for unknown reasons)
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
    conda run --no-capture-output -n sodah-sdgp python dataset_generator/conveyor_belt_dataset_generator.py \
      --headless \
      --output_dir "$output_dir" \
      --usd_dir "$usd_dir" \
      --config_file "$config_file" \
      --writer ResumableBasicWriter rgb=True camera_params=True \
      --writer WorldPoseWriter
  done
done
set -e  # Re-enable exit on exit code != 0

## Convert datasets
echo "Converting the generated datasets"
for dir in "${config_dirs[@]}"; do
  dataset_type_dir="$CUSTOM_DATASET_DIR/replicator_dataset/$dir"
  for dataset_dir in "$dataset_type_dir"/*; do
    echo "Converting $dataset_dir ..."
    conda run --no-capture-output -n sodah-sdgp python dataset_converter/dataset_converter.py \
      --obj_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ/" \
      --replicator_data_dir "$dataset_dir" \
      --output_dir "$CUSTOM_DATASET_DIR/converted/$dir/" \
      --converter ReplicatorToRocaEval
  done
done
