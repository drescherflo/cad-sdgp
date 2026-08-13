#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set variables
CUSTOM_DATASET_DIR="$HOME/custom_dataset"

# Run dataset generation
cd "$(dirname "${BASH_SOURCE[0]}")/.."

## Generate 6 DOF dataset
echo "Generating 6 DOF dataset"
pixi run python dataset_generator/6_dof_dataset_generator.py \
   --headless \
   --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
   --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" \
   --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" \
   --writer ResumableBasicWriter rgb=True camera_params=True \
   --writer WorldPoseWriter

## Generate conveyor dataset
echo "Generating the conveyor belt dataset"
pixi run python dataset_generator/conveyor_belt_dataset_generator.py \
  --headless \
  --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" \
  --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/conveyor_generator_config.json" \
  --writer ResumableBasicWriter rgb=True camera_params=True \
  --writer WorldPoseWriter

## Convert dataset
echo "Converting datasets to ROCA format"
pixi run python dataset_converter/dataset_converter.py \
  --obj_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ/" \
  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --output_dir "$CUSTOM_DATASET_DIR/converted/" \
  --converter ReplicatorToRoca
