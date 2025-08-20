#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set variables
CUSTOM_DATASET_DIR="$HOME/custom_dataset"

# Run dataset generation
## STEP to OBJ
echo "Converting STEP files to OBJ"
conda run --no-capture-output -n sodah-sdgp python step_to_obj_conversion/step_to_obj.py \
  --input_dir "$CUSTOM_DATASET_DIR/cad_models/step/" \
  --output_dir "$CUSTOM_DATASET_DIR/cad_models/obj/"

## OBJ to USD
echo "Converting OBJ to USD"
conda run --no-capture-output -n sodah-sdgp python obj_to_usd_conversion/obj_to_usd.py \
  --input_dir "$CUSTOM_DATASET_DIR/cad_models/obj" \
  --output_dir "$CUSTOM_DATASET_DIR/cad_models/usd"

## Generate config for 6 DOF dataset
echo "Generating config for the 6 DOF dataset generation"
conda run --no-capture-output -n sodah-sdgp python custom_dataset/dataset_generation/config_generation/6_dof_config_generator.py \
   --usd_dir "$CUSTOM_DATASET_DIR/cad_models/usd/" \
   --out_path "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" \
   --num_frames_per_object 1000

## Generate 6 DOF dataset
echo "Generating 6 DOF dataset"
conda run --no-capture-output -n sodah-sdgp python custom_dataset/dataset_generation/data_generation/6_dof_dataset_generator.py \
   --headless \
   --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
   --usd_dir "$CUSTOM_DATASET_DIR/cad_models/usd/" \
   --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" \
   --writer ResumableBasicWriter rgb=True camera_params=True \
   --writer WorldPoseWriter


## Generate config for conveyor dataset
echo "Generating config for the conveyor belt dataset generation"
conda run --no-capture-output -n sodah-sdgp python custom_dataset/dataset_generation/config_generation/conveyor_config_generator.py \
  --usd_dir "$CUSTOM_DATASET_DIR/cad_models/usd/" \
  --out_path "$CUSTOM_DATASET_DIR/dataset_generator_configs/conveyor_generator_config.json" \
  --object_init_min_x -6.0 \
  --object_init_max_x 2.0 \
  --conveyor_belt_speed 0.2 \
  --num_objects_per_scene 200 \
  --num_frames_per_scene 100 \
  --num_scenes_per_object 10

## Generate conveyor dataset
echo "Generating the conveyor belt dataset"
conda run --no-capture-output -n sodah-sdgp python custom_dataset/dataset_generation/data_generation/conveyor_belt_dataset_generator.py \
  --headless \
  --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --usd_dir "$CUSTOM_DATASET_DIR/cad_models/usd/" \
  --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/conveyor_generator_config.json" \
  --writer ResumableBasicWriter rgb=True camera_params=True \
  --writer WorldPoseWriter

## Convert dataset
echo "Converting datasets to ROCA format"
conda run --no-capture-output -n sodah-sdgp python custom_dataset/dataset_conversion/dataset_converter.py \
  --obj_dir "$CUSTOM_DATASET_DIR/cad_models/obj/" \
  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --output_dir "$CUSTOM_DATASET_DIR/converted/" \
  --converter ReplicatorToRoca
