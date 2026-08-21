#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Run from the repository root, so that the relative script paths and the pixi manifest are found.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Set variables
CUSTOM_DATASET_DIR="$HOME/custom_dataset"
# Set to "true" to render the objects with the materials extracted from the CAD models
# instead of the randomized materials from the generator configs
KEEP_CAD_MATERIALS="${KEEP_CAD_MATERIALS:-false}"

# Build the optional flag for the dataset generators
KEEP_CAD_MATERIALS_FLAG=()
if [ "$KEEP_CAD_MATERIALS" = "true" ]; then
  KEEP_CAD_MATERIALS_FLAG=(--keep_cad_materials)
fi

# Run dataset generation
## STEP to OBJ
echo "Converting STEP files to OBJ"
pixi run python asset_converter/step_to_obj.py \
  --input_dir "$CUSTOM_DATASET_DIR/cad_models/step/" \
  --output_dir "$CUSTOM_DATASET_DIR/cad_models/obj/"

## OBJ to USD
echo "Converting OBJ to USD"
pixi run python asset_converter/obj_to_usd.py \
  --input_dir "$CUSTOM_DATASET_DIR/cad_models/obj" \
  --output_dir "$CUSTOM_DATASET_DIR/cad_models/usd"

## Generate config for 6 DOF dataset
echo "Generating config for the 6 DOF dataset generation"
pixi run python dataset_config_generator/6_dof_config_generator.py \
   --usd_dir "$CUSTOM_DATASET_DIR/cad_models/usd/" \
   --out_path "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" \
   --num_frames_per_object 1000

## Generate 6 DOF dataset
echo "Generating 6 DOF dataset"
pixi run python dataset_generator/6_dof_dataset_generator.py \
   --headless \
   --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
   --usd_dir "$CUSTOM_DATASET_DIR/cad_models/usd/" \
   --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" \
   --writer ResumableBasicWriter rgb=True camera_params=True distance_to_image_plane=True instance_segmentation=True colorize_instance_segmentation=False \
   --writer WorldPoseWriter \
   "${KEEP_CAD_MATERIALS_FLAG[@]}"


## Generate config for conveyor dataset
echo "Generating config for the conveyor belt dataset generation"
pixi run python dataset_config_generator/conveyor_config_generator.py \
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
pixi run python dataset_generator/conveyor_belt_dataset_generator.py \
  --headless \
  --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --usd_dir "$CUSTOM_DATASET_DIR/cad_models/usd/" \
  --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/conveyor_generator_config.json" \
  --writer ResumableBasicWriter rgb=True camera_params=True distance_to_image_plane=True instance_segmentation=True colorize_instance_segmentation=False \
  --writer WorldPoseWriter \
  "${KEEP_CAD_MATERIALS_FLAG[@]}"

## Convert dataset
echo "Converting datasets to ROCA and BOP format"
pixi run python dataset_converter/dataset_converter.py \
  --obj_dir "$CUSTOM_DATASET_DIR/cad_models/obj/" \
  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --output_dir "$CUSTOM_DATASET_DIR/converted/" \
  --converter ReplicatorToRoca \
  --converter ReplicatorToBop
