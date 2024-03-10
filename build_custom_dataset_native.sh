#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set variables
CUSTOM_DATASET_DIR="$HOME/custom_dataset"
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

## Generate config for 6 DOF dataset
# echo "Generating config for the 6 DOF dataset generation"
#source custom_dataset/dataset_generation/config_generation/venv/bin/activate
#python custom_dataset/dataset_generation/config_generation/6_dof_config_generator.py \
#   --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" \
#   --out_path "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" \
#   --num_frames_per_object 1000
#deactivate

## Generate 6 DOF dataset
# echo "Generating 6 DOF dataset"
# "$ISAAC_SIM_INSTALL_DIR"/python.sh custom_dataset/dataset_generation/data_generation/6_dof_dataset_generator.py \
#   --headless \
#   --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
#   --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" \
#   --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" \
#   --writer ResumableBasicWriter rgb=True camera_params=True \
#   --writer WorldPoseWriter


## Generate config for conveyor dataset
echo "Generating config for the conveyor belt dataset generation"
source custom_dataset/dataset_generation/config_generation/venv/bin/activate
python custom_dataset/dataset_generation/config_generation/conveyor_config_generator.py \
  --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" \
  --out_path "$CUSTOM_DATASET_DIR/dataset_generator_configs/conveyor_generator_config.json" \
  --object_init_min_x -6.0 \
  --object_init_max_x 2.0 \
  --conveyor_belt_speed 0.2 \
  --num_objects_per_scene 200 \
  --num_frames_per_scene 100 \
  --num_scenes_per_object 10
deactivate

## Generate conveyor dataset
echo "Generating the conveyor belt dataset"
"$ISAAC_SIM_INSTALL_DIR"/python.sh custom_dataset/dataset_generation/data_generation/conveyor_belt_dataset_generator.py \
  --headless \
  --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" \
  --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/conveyor_generator_config.json" \
  --writer ResumableBasicWriter rgb=True camera_params=True \
  --writer WorldPoseWriter

## Convert dataset
echo "Converting datasets to ROCA format"
source custom_dataset/dataset_conversion/venv/bin/activate
python custom_dataset/dataset_conversion/dataset_converter.py \
  --obj_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ/" \
  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/conveyor_belt/" \
  --output_dir "$CUSTOM_DATASET_DIR/converted/"
deactivate

#  --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" \
