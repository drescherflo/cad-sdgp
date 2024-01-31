#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set variables
CUSTOM_DATASET_DIR="$HOME/custom_dataset"
ISAAC_SIM_INSTALL_DIR="$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1"

# Run dataset generation
## STEP to OBJ
conda run --no-capture-output -n step-to-obj python step_to_obj_conversion/step_to_obj.py --input_dir "$CUSTOM_DATASET_DIR/cad_models/STEP/" --output_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ/"

## OBJ to USD
"$ISAAC_SIM_INSTALL_DIR"/python.sh "$ISAAC_SIM_INSTALL_DIR/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py" --folders "$CUSTOM_DATASET_DIR/cad_models/OBJ"

## Generate config for 6 DOF dataset
source custom_train_data/dataset_generation/config_generation/venv/bin/activate
python custom_train_data/dataset_generation/config_generation/data_generation_config_generator.py --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" --out_path "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" --num_frames_per_object 1000
deactivate

## Generate 6 DOF dataset
"$ISAAC_SIM_INSTALL_DIR"/python.sh custom_train_data/dataset_generation/data_generation/6_dof_dataset_generator.py --headless --output_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" --usd_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ_converted/" --config_file "$CUSTOM_DATASET_DIR/dataset_generator_configs/6_dof_generator_config.json" --writer ResumableBasicWriter rgb=True camera_params=True --writer WorldPoseWriter

## Convert dataset
source custom_train_data/dataset_conversion/venv/bin/activate
python custom_train_data/dataset_conversion/dataset_converter.py --obj_dir "$CUSTOM_DATASET_DIR/cad_models/OBJ/" --replicator_data_dir "$CUSTOM_DATASET_DIR/replicator_dataset/6_dof/" --output_dir "$CUSTOM_DATASET_DIR/converted/"
deactivate
