#!/bin/bash

ISAAC_SIM_PYTHON_INTERPRETER="$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh"
$ISAAC_SIM_PYTHON_INTERPRETER 6_dof_training_dataset_generator.py --headless --output_dir "$HOME/custom_replicator_dataset/" --usd_dir "../../../CAD Models/OBJ_converted" --config_file "../config_generation/generated_configs/6_dof_only_simple_object.json"
