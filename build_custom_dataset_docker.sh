#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set custom dataset dir as environment variable
export CUSTOM_DATASET_DIR="$HOME/custom_dataset"

# Run dataset generation
echo "Converting STEP files to OBJ"
docker compose run step_to_obj_converter

echo "Converting OBJ to USD"
docker compose run obj_to_usd_converter

echo "Generating config for the 6 DOF dataset generation"
docker compose run config_generator_for_6_dof_dataset_generation

echo "Generating 6 DOF dataset"
docker compose run 6_dof_dataset_generator

echo "Generating config for the conveyor belt dataset generation"
docker compose run config_generator_for_conveyor_belt_dataset_generation

echo "Generating the conveyor belt dataset"
docker compose run conveyor_dataset_generator

echo "Converting datasets to ROCA format"
docker compose run dataset_converter
