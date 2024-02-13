#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Set custom dataset dir as environment variable
export CUSTOM_DATASET_DIR="$HOME/custom_dataset"

# Run dataset generation
docker compose run step_to_obj_converter
docker compose run obj_to_usd_converter
docker compose run config_generator_for_6_dof_dataset_generation
docker compose run 6_dof_dataset_generator
docker compose run dataset_converter
# TODO: Add conveyor config and dataset generation