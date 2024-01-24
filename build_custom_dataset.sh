#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Run dataset generation
docker compose run step_to_obj_converter
docker compose run obj_to_usd_converter
docker compose run config_generator_for_6_dof_dataset_generation
docker compose run 6_dof_dataset_generator

# Clean up
docker compose down
