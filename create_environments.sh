#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Create environments
cd custom_dataset/dataset_conversion/
./setup_env.sh
cd -

cd custom_dataset/dataset_generation/config_generation/
./setup_env.sh
cd -

cd step_to_obj_conversion/
./setup_env.sh

