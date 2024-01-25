#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Create environments
cd custom_train_data/dataset_conversion/
./setup_env.sh
cd -

cd custom_train_data/dataset_generation/config_generation/
./setup_env.sh
cd -

cd step_to_usd_conversion/
./setup_env.sh

