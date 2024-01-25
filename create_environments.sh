#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Create environments
cd custom_train_data/dataset_conversion/
./setup_env.sh
cd -


