#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Create environment
conda env create -f environment.yaml
conda activate sodah-sdgp
