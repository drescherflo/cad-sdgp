#!/bin/bash

# Enable strict mode.
set -euo pipefail

# Create virtual environment
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
