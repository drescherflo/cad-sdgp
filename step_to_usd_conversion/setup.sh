#!/bin/bash

# Create and activate conda environment
yes | conda create -n step_to_usd python=3.10
conda activate step_to_usd

# Install pythonocc-core
yes | conda update --all
yes | conda install -c conda-forge pythonocc-core=7.7.2

# Install stl-obj-convertor
yes | pip install --upgrade pip
yes | pip install stl-obj-convertor
