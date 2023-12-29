#!/bin/bash

# Create and activate conda environment
yes | conda create -n pyoccenv python=3.10
conda activate pyoccenv

# Install pythonocc-core
yes | conda install -c conda-forge pythonocc-core=7.7.2
