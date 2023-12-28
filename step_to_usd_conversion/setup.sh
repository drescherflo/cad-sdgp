#!/bin/bash

# Create and activate conda environment
conda create -n roca python=3.10
source activate pyoccenv

# Install pythonocc-core
conda install -c conda-forge pythonocc-core=7.7.2
