#!/bin/bash
# Warning! This script must not be used in the dockerfile to create the image, because conda activate cannot be used!

yes | conda env create -f environment.yaml
