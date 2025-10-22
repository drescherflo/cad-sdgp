# Synthetic Data Generation Pipeline

A Synthetic Data Generation Pipeline (SDGP) that can generate synthetic training data for various neural networks (currently: ROCA) using CAD models in STEP format and NVIDIA Isaac Sim.

## Prerequisites

- NVIDIA RTX GPU
- Installed NVIDIA driver
- Anaconda / Miniconda

### Install Miniconda

- Download the Anaconda installer

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

 - Run the installer

```bash
bash ~/Miniconda3-latest-Linux-x86_64.sh -b -u
```

- Finish the installation and reload the terminal

```bash
source ~/.bashrc
```

### Set up the Conda environment

```bash
conda env create -f environment.yaml
```

## Using the SDGP

- Place CAD models in STEP format into the directory `~/custom_dataset/cad_models/step`
- Run the pipeline

```bash
./create_dataset.sh
```

- The dataset for ROCA will be located at `~/custom_dataset/converted/6_dof/ReplicatorToROCA`
- The working directory is defined in the Bash script via the variable `CUSTOM_DATASET_DIR`, which points to `~/custom_dataset`. If a different path should be used, this variable can be modified.

## Development

```bash
conda env create -f environment.yaml
conda activate sodah-sdgp
python -m isaacsim --generate-vscode-settings
``` 

## Sources

- https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html (12 Aug 2025)

## Reproducing the Results of "Synthetic Data Generation Pipeline for CAD-Based Object Reconstruction and Pose Estimation"

Instructions can be found in the directory [paper-reproduction](paper-reproduction).


## Converting multiple obj files

- activate Isaac Sim Python environment and run command (adjust paths)

```bash
~/isaacsim/python.sh \
  ~/ros2_ws/src/synthetic-data-generation-pipeline/asset_converter/obj_to_usd_recursive.py \
  --input_root  ~/ros2_ws/resource/EGAD/train \
  --output_root ~/ros2_ws/resource/EGAD_usd/train
```

the script will automatically launch isaac sim in headless mode and convert all .obj files in folder input_root to .usd files and place them in output_root
