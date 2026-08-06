# Synthetic Data Generation Pipeline

A Synthetic Data Generation Pipeline (SDGP) that can generate synthetic training data for various neural networks (currently: ROCA) using CAD models in STEP format and NVIDIA Isaac Sim.

## Prerequisites

- NVIDIA RTX GPU
- Installed NVIDIA driver
- Linux with glibc 2.35 or newer (Ubuntu 22.04+), as required by the Isaac Sim wheels
- pixi

### Install pixi

- Run the installer

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

- Reload the terminal

```bash
source ~/.bashrc
```

### Set up the environment

```bash
pixi install
```

This resolves the environment from `pixi.toml` / `pixi.lock`.

The workspace defines two environments:

| Environment   | Purpose                                                                     |
|---------------|-----------------------------------------------------------------------------|
| `default`     | The SDGP pipeline itself (Isaac Sim, pythonocc-core, ...)                    |
| `paper-repro` | The evaluation scripts in `paper-reproduction/`, without Isaac Sim           |

Commands are run with `pixi run <command>`; the `paper-repro` environment is selected via
`pixi run -e paper-repro <command>`.

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
pixi install
pixi run python -m isaacsim --generate-vscode-settings
```

Use `pixi shell` to drop into an activated shell instead of prefixing every command with `pixi run`.

## Sources

- https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html (12 Aug 2025)

## Reproducing the Results of "Synthetic Data Generation Pipeline for CAD-Based Object Reconstruction and Pose Estimation"

Instructions can be found in the directory [paper-reproduction](paper-reproduction).

