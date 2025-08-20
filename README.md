# Synthetic Data Generation Pipeline
Eine Synthetic Data Generation Pipeline (SDGP), die mithilfe von CAD-Modellen im STEP-Format und NVIDIA Isaac Sim synthetische Trainingsdaten für verschiedene Neuronale Netze (momentan: ROCA) erzeugen kann.

## Voraussetzungen
- NVIDIA RTX GPU
- Installierter NVIDIA Treiber
- Anaconda / Miniconda

### Miniconda installieren
- Anaconda Installer herunterladen
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  ```

- Installer ausführen
  ```bash
  bash ~/Miniconda3-latest-Linux-x86_64.sh -b -u
  ```

- Installation abschließen und Terminal neu laden
  ```bash
  source ~/.bashrc
  ```

### Conda Environment einrichten
```bash
./create_environments.sh
```

## Verwendung der SDGP
- CAD-Modelle im STEP-Format in Verzeichnis `~/custom_dataset/cad_models/step` legen
- Pipeline ausführen
  ```bash
  ./build_custom_dataset_native.sh
  ```
- Datensatz für ROCA liegt unter `~/custom_dataset/converted/6_dof/ReplicatorToROCA`

- **Hinweise**:
  - Arbeitsverzeichnis ist in Bash-Skripten über Variable `CUSTOM_DATASET_DIR` dir auf `~/custom_dataset`
  - Soll anderer Pfad verwendet werden, kann diese Variable abgeändert werden

## Development
```bash
conda env create -f environment.yaml
conda activate sodah-sdgp
python -m isaacsim --generate-vscode-settings
```

## Sources
- https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html (12.08.2025)
