# ROCA Setup
## Anaconda installieren
- Aktuellen Installer von [Website](https://www.anaconda.com/download#downloads) laden
  Zum Zeitpunkt der Erstellung dieser Anleitung ist dies https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
- Installer ausführbar machen
  ```bash
  chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
  ```
- Installer ausführen und alle Anfragen mit `yes` beantworten oder mit `Enter` bestätigen
  ```bash  
  ./Anaconda3-2023.09-0-Linux-x86_64.sh
  ```

## ROCA einrichten
- ROCA-Repo clonen
  ```bash
  git clone https://github.com/cangumeli/ROCA.git
  ```

- `requirements.txt` mit nachfolgendem Inhalt aktualisieren, da unvollständig
  ```txt
  dataclasses==0.6
  opencv-python==4.5.4.60
  numpy-quaternion==2021.11.4.15.26.3
  pandas==1.3.4
  scipy==1.7.2
  open3d==0.13.0
  trimesh==3.9.35
  rtree==0.9.7
  ```

- Conda-Umgebung einrichten und Abhängigkeiten installieren
  ```bash
  cd ROCA
  source setup.sh
  ```
  Rückfragen mit `y` und `Enter` bestätigen
- `pytorch3d` nachinstallieren, da `setup.sh` eine nicht existierende Variante installieren möchte
  ```bash
  conda install pytorch3d -c pytorch3d
  ```
- `Pillow und numpy` downgraden, da mit torchvision eine zu neue Version intalliert wird
  ```bash
  pip install Pillow==9.5.0 numpy==1.22.4
  ```

### Variante 1: Demo-Daten zum Testen herunterladen (kein Training nötig)
- Demo-Daten vom [zugehörigen Google Drive](https://tex.stackexchange.com/questions/12806/guidelines-for-customizing-biblatex-styles) laden und innerhalb des Repos entpacken
  Im Repo-Verzeichnis (`ROCA`) sollten daraufhin die drei Verzeichnisse `Models`, `Data` und `Outputs` liegen
- Im Verzeichnis `Data` ebenfalls die drei ZIP-Dateien `Dataset.zip`, `Images.zip` und `Rendering.zip` entpacken
  Die Verzeichnisstruktur sollte dann aussehen wie folgt (limitiert auf 2 Ebenen):
  ````text
  ROCA/
  ├── Data
  │   ├── Dataset
  │   ├── Dataset.zip
  │   ├── full_annotations.json
  │   ├── Images
  │   ├── Images.zip
  │   ├── Rendering
  │   └── Rendering.zip
  ├── LICENSE
  ├── metadata
  │   ├── labelids_all.txt
  │   ├── labelids.txt
  │   ├── scan2cad_taxonomy_9.json
  │   ├── scan2cad_taxonomy.json
  │   ├── scannet_clusters_25k.json
  │   ├── scannetv2_train.txt
  │   ├── scannetv2_val.txt
  │   ├── scenes.txt
  │   └── val_images.txt
  ├── Models
  │   ├── config.yaml
  │   └── model_best.pth
  ├── network
  │   ├── assets
  │   ├── demo.py
  │   ├── env.sh
  │   ├── main.py
  │   ├── roca
  │   └── run.sh
  ├── Outputs
  │   ├── per_frame_best.json
  │   └── raw_results.csv
  ├── README.md
  ├── renderer
  │   ├── create_cad_db.py
  │   ├── create_scenes.py
  │   ├── env.sh
  │   ├── rendering_to_coco.py
  │   ├── render.py
  │   ├── resize_images.py
  │   ├── run.sh
  │   ├── scan2cad_rasterizer.cpython-38-x86_64-linux-gnu.so
  │   ├── utils
  │   └── voxelize_cads.py
  ├── requirements.txt
  └── setup.sh
  ````

### Variante 2: Neuronales Netz selbst trainieren
#### ShapeNet-Datensatz herunterladen
- Auf https://shapenet.org/signup/ registrieren
- Auf https://huggingface.co/join registrieren
- Unter https://huggingface.co/datasets/ShapeNet/ShapeNetCore Zugriff auf Datensatz anfragen
- Unter https://huggingface.co/settings/keys einen SSH-Key für git-Zugriff hinzufügen
- git lfs installieren
  ```bash
  sudo apt install git-lfs
  git lfs install
  ```
- Datensatz herunterladen
  ```bash
  git clone git@hf.co:datasets/ShapeNet/ShapeNetCore ~/Data/ShapeNetCore.v2/raw
  ```
- Datensatz entpacken
  ```bash
  unzip '~/Data/ShapeNetCore.v2/raw/*.zip' -d ~/Data/ShapeNetCore.v2
  ```

#### Scan2CAD-Datensatz herunterladen
- Auf https://goo.gl/forms/gJRMjzj05whyJDlO2 registrieren
- Datensatz herunterladen und entpacken
  ```bash
  mkdir -p ~/Data/Scan2CAD/raw/
  curl -L http://kaldir.vc.in.tum.de/scan2cad_download_link -o ~/Data/Scan2CAD/raw/Scan2CAD.zip
  unzip ~/Data/Scan2CAD/raw/Scan2CAD.zip -d ~/Data/Scan2CAD
  ```

#### ScanNet (ScanNet25k)-Datensatz herunterladen
- Die [ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) ausfüllen und an <scannet@googlegroups.com> senden.
- Download-Skript herunterladen und ausführbar machen
  ```bash
  mkdir -p ~/Data/ScanNet25k/raw/
  curl -L http://kaldir.vc.in.tum.de/scannet/download-scannet.py -o ~/Data/ScanNet25k/raw/download-scannet.py
  chmod +x ~/Data/ScanNet25k/raw/download-scannet.py
  ```
- ScanNet25k herunterladen
  ```bash
  python ~/Data/ScanNet25k/raw/download-scannet.py -o ~/Data/ScanNet25k/raw/ --preprocessed_frames
  unzip ~/Data/ScanNet25k/raw/tasks/scannet_frames_25k.zip -d ~/Data/ScanNet25k/tasks
  ```

#### Scan2CADRasterizer-lib erstellen
```bash
git clone https://github.com/cangumeli/Scan2CADRasterizer/tree/main
cd Scan2CADRasterizer
pip install .
cp build/lib.linux-x86_64-cpython-38/scan2cad_rasterizer.cpython-38-x86_64-linux-gnu.so ../renderer/scan2cad_rasterizer.cpython-38-x86_64-linux-gnu.so
cd ..
rm -rf Scan2CADRasterizer
```

#### Daten vorbereiten
```bash
cd renderer
sh run.sh
cd ..
```

#### Netz trainieren
```bash
cd network
sh run.sh
cd ..
```

## Demo ausführen
Die Demo (ausgeführt auf den Daten in Ordner 'network/assets/')
```bash
python demo.py --model_path ../Models/model_best.pth --data_dir ../Data/Dataset --config_path ../Models/config.yaml
```
oder wenn alle trainierten CAD-Modelle (außer Tische) berücksichtigt werden sollen und nicht nur Objekte in der Szene
```bash
python demo.py --model_path ../Models/model_best.pth --data_dir ../Data/Dataset --config_path ../Models/config.yaml --wild
```

# Quellen
- https://github.com/cangumeli/ROCA/tree/main (07.12.2023)
