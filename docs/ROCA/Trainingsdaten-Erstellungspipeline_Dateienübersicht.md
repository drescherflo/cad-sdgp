# Auflistung aller Skripte und deren Eingabe-/Ausgabe-Daten
Hinweise:
- Data ist das Verzeichnis, in welchem die Datensätze ScanNet, ShapeNet und Scan2CAD abgelegt wurden
- ROCA ist das Root-Verzeichnis des ROCA-Git-Repositories.
- Die eingerückten Aufzählungen unter JSON-Dateien sind die verwendeten Keys.


## resize_images.py
### Input
- Data/ScanNet25k/tasks/scannet_frames_25k/scene*/intrinsics_color.txt
- Data/ScanNet25k/tasks/scannet_frames_25k/scene*/color/*.jpg
- Data/ScanNet25k/tasks/scannet_frames_25k/scene*/pose/*.txt

### Output
- Data/Images/tasks/scannet_frames_25k/scene*/pose/*.txt
- Data/Images/tasks/scannet_frames_25k/scene*/intrinsics_color.txt
- Data/Images/tasks/scannet_frames_25k/scene*/color/*.jpg


## render.py
### Input
- Data/Scan2CAD/full_annotations.json
  - id_scan
  - trs
    - translation
    - rotation
    - scale
  - aligned_models
    - sym
    - catid_cad
    - id_cad
    - trs
      - translation
      - rotation
      - scale
- ROCA/metadata/scan2cad_taxonomy_9.json
  - shapenet
- ROCA/metadata/labelids_all.txt
- Data/ShapeNetCore.v2/*1/*2/models/model_normalized.obj (*1: cat_id aus full_annotations.json, *2: id_cad aus full_annotations.json)
- Data/Images/tasks/scannet_frames_25k/scene*1/pose/*.txt  (*1: id_scan aus full_annotations.json)
- Data/Images/tasks/scannet_frames_25k/scene*1/pose/*.txt  (*1: id_scan aus full_annotations.json)
- Data/Images/tasks/scannet_frames_25k/scene*1/intrinsics_color.txt  (*1: id_scan aus full_annotations.json)
- Data/Images/tasks/scannet_frames_25k/scene*1/color/*any*.jpg  (*1: id_scan aus full_annotations.json)

### Output
- Data/Rendering/scene*1/noc wird zwar vorbereitet aber nie berechnet/angelegt (*1: id_scan aus full_annotations.json)
- Data/Rendering/scene*1/depth (*1: id_scan aus full_annotations.json)
- Data/Rendering/scene*1/instance (*1: id_scan aus full_annotations.json)
- Data/Dataset/scan2cad_image_alignments.json


## rendering_to_coco.py
### Input
- Data/Dataset/scan2cad_image_alignments.json
- ROCA/metadata/scan2cad_taxonomy_9.json
  - name
  - shapenet
- ROCA/metadata/labelids.txt
- ROCA/metadata/val_images.txt
- ROCA/metadata/scannetv2_val.txt
- ROCA/metadata/scannetv2_train.txt
- Data/Rendering/scene*1/instance/*2.png (*1 und *2 aus scan2cad_image_alignments.json)

### Output
- Data/Dataset/scan2cad_instances_train.json
- Data/Dataset/scan2cad_instances_val.json
- Data/Dataset/scan2cad_alignment_classes.json
- Data/Dataset/scan2cad_rendering_config.json


## create_scenes.py
### Input
- Data/Scan2CAD/full_annotations.json
  - id_scan
  - trs
    - translation
    - rotation
    - scale
  - aligned_models
    - cat_id
    - id_cad
    - trs
      - translation
      - rotation
      - scale
      - sym
- Data/Dataset/scan2cad_instances_val.json
- ROCA/metadata/scannetv2_val.txt

### Output
- Data/Dataset/scan2cad_val_scenes.json


## create_cad_db.py
### Input
- ROCA/metadata/scannetv2_val.txt
- ROCA/metadata/scannetv2_train.txt
- Data/Dataset/scan2cad_instances_train.json
- Data/Dataset/scan2cad_instances_val.json
- Data/ShapeNetCore.v2/*1/*2/models/model_normalized.obj (*1: cat_id aus scan2cad_instances_(train|val).json, *2: id_cad aus scan2cad_instances_(train|val).json)

### Output
- Data/Dataset/scan2cad_train_cads.pkl
- Data/Dataset/scan2cad_val_cads.pkl


## voxelize_cads.py
### Input
- Data/Dataset/scan2cad_train_cads.pkl
- Data/Dataset/scan2cad_val_cads.pkl

### Output
- Data/Dataset/train_grids_32.pkl
- Data/Dataset/val_grids_32.pkl


# Zu erzeugende Dateien für Erzeugung von eigenen Trainingsdaten
- Data/ScanNet25k/tasks/scannet_frames_25k/scene*/intrinsics_color.txt
- Data/ScanNet25k/tasks/scannet_frames_25k/scene*/color/*.jpg
- Data/ScanNet25k/tasks/scannet_frames_25k/scene*/pose/*.txt
- Data/Scan2CAD/full_annotations.json
  - id_scan
  - trs
    - translation
    - rotation
    - scale
  - aligned_models
    - sym
    - catid_cad
    - id_cad
    - trs
      - translation
      - rotation
      - scale
- Data/ShapeNetCore.v2/*1/*2/models/model_normalized.obj (*1: cat_id aus full_annotations.json, *2: id_cad aus full_annotations.json)
- ROCA/metadata/scan2cad_taxonomy_9.json 
  - shapenet
  - name
- ROCA/metadata/labelids_all.txt
- ROCA/metadata/labelids.txt
- ROCA/metadata/val_images.txt
- ROCA/metadata/scannetv2_val.txt
- ROCA/metadata/scannetv2_train.txt

Zusätzlich benötigt die Trainingspipeline von ROCA noch folgende Dateien, wenn nur mit einem CAD-Modell trainiert werden soll:
- ROCA/metadata/points_val.pkl
- ROCA/metadata/points_train.pkl

# Verwendete Koordinatensysteme in den Datensätzen
- World Space
- Model Space
- Scan Space
- Camera Space
