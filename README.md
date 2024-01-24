# Trainingsdatenerstellungspipeline

- Docker-Images stehen zur Verfügung
- Alternativ können Images gebaut werden mit
  ```bash
  docker compose build
  ```

## STEP zu USD Konvertierung
### STEP zu OBJ
- STEP-Dateien in Verzeichnis "CAD Models/STEP" legen
- Konverter ausführen mit
  ```bash
  docker compose run step_to_obj_converter
  ```
- Konvertierte STEP-Dateien liegen im OBJ-Format im Verzeichnis "CAD Models/STEP"
- Angabe alternativer Pfade oder Skalierungsfaktoren:
  ```bash
  docker compose run step_to_obj_converter --input_dir "PFAD zu Verzeichnis mit STEP-Dateien" --output-dir "PFAD zu Verzeichnis mit OBJ-Dateien" --scale_factor 0.001
  ```
