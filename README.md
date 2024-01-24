# Trainingsdatenerstellungspipeline

- Docker-Images stehen auf Dockerhub zur Verfügung (mit Ausnahme von NVIDIA Isaac Sim Images)
- Alternativ können Images gebaut werden mit
  ```bash
  docker compose build
  ```
  
- Soll Docker nicht verwendet werden stehen READMEs in den Ordnern der einzelnen Komponenten mit Installationsanleitungen zur Verfügung.
- Argumente für Python-Skripte können mit Flag -h angezeigt werden oder sind in den Skripten selbst dokumentiert 

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
