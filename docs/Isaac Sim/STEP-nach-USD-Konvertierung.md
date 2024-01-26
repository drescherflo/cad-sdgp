# Konvertierung von CAD-Modellen von STEP nach USD
Es existiert keine direkte Konvertierungsmöglichkeit.
Aus diesem Grund besteht der Konvertierungsprozess aus zwei Schritten, nämlich der Konvertierung von STEP nach OBJ und von OBJ nach USD.

Für den ersten Schritt wurde das Skript `step_to_obj.py` entwickelt.
Eine ausführliche Bedienungsanleitung befindet sich [hier](../../step_to_obj_conversion).
Für den zweiten Schritt bietet NVIDIA unter `~/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/api/omni.kit.asset_converter` das Skript `asset_usd_converter.py` an.
Dieses darf aufgrund der eingetragenen Lizenz nicht verändert oder verteilt werden.

## Konvertierung von OBJ-Dateien in USD-Dateien
NVIDIA stellt im Rahmen von Omniverse Isaac Sim ein Skript `asset_usd_converter.py ` zur Verfügung, welches 3D-Modelle im STL, OBJ oder FBX Format in das USD Format konvertiert.
Es kann verwendet werden, um die mittels `step_to_obj.py` erstellen OBJ-Dateien nach USD zu konvertieren.
Das Skript muss über den Python-Interpreter ausgeführt werden, welcher von Isaac Sim mitgeliefert wird.

### Voraussetzungen
- NVIDIA Omniverse Isaac Sim 2023.1.1 muss installiert sein.

### Argumente 
- `--folders`: Pflichtargument. Liste der Ordner, die konvertiert werden sollen (durch Leerzeichen getrennt).
- `--max-models`: Maximale Anzahl von Modellen pro Ordner, die konvertiert werden sollen. Standardwert ist 50.
- `--load-materials`: Wenn angegeben, werden Materialien aus den Meshes geladen.

### Beispielbefehl

```bash
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh ~/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py --folders [Ordnerpfade] [--max-models [Maximale Anzahl]] [--load-materials]
```

### Wichtige Hinweise
Die Ausgabeordner befinden sich im selben Verzeichnis, wie die Eingabe-Ordner und haben den Zusatz `_converted` im Namen.
