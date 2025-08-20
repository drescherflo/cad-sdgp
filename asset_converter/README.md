# asset_converter
Die hier enthaltenen Skripte können verwendet werden, um CAD-Modelle von STEP nach OBJ oder von OBJ nach USD zu konvertieren.

## step_to_obj.py
Das Skript `step_to_obj.py` dient zur Konvertierung von STEP-Dateien (`.stp` oder `.step`) in OBJ-Dateien (`.obj`).

### Argumente
- `--input_dir`: Pflichtargument. Verzeichnis, das die STEP-Dateien enthält. 
- `--output_dir`: Pflichtargument. Zielverzeichnis für die konvertierten OBJ-Dateien. 
- `--scale_factor`: Optionaler Skalierungsfaktor für die OBJ-Dateien. Standardwert ist 0.001.

### Beispielbefehl
```
python step_to_obj.py --input_dir "/pfad/zum/input-verzeichnis" --output_dir "/pfad/zum/output-verzeichnis" --scale_factor 0.01
```

### Wichtige Hinweise
- Die Pfade zu den Verzeichnissen müssen existieren und korrekt angegeben werden.
- Während der Konvertierung werden temporäre STL-Dateien im Prozess erstellt, die nach Abschluss des Vorgangs automatisch gelöscht werden.
- Der standardmäßige Skalierungsfaktor von 0.001 ist typisch, wenn das Objekt in Millimetern entworfen wurde. Der Skalierungsfaktor kann bei Bedarf angepasst werden.

## obj_to_usd.py
Das Skript `obj_to_usd.py` dient zur Konvertierung von OBJ-Dateien (`.obj`) in USD-Dateien (`.usd`).

### Argumente
- `--input_dir`: Pflichtargument. Verzeichnis, das die OBJ-Dateien enthält. 
- `--output_dir`: Pflichtargument. Zielverzeichnis für die konvertierten USD-Dateien. 

### Beispielbefehl
```
python obj_to_usd.py --input_dir "/pfad/zum/input-verzeichnis" --output_dir "/pfad/zum/output-verzeichnis"
```

### Wichtige Hinweise
- Die Pfade zu den Verzeichnissen müssen existieren und korrekt angegeben werden.
