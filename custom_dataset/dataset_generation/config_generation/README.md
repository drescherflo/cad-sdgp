# Gebrauchsanweisung

Das Skript `data_generation_config_generator.py` dient zur Generierung einer Konfigurationsdatei für die Erstellung von Trainingsdaten. 
Es erstellt eine detaillierte Konfiguration, die Kameraeinstellungen, Materialkonfigurationen, Szenenkonfigurationen und weitere Parameter enthält. 
Die generierte Konfiguration wird im JSON-Format gespeichert.

## Argumente
- `usd_dir`: Pflichtargument. Pfad zum Verzeichnis mit USD-Dateien. Kein Standardwert.
- `out_path`: Ausgabepfad für die Konfigurationsdatei. Standardwert ist "config.json".
- `frame_width`: Breite des Kamera-Rahmens in Pixel. Standardwert ist 480.
- `frame_height`: Höhe des Kamera-Rahmens in Pixel. Standardwert ist 360.
- `sub_frames_per_frame`: Anzahl der Unterframes pro Frame. Standardwert ist 32.
- `num_random_materials`: Anzahl der zufälligen Materialien. Standardwert ist 100.
- `probability_of_glass_material`: Wahrscheinlichkeit für Glasmaterial. Standardwert ist 0.5.
- `num_frames_per_object`: Anzahl der Frames pro Objekt. Standardwert ist 1000.
- `num_objects_per_frame`: Anzahl der Objekte pro Frame. Standardwert ist 20.
- `num_sphere_lights`: Anzahl der Kugellichter. Standardwert ist 5.
- `train_val_split`: Anteil der Validierungsdaten. Standardwert ist 0.2.
- `min_x`: Minimale X-Koordinate für Objektplatzierung. Standardwert ist -2.
- `max_x`: Maximale X-Koordinate für Objektplatzierung. Standardwert ist 2.
- `min_y`: Minimale Y-Koordinate für Objektplatzierung. Standardwert ist -1.
- `max_y`: Maximale Y-Koordinate für Objektplatzierung. Standardwert ist 1.
- `cam_distance_to_background`: Distanz der Kamera zur Hintergrundebene. Standardwert ist 5.

## Beispielbefehl
```bash
python data_generation_config_generator.py --usd_dir "/pfad/zum/verzeichnis" --out_path "/pfad/zu/config.json" --frame_width 1920 --frame_height 1080 --sub_frames_per_frame 5 --num_frames_per_object 10 --num_objects_per_frame 3 --num_random_materials 20 --num_sphere_lights 5 --train_val_split 0.2 --probability_of_glass_material 0.3 --cam_distance_to_background 10.0 --min_x -5.0 --max_x 5.0 --min_y -5.0 --max_y 5.0
```

## Wichtige Hinweise
- Das Skript stellt sicher, dass das angegebene Verzeichnis (`usd_dir`) existiert und USD-Dateien enthält.
- Der Ausgabepfad (`out_path`) sollte auf einen gültigen Speicherort verweisen. Existiert die Datei bereits, wird das Skript abgebrochen.
- `cam_distance_to_background` muss größer als 0 sein.
- `train_val_split` gibt den Anteil der Daten für das Validierungsset an und sollte zwischen 0 und 1 liegen.
