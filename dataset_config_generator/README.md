# Gebrauchsanweisung

## Konfig-Generator für die Erstellung eines universal einsetzbaren 6-DOF-Datensatzes
Das Skript `6_dof_config_generator.py` dient zur Generierung einer Konfigurationsdatei für die Erstellung von Trainingsdaten. 
Es erstellt eine detaillierte Konfiguration, die Kameraeinstellungen, Materialkonfigurationen, Szenenkonfigurationen und weitere Parameter enthält. 
Die generierte Konfiguration wird im JSON-Format gespeichert.

### Argumente
- `--usd_dir`: Pflichtargument. Pfad zum Verzeichnis mit USD-Dateien. Kein Standardwert.
- `--out_path`: Ausgabepfad für die Konfigurationsdatei. Standardwert ist "config.json".
- `--frame_width`: Breite des Kamera-Rahmens in Pixel. Standardwert ist 480.
- `--frame_height`: Höhe des Kamera-Rahmens in Pixel. Standardwert ist 360.
- `--sub_frames_per_frame`: Anzahl der Unterframes pro Frame. Standardwert ist 32.
- `--num_random_materials`: Anzahl der zufälligen Materialien. Standardwert ist 100.
- `--probability_of_glass_material`: Wahrscheinlichkeit für Glasmaterial. Standardwert ist 0.5.
- `--num_frames_per_object`: Anzahl der Frames pro Objekt. Standardwert ist 1000.
- `--num_objects_per_frame`: Anzahl der Objekte in der Szene. Standardwert ist 20.
- `--num_sphere_lights`: Anzahl der Kugellichter. Standardwert ist 5.
- `--sphere_min_intensity`: Minimale Intensität der Kugellichter. Standardwert ist 5000.
- `--sphere_max_intensity`: Maximale Intensität der Kugellichter. Standardwert ist 50000.
- `--train_val_split`: Anteil der Validierungsdaten. Standardwert ist 0.2.
- `--min_x`: Minimale X-Koordinate für Objektplatzierung. Standardwert ist -2.
- `--max_x`: Maximale X-Koordinate für Objektplatzierung. Standardwert ist 2.
- `--min_y`: Minimale Y-Koordinate für Objektplatzierung. Standardwert ist -1.
- `--max_y`: Maximale Y-Koordinate für Objektplatzierung. Standardwert ist 1.
- `--cam_distance_to_background`: Distanz der Kamera zur Hintergrundebene. Standardwert ist 5.
- `--dome_light_min_intensity`: Minimale Intensität der Dom-Szenenbeleuchtung. Standardwert ist 5000.
- `--dome_light_max_intensity`: Minimale Intensität der Dom-Szenenbeleuchtung. Standardwert ist 50000.

### Beispielbefehl
```bash
python 6_dof_config_generator.py --usd_dir "/pfad/zum/verzeichnis" --out_path "/pfad/zu/config.json" --frame_width 1920 --frame_height 1080 --sub_frames_per_frame 5 --num_frames_per_object 10 --num_objects_per_frame 3 --num_random_materials 20 --num_sphere_lights 5 --sphere_min_intensity 5000.0 --sphere_max_intensity 50000.0 --train_val_split 0.2 --probability_of_glass_material 0.3 --cam_distance_to_background 10.0 --min_x -5.0 --max_x 5.0 --min_y -5.0 --max_y 5.0 --dome_light_min_intensity 5000.0 --dome_light_max_intensity 50000.0
```

### Wichtige Hinweise
- Das Skript stellt sicher, dass das angegebene Verzeichnis (`usd_dir`) existiert und USD-Dateien enthält.
- Der Ausgabepfad (`out_path`) sollte auf einen gültigen Speicherort verweisen. Existiert die Datei bereits, wird das Skript abgebrochen.
- `cam_distance_to_background` muss größer als 0 sein.
- `train_val_split` gibt den Anteil der Daten für das Validierungsset an und sollte zwischen 0 und 1 liegen.



## Konfig-Generator für die Erstellung eines Datensatzes in einer Förderband-Szenario
Das Skript `conveyor_config_generator.py` generiert eine Konfigurationsdatei für die Erstellung eines Datensatzes in einem Förderbandszenario, inklusive Objekt-, Kamera-, Beleuchtungs- und Materialkonfigurationen.

### Argumente
- `--usd_dir`: Pflichtargument. Verzeichnis mit den USD-Dateien der CAD-Modelle. Kein Standardwert.
- `--out_path`: Ausgabepfad für die Konfigurationsdatei. Standard ist config.json.
- `--frame_width`: Breite der generierten Bilder in Pixel. Standard ist 480.
- `--frame_height`: Höhe der generierten Bilder in Pixel. Standard ist 360.
- `--sub_frames_per_frame`: Frames zwischen den gespeicherten Frames zur Vermeidung von Artefakten. Standard ist 32.
- `--num_random_materials`: Anzahl der generierten zufälligen Materialien. Standard ist 1000.
- `--probability_of_glass_material`: Wahrscheinlichkeit für die Erzeugung von Glasmaterialien. Standard ist 0.5.
- `--num_frames_per_scene`: Frames pro Szene. Der Standardwert wird automatisch berechnet.
- `--num_objects_per_scene`: Objekte pro Szene. Standard ist 100.
- `--num_scenes_per_object`: Szenen pro Objekt im USD-Verzeichnis. Standard ist 10.
- `--num_sphere_lights`: Kugellichter mit zufälliger Farbe pro Frame. Standard ist 5.
- `--train_val_split`: Anteil der Validierungsdaten am gesamten Datensatz. Standard ist 0.2.
- `--object_init_min_x`: Minimale x-Koordinate der Initialposition der Objekte. Standard ist -2.0.
- `--object_init_max_x`: Maximale x-Koordinate der Initialposition der Objekte. Standard ist -1.5.
- `--object_init_min_y`: Minimale y-Koordinate der Initialposition der Objekte. Standard ist -0.4.
- `--object_init_max_y`: Maximale y-Koordinate der Initialposition der Objekte. Standard ist 0.4.
- `--object_init_min_z`: Minimale z-Koordinate der Initialposition der Objekte. Standard ist 3.
- `--object_init_max_z`: Maximale z-Koordinate der Initialposition der Objekte. Standard ist 6.
- `--sphere_min_x`: Minimale x-Koordinate der Kugellichter. Standard ist -2.5.
- `--sphere_max_x`: Maximale x-Koordinate der Kugellichter. Standard ist 2.5.
- `--sphere_min_y`: Minimale y-Koordinate der Kugellichter. Standard ist -0.5.
- `--sphere_max_y`: Maximale y-Koordinate der Kugellichter. Standard ist 0.5.
- `--sphere_min_z`: Minimale z-Koordinate der Kugellichter. Standard ist 2.3.
- `--sphere_max_z`: Maximale z-Koordinate der Kugellichter. Standard ist 2.7.
- `--sphere_min_intensity`: Minimale Intensität der Kugellichter. Standard ist 1000.
- `--sphere_max_intensity`: Maximale Intensität der Kugellichter. Standard ist 5000.
- `--camera_pos_min_x`: Minimale x-Koordinate der Kameraposition. Standard ist -0.5.
- `--camera_pos_max_x`: Maximale x-Koordinate der Kameraposition. Standard ist 0.5.
- `--camera_pos_min_y`: Minimale y-Koordinate der Kameraposition. Standard ist -0.3.
- `--camera_pos_max_y`: Maximale y-Koordinate der Kameraposition. Standard ist 0.3.
- `--camera_pos_min_z`: Minimale z-Koordinate der Kameraposition. Standard ist 3.
- `--camera_pos_max_z`: Maximale z-Koordinate der Kameraposition. Standard ist 6.
- `--camera_rot_min_x`: Minimale Rotation der Kamera um die x-Achse. Standard ist -180.
- `--camera_rot_max_x`: Maximale Rotation der Kamera um die x-Achse. Standard ist 180.
- `--camera_rot_min_y`: Minimale Rotation der Kamera um die y-Achse. Standard ist -120.
- `--camera_rot_max_y`: Maximale Rotation der Kamera um die y-Achse. Standard ist -60.
- `--camera_rot_min_z`: Minimale Rotation der Kamera um die z-Achse. Standard ist -180.
- `--camera_rot_max_z`: Maximale Rotation der Kamera um die z-Achse. Standard ist 180.
- `--distant_light_min_rot_x`: Minimale Rotation des Distanzlichts um die x-Achse. Standard ist -90.
- `--distant_light_max_rot_x`: Maximale Rotation des Distanzlichts um die x-Achse. Standard ist 90.
- `--distant_light_min_rot_y`: Minimale Rotation des Distanzlichts um die y-Achse. Standard ist -90.
- `--distant_light_max_rot_y`: Maximale Rotation des Distanzlichts um die y-Achse. Standard ist 90.
- `--distant_light_min_rot_z`: Minimale Rotation des Distanzlichts um die z-Achse. Standard ist -180.
- `--distant_light_max_rot_z`: Maximale Rotation des Distanzlichts um die z-Achse. Standard ist 180.
- `--distant_light_min_intensity`: Minimale Intensität des Distanzlichts. Standard ist 200.
- `--distant_light_max_intensity`: Maximale Intensität des Distanzlichts. Standard ist 1000.
- `--conveyor_belt_speed`: Geschwindigkeit des Förderbands in der Simulation. Standard ist basierend auf Voreinstellungen.
- `--min_x_pos_for_record_start`: Minimale x-Koordinate, ab der die Datenaufzeichnung beginnt. Standard ist basierend auf Voreinstellungen.
- `--render_frequency`: Renderfrequenz in Hz. Standard ist 60.
- `--physics_frequency`: Frequenz, mit der die Physik berechnet wird. Standard ist 360.

### Beispielbefehl
```bash
python conveyor_config_generator.py --usd_dir "/pfad/zu/usd_modellen" --out_path "meine_config.json"
```

### Wichtige Hinweise
- Das Skript stellt sicher, dass das angegebene Verzeichnis (`usd_dir`) existiert und USD-Dateien enthält.
- Der Ausgabepfad (`out_path`) sollte auf einen gültigen Speicherort verweisen. Existiert die Datei bereits, wird das Skript abgebrochen.
- `train_val_split` gibt den Anteil der Daten für das Validierungsset an und sollte zwischen 0 und 1 liegen.
