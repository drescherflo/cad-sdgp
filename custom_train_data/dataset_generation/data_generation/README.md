# Gebrauchsanweisung für `6_dof_dataset_generator.py`
Das Skript `6_dof_dataset_generator.py` generiert Trainingsdatensätze für Anwendungen mit sechs Freiheitsgraden (6-DOF). 
Es verwendet eine Konfigurationsdatei und USD-Modelle, um Szenen für das Training zu erstellen.
Die Daten werden mittels sogenannter Writer erfasst und auf den Datenträger geschrieben. 

Das Skript muss mit dem NVIDIA Isaac Sim Python-Interpreter ausgeführt werden.
Dieser befindet sich für NVIDIA Isaac Sim 2023.1.1 unter Verwendung der Standardeinstellungen während der Installation unter `~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh`.

## Argumente
- `--headless`: Startet das Skript im Headless-Modus. Keine grafische Benutzeroberfläche wird angezeigt.
- `--output_dir`: Pflichtargument. Gibt das Ausgabeverzeichnis an, in dem die generierten Daten gespeichert werden.
- `--usd_dir`: Pflichtargument. Verzeichnis, das die USD-Versionen (Universal Scene Description) der CAD-Modelle enthält, die für die Datengenerierung verwendet werden.
- `--config_file`: Pflichtargument. Pfad zur JSON-Konfigurationsdatei, die die zu generierenden Szenen beschreibt. Zu Erstellen mit [data_generation_config_generator.py](../config_generation).
- `--writer`: Pflichtargument. Konfiguriert Writer aus dem Plugin-Paket `resumable_writers`. Ein Writer wird nur dann geladen, wenn dessen Klassenname korrekt geschrieben wird. Die Argumente des Writers müssen das Format `argument=(true|false)` haben. Jeder Wert `!= true` wird als `False` interpretiert. Dieses Argument kann mehrfach hinzugefügt werden.

## Beispielbefehl
```bash
python.sh 6_dof_dataset_generator.py --headless --output_dir "/pfad/zum/output" --usd_dir "/pfad/zum/usd" --config_file "/pfad/zur/config.json" --writer Writer1 writer1_arg=true writer1_arg2=false --writer Writer2
```

## Wichtige Hinweise
- Alle Pflichtargumente (`--output_dir`, `--usd_dir`, `--config_file`, `--writer`) müssen korrekt angegeben werden, um das Skript erfolgreich auszuführen.
- Der `--headless`-Modus ist optional und nützlich für Umgebungen ohne grafische Benutzeroberfläche.
- Das ist durch verschiedene Writer erweiterbar, die im `resumable_writers` Plugin-Paket definiert sind. Weitere Writer können hinzugefügt werden, müssen allerdings von der abstrakten Klasse `ResumableWriterInterface` erben.

## Bereits implementierte Writer
### ResumableBasicWriter
Während der Generierung der Trainingsdaten wird die Simulation mehrfach zurückgesetzt.
Dies hat zur Folge, dass das von NVIDIA bereitgestellte BasicWriter die bisher erzeugten Trainingsdaten überschreibt.

Der ResumableBasicWriter ist ein Wrapper um den BasicWriter von NVIDIA und verhindert das Überschreiben von bereits erzeugten Trainingsdaten.
FÜr die Initialisierung des BasicWriter über die Kommandozeilenargumente werden nur die Argumente vom Typ `bool` unterstützt.
Die Argumente des NVIDIA BasicWriter befinden sich in der zugehörigen [API-Dokumentation](https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/omni.replicator.core/docs/API.html#basicwriter).

### WorldPoseWriter
Der WorldPoseWriter speichert Position und Orientierung für jedes sichtbare Objekt im Kamerabild und das zugehörige semantische Label.
Er unterstützt keine Argumente, die über die Kommandozeile angegeben werden können.
