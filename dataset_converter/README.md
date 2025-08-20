# Gebrauchsanweisung

## Kurze Beschreibung
Das Skript `dataset_converter.py` dient dazu, Datensätze aus dem Format von NVIDIA Replicator in ein anderes umzuwandeln. 
Es lädt Konverter-Plugins aus dem Paket `converter_plugins` und wendet diese auf die bereitgestellten Daten an.

## Argumente
- `--obj_dir`: Pflichtargument. Verzeichnis, das die Quelldateien im OBJ-Format enthält. Kein Standardwert.
- `--replicator_data_dir`: Pflichtargument. Verzeichnis, das die Quelldateien im Replicator-Datenformat enthält. Kein Standardwert.
- `--output_dir`: Pflichtargument. Zielverzeichnis für die konvertierten Dateien. Jedes Plugin erhält ein eigenes Unterverzeichnis. Kein Standardwert.

## Beispielbefehl
```bash
python dataset_converter.py --obj_dir "/pfad/zum/obj_verzeichnis" --replicator_data_dir "/pfad/zum/replicator_verzeichnis" --output_dir "/pfad/zum/ausgabe_verzeichnis"
```

## Wichtige Hinweise
- Das Skript erfordert, dass die Verzeichnisse `obj_dir` und `replicator_data_dir` existieren und entsprechende Dateien enthalten.
- Das Zielverzeichnis `output_dir` wird für die Speicherung der konvertierten Dateien verwendet.
- Eigene Konvertierungsroutinen können durch Erstellen eines eigenen Plugins hinzugefügt werden. Sie werden automatisch ausgeführt, wenn sie von der Klasse `ConverterInterface` erben und im Paket `converter_plugins` liegen.

## Existierende Konverter-Plugins
### ReplicatorToROCA
Dieses Plugin wandelt die Ausgabe von NVIDIA-Replicator in das Format, welches [ROCA](https://github.com/drescherflo/ROCA) für das Training benötigt.
