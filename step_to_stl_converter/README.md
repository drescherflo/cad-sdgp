### Gebrauchsanweisung für das Python-Skript zur Konvertierung von STEP-Dateien in STL-Dateien

Das Skript `step_to_stl.py` ermöglicht die Konvertierung von 3D-Modellen im STEP-Format (`.stp`) in das STL-Format (`.stl`). 
Das Skript wird über die Kommandozeile gesteuert und erfordert die Angabe von Eingabe- und Ausgabeverzeichnissen.

#### Voraussetzungen:
- Python muss auf Ihrem System installiert sein.
- Anaconda muss auf Ihrem System installiert sein.
- Die Python-Bibliotheken `pythonocc-core` muss installiert sein. 

Wurde Anaconda ([hier herunterladen](https://www.anaconda.com/download)) installiert, kann die Conda-Umgebung durch das Laden von `setup.sh` eingerichtet und geladen werden.

```shell
source setup.sh
```

#### Benutzung:

1. **Öffnen Sie das Terminal:** Wechseln Sie zum Verzeichnis, in dem das Skript gespeichert ist.
2. **Laden Sie die Conda-Umgebung**: Verwenden Sie den folgenden Befehl:
   ```bash
   conda activate step_to_stl
   ``` 
4. **Führen Sie das Skript aus:** Verwenden Sie den folgenden Befehl, wobei Sie die Pfade zu den Verzeichnissen entsprechend anpassen:
   ```bash
   python step_to_stl.py --input_dir [Pfad_zum_Eingabe-Verzeichnis] --output_dir [Pfad_zum_Ausgabe-Verzeichnis]
   ```
   - `--input_dir`: Pfad zum Verzeichnis, das die `.stp`-Dateien enthält.
   - `--output_dir`: Pfad zum Verzeichnis, in das die konvertierten `.stl`-Dateien gespeichert werden sollen.

#### Beispiel:

```
python step_to_stl.py --input_dir ~/STEP --output_dir ~/STL
```

#### Wichtige Hinweise:

- Die Bibliothek `pythonocc-core` kann nur mittels Anaconda installiert werden.
- Das Skript liest alle `.stp`-Dateien aus dem angegebenen Eingabeverzeichnis und speichert die konvertierten `.stl`-Dateien im Ausgabeverzeichnis.
- Wenn das Ausgabeverzeichnis nicht existiert, wird es automatisch erstellt.
- Das Skript beendet sich mit einer Fehlermeldung, wenn kein Eingabe- oder Ausgabeverzeichnis angegeben wird oder das Eingabeverzeichnis nicht existiert.

#### Fehlerbehandlung:

- Stellen Sie sicher, dass die Pfade zu den Verzeichnissen korrekt sind.
- Überprüfen Sie, ob die erforderlichen Python-Bibliotheken installiert sind.
- Das Skript unterstützt nur die Konvertierung von `.stp`-Dateien. Andere Dateiformate werden ignoriert.