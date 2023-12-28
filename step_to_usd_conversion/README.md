# Konvertierung von CAD-Modellen von STEP nach USD
Es existiert keine direkte Konvertierungsmöglichkeit.
Aus diesem Grund besteht der Konvertierungsprozess aus zwei Schritten, nämlich der Konvertierung von STEP nach STL und von STL nach USD.

Für den ersten Schritt wurde das Skript `step_to_stl.py` entwickelt.
Für den zweiten Schritt bietet NVIDIA unter `~/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/api/omni.kit.asset_converter` das Skript `asset_usd_converter.py` an.
Dieses darf aufgrund der eingetragenen Lizenz nicht verändert oder verteilt werden.

### Konvertierung von STEP-Dateien in STL-Dateien

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
3. **Führen Sie das Skript aus:** Verwenden Sie den folgenden Befehl, wobei Sie die Pfade zu den Verzeichnissen entsprechend anpassen:
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

### Konvertierung von STEP-Dateien in STL-Dateien
Dieses Skript konvertiert 3D-Modelle im STL, OBJ oder FBX Format in das USD Format. Es ist für die Verwendung mit dem NVIDIA Omniverse Kit entwickelt.

#### Voraussetzungen
- NVIDIA Omniverse Isaac Sim 2023.1.1 muss installiert sein.

#### Verwendung

1. **Führen Sie das Skript aus.**
   Verwenden Sie die folgende Kommandozeile, um das Skript zu starten:

   ```bash
   ~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh ~/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py --folders [Ordnerpfade] [--max-models [Maximale Anzahl]] [--load-materials]
   ```

   - `--folders`: Liste der Ordner, die konvertiert werden sollen (durch Leerzeichen getrennt).
   - `--max-models`: Maximale Anzahl von Modellen pro Ordner, die konvertiert werden sollen (optional). Der Standardwert ist 50.
   - `--load-materials`: Wenn angegeben, werden Materialien aus den Meshes geladen (optional).

2. **Überprüfen Sie die Ergebnisse.**
   Das Skript speichert die konvertierten USD-Modelle in den entsprechenden Ausgabeordnern.
   Die Ausgabeordner haben den Zusatz `_converted` im Namen.

#### Beispiel:

```
  ~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh ~/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py --folders ~/STL
```
