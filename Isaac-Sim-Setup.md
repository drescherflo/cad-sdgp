# Isaac Sim Setup
## Isaac Sim Installieren
- Omniverse Launcher installieren (Achtung, Firefox als snap geht nicht)
- Omniverse Launcher starten und anmelden
  - Achtung! Firefox ist unter Ubuntu 22.04 standardmäßig als snap installiert. Es kann sien, dass Firefox dann nicht mit dem Omniverse Launcher während des Logins kommunizieren kann. Sollte dem der Fall sein, muss ein Browser verwendet werden, welcher beispielsweise über eine *.deb installiert wurde
- Im Launcher unter `Exchange` folgende Apps installieren
  - `Omniverse Cache`
  - `Isaac Sim` (Version 2022.2.1, weil ros2_bridge unter 2023.* nicht startet)

## Entwicklungsumgebung einrichten
Für das Debugging der Python-Skripte, welche für die Erstellung von Trainingsdaten und Simulationen verwendet werden, empfiehlt sich die Einrichtung einer Entwicklungsumgebung.

Die nachfolgenden Anleitungen gehen davon aus, dass Isaac Sim im Standardverzeichnis `/home/<user>/.local/share/ov/pkg/isaac_sim-2022.2.1` und in Version `2022.2.1` installiert wurde.
Ansonsten müssen die Pfade entsprechend angepasst werden.

### Visual Studio Code
Im Ordner `.vscode` sind die nötigen Workspace-Konfigurationen abgelegt, die die direkte Verwendung des Roots dieses Repos als Projektordner erlaubt.
Hierbei handelt es sich um modifizierte Versionen der Konfigurationen im Verzeichnis `~/.local/share/ov/pkg/isaac_sim-2022.2.1/.vscode`.

Es sei darauf hingewiesen, dass Visual Studio Code nicht alle Funktionen der Isaac Sim Packages erkennt und deshalb Fehler im Code meldet, welche keine sind.
Dementsprechend funktioniert auch IntelliSense nicht korrekt.

### PyCharm
#### Python-Interpreter konfigurieren
- Unter `File` &rarr; `Settings` &rarr; `Project` &rarr; `Python Interpreter` &rarr; `Add Interpreter` &rarr; `Add Local Interpreter...`
- `Virtual Environment` wählen
  - Environment: `Existing`
  - Interpreter: `/home/<user>/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh`
  - Mit `OK` bestätigen
- Mit `OK` bestätigen

#### Fehlende Pakete konfigurieren
- Unter `File` &rarr; `Settings` &rarr; `Project` &rarr; `Python Interpreter` das Python-Interpreter-Drop-Down anklicken und `Show all...` auswählen
- Im Pop-Up rechts neben dem Filter-Symbol auf `Show Interpreter Paths` klicken
- Mit `+` alle Unterordner(!) der folgenden Verzeichnisse markieren und mit `OK` hinzufügen (sofern verfügbar)
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/exts/`
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/extscache/`
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/extensions/`
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/exts/`
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/extsPhysics/`
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/extscore/`
- Außerdem folgende Verzeichnisse zur Liste hinzufügen (sofern verfügbar):
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/extensions/extensions-bundled`
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/kernel/py`
  - `~/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/plugins/bindings-python`
- Alle Fenster durch Klick auf `OK` schließen

# Quellen
- https://forums.developer.nvidia.com/t/setup-pycharm-to-work-with-isaac-sim-solution/226050/4 (20.12.2023)
