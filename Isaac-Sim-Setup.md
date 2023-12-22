# Isaac Sim Setup
## Isaac Sim Installieren
- Omniverse Launcher installieren (Achtung, Firefox als snap geht nicht)
- Omniverse Launcher starten und anmelden
  - Achtung! Firefox ist unter Ubuntu 22.04 standardmäßig als snap installiert. Es kann sien, dass Firefox dann nicht mit dem Omniverse Launcher während des Logins kommunizieren kann. Sollte dem der Fall sein, muss ein Browser verwendet werden, welcher beispielsweise über eine *.deb installiert wurde
- Im Launcher unter `Exchange` folgende Apps installieren
  - `Omniverse Cache`
  - `Isaac Sim`
- Im Launcher unter `Nucleus` eine lokale Instanz einrichten

## Entwicklungsumgebung einrichten
Für das Debugging der Python-Skripte, welche für die Erstellung von Trainingsdaten und Simulationen verwendet werden, empfiehlt sich die Einrichtung einer Entwicklungsumgebung.

Die nachfolgenden Anleitungen gehen davon aus, dass Isaac Sim im Standardverzeichnis `/home/<user>/.local/share/ov/pkg/isaac_sim-2023.1.1` und in Version `2023.1.1` installiert wurde.
Ansonsten müssen die Pfade entsprechend angepasst werden.

Es wird folgendes Setup für die Entwicklung empfohlen:
Visual Studio Code dient als Debugger, da nicht alle Funktionen der Isaac Sim Packages erkannt und deshalb Fehler im Code gemeldet werden, welche keine sind.
Außerdem funktioniert auch IntelliSense (die Code-Komplettierung) nicht korrekt.
PyCharm hat diese Probleme nicht und wird entsprechend als Code-Editor eingesetzt.
Allerdings hat PyCharm einige Probleme mit den Plugins, die Isaac Sim beim Starten lädt, was zu einem unvollständigen Start von Isaac Sim führt.

Somit ergibt sich für optimales Entwickeln ein Setup von zwei Editoren / IDEs.
Nachfolgend sind die nötigen Einrichtungsschritte dokumentiert.

### Visual Studio Code
Im Ordner `.vscode` sind die nötigen Workspace-Konfigurationen abgelegt, die die direkte Verwendung des Roots dieses Repos als Projektordner erlaubt.
Hierbei handelt es sich um modifizierte Versionen der Konfigurationen im Verzeichnis `~/.local/share/ov/pkg/isaac_sim-2023.1.1/.vscode`.

Es sei darauf hingewiesen, dass Visual Studio Code nicht alle Funktionen der Isaac Sim Packages erkennt und deshalb Fehler im Code meldet, welche keine sind.
Dementsprechend funktioniert auch IntelliSense nicht korrekt.

### PyCharm
#### Python-Interpreter konfigurieren
- Unter `File` &rarr; `Settings` &rarr; `Project` &rarr; `Python Interpreter` &rarr; `Add Interpreter` &rarr; `Add Local Interpreter...`
- `Virtual Environment` wählen
  - Environment: `Existing`
  - Interpreter: `/home/<user>/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh`
  - Mit `OK` bestätigen
- Mit `OK` bestätigen

#### Fehlende Pakete konfigurieren
- Unter `File` &rarr; `Settings` &rarr; `Project` &rarr; `Python Interpreter` das Python-Interpreter-Drop-Down anklicken und `Show all...` auswählen
- Im Pop-Up rechts neben dem Filter-Symbol auf `Show Interpreter Paths` klicken
- Mit `+` alle Unterordner(!) der folgenden Verzeichnisse markieren und mit `OK` hinzufügen:
  - `~/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/`
  - `~/.local/share/ov/pkg/isaac_sim-2023.1.1/extscache/`
  - `~/.local/share/ov/pkg/isaac_sim-2023.1.1/extsPhysics`
  - `~/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/exts/`
  - `~/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/extscore/`
- Außerdem folgende Verzeichnisse zur Liste hinzufügen:
  - `~/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/kernel/py`
  - `~/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python3.10/site-packages`
- Alle Fenster durch Klick auf `OK` schließen

# Quellen
- https://forums.developer.nvidia.com/t/setup-pycharm-to-work-with-isaac-sim-solution/226050/4 (20.12.2023)
