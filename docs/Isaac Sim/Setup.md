# Isaac Sim Setup
## Isaac Sim Installieren
- Omniverse Launcher von [hier](https://www.nvidia.com/de-de/omniverse/download/) herunterladen
  - Direktlink für Linux: https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage
  - Direktlink für Windows: https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-win.exe
  ```bash
  curl -sSL https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage -o ~/Downloads/omniverse-launcher-linux.AppImage
  ```

- libfuse2 installieren
  ```bash
  sudo apt update
  sudo apt install libfuse2
  ```

- Omniverse Launcher ausführen
  ```bash
  chmod +x ~/Downloads/omniverse-launcher-linux.AppImage
  ~/Downloads/omniverse-launcher-linux.AppImage
  ```

- Im Omniverse Launcher mit NVIDIA Developer Account anmelden
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

## Lokale ROS2-Installation in Isaac Sim verwenden
- `vision_msgs` installieren
  ```bash
  sudo apt install ros-humble-vision-msgs
  ```
- Falls noch nicht automatisch durch Eintrag in '~/.bashrc' geschehen: ROS-Umgebung laden
  ```bash
  source /opt/ros/humble/setup.bash
  ```
- Datei `fastdds.xml` in `~/.ros` erstellen und korrekten Inhalt eintragen
  ```bash
  mkdir ~/.ros
  echo '
  <?xml version="1.0" encoding="UTF-8" ?>

  <license>Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
  NVIDIA CORPORATION and its licensors retain all intellectual property
  and proprietary rights in and to this software, related documentation
  and any modifications thereto.  Any use, reproduction, disclosure or
  distribution of this software and related documentation without an express
  license agreement from NVIDIA CORPORATION is strictly prohibited.</license>


  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles" >
      <transport_descriptors>
          <transport_descriptor>
              <transport_id>UdpTransport</transport_id>
              <type>UDPv4</type>
          </transport_descriptor>
      </transport_descriptors>

      <participant profile_name="udp_transport_profile" is_default_profile="true">
          <rtps>
              <userTransports>
                  <transport_id>UdpTransport</transport_id>
              </userTransports>
              <useBuiltinTransports>false</useBuiltinTransports>
          </rtps>
      </participant>
  </profiles>
  ' > ~/.ros/fastdds.xml
  ```
- .bashrc anpassen und neuladen
  ```bash
  echo 'export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml' >> ~/.bashrc
  source ~/.bashrc
  ```
- Falls NUCLEUS-Launcher verwendet wird: Im NUCLEUS-Launcher unter `Extra Args` `export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml` eintragen

# Quellen
- https://forums.developer.nvidia.com/t/setup-pycharm-to-work-with-isaac-sim-solution/226050/4 (20.12.2023)
- https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_ros.html#running-native-ros (09.01.2024)