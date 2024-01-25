# Trainingsdatenerstellungspipeline

- Pipeline kann entweder nativ ausgeführt werden oder über Container
- Verwendung von Containern wird empfohlen, da deutlich weniger Abhängigkeiten installiert werden müssen, welche nicht automatisiert installiert werden können
- Benötigte Container-Images stehen auf Dockerhub zur Verfügung (mit Ausnahme von NVIDIA Isaac Sim Images)
- Alternativ können Images gebaut werden mit
  ```bash
  docker compose --profile manual build
  ```
  
- Sollen Container nicht verwendet werden stehen READMEs in den Ordnern der einzelnen Komponenten mit Installationsanleitungen zur Verfügung.
- Argumente für Python-Skripte können mit Flag -h angezeigt werden und sind in den Skripten selbst dokumentiert

## Voraussetzungen
- Pipeline verwendet für Erzeugung der Trainingsdaten NVIDIA Isaac Sim 2023.1.1
- Dieser benötigt:
  - NVIDIA RTX GPU
  - Installierten NVIDIA Treiber
  - NVIDIA Developer Account

## Voraussetzungen für Verwendung von Pipeline mit Containern
- Container-Engine (nachfolgend wird Docker verwendet, Verwendung von Podman sollte auch möglich sein, wurde aber nicht getestet)
- NVIDIA Container Toolkit
- Zugriff auf das Isaac Sim Container Image

### Docker installieren (Linux Ubuntu)
- Repository hinzufügen
  ```bash
  # Add Docker's official GPG key:
  sudo apt-get update
  sudo apt-get install ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  
  # Add the repository to Apt sources:
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  ```

- Paketliste aktualisieren
  ```bash
  sudo apt update
  ```

- Docker installieren
  ```bash
  sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  ```
  
- Aktuellen Nutzer zur Gruppe `docker` hinzufügen
  ```bash
  sudo usermod -aG docker $USER
  ```

- PC neustarten


### NVIDIA Container Toolkit installieren (Linux Ubuntu)
- Repository hinzufügen
  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  ```

- Paketliste aktualisieren
  ```bash
  sudo apt update
  ```

- NVIDIA Container Toolkit installieren
  ```bash
  sudo apt install -y nvidia-container-toolkit
  ```

- Container-Runtime konfigurieren
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  ```

- Docker daemon neustarten
  ```bash
  sudo systemctl restart docker
  ```

### Zugriff auf Isaac Sim Image erhalten
- Unter https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim auf 'Get Container' klicken
- Mit NVIDIA Developer Account anmelden oder registrieren
- Rechts oben auf `Benutername` &rarr; `Setup` klicken
- Auf `Get API Key` klicken
- Auf `Generate API Key klicken`
- Bei NVIDIA Container Registry anmelden
```bash
docker login nvcr.io

Username: $oauthtoken
Password: <API-Key>
```

## Verwendung
### Trainingsdatensatz mit Containern generieren
- CAD-Modelle im STEP-Format in Verzeichnis `~/custom_dataset/cad_models/STEP` legen
- Pipeline ausführen
  ```bash
  ./build_custom_dataset_docker.sh
  ```
- Datensatz für ROCA liegt unter `~/custom_dataset/ReplicatorToROCA`
- Soll Datensatz für natives Training und nicht in einem Container verwendet werden, kann es nötig sein, die Eigentümer der Daten zu korrigieren
  ```bash
  cd ~/custom_dataset
  sudo chown -R $(whoami) ./
  ```

# Quellen
- https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html (24.01.2023)
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html (24.01.2023)
- https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html (24.01.2023)
- https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository (25.01.2023)
- https://docs.docker.com/engine/install/linux-postinstall/ (25.01.2023)
