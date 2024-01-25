# Trainingsdatenerstellungspipeline

- Docker-Images stehen auf Dockerhub zur Verfügung (mit Ausnahme von NVIDIA Isaac Sim Images)
- Alternativ können Images gebaut werden mit
  ```bash
  docker compose --profile manual build
  ```
  
- Soll Docker nicht verwendet werden stehen READMEs in den Ordnern der einzelnen Komponenten mit Installationsanleitungen zur Verfügung.
- Argumente für Python-Skripte können mit Flag -h angezeigt werden oder sind in den Skripten selbst dokumentiert

## Voraussetzungen
### Isaac Sim Docker Images
- benötigen NVIDIA RTX GPU
- benötigen installierten NVIDIA Treiber
- benötigen NVIDIA Container Toolkit
- NVIDIA Developer Account

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
  sudo apt-get install -y nvidia-container-toolkit
  ```

- Container-Runtime konfigurieren
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  ```

- Docker daemon neustarten
  ```bash
  sudo systemctl restart docker
  ```

### Zugriff auf Isaac Sim Container erhalten
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

## Trainingsdatensatz generieren
- CAD-Modelle im STEP-Format in Verzeichnis `~/custom_dataset/cad_models/STEP` legen
- Pipeline ausführen
  ```bash
  ./build_custom_dataset_docker.sh
  ```
- Datensatz für ROCA liegt unter `~/custom_dataset/ReplicatorToROCA`

# Quellen
- https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html (24.01.2023)
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html (24.01.2023)
- https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html (24.01.2023)
