# dataset_generator

The scripts in this folder generate synthetic training datasets using NVIDIA Isaac Sim.

## 6_dof_dataset_generator.py

The script `6_dof_dataset_generator.py` creates training datasets for applications with six degrees of freedom (6‑DOF). 
It uses a configuration file and USD models to build scenes for training. The data are captured by so‑called writers and written to disk.

### Arguments

- `--headless`: Runs the script in headless mode (no graphical user interface).
- `--output_dir`: **required**. Directory where the generated data will be stored.
- `--usd_dir`: **required**. Directory containing the USD (Universal Scene Description) versions of the CAD models used for data generation.
- `--config_file`: **required**. Path to the JSON configuration file that describes the scenes to be generated (created with `6_dof_config_generator.py`).
- `--writer`: **required**. Configures a writer from the `resumable_writers` plugin package. A writer is loaded only when its class name is spelled correctly. Writer arguments must follow the format `argument=(true|false)`; any value other than `true` is interpreted as `False`. This argument can be supplied multiple times.
- `--keep_cad_materials`: Renders the objects with the materials extracted from the CAD models instead of the randomized materials from the configuration file (see [Object materials](#object-materials)).

### Example command

```bash
python 6_dof_dataset_generator.py \
  --headless \
  --output_dir "/path/to/output" \
  --usd_dir "/path/to/usd" \
  --config_file "/path/to/config.json" \
  --writer Writer1 writer1_arg=true writer1_arg2=false \
  --writer Writer2
```

### Important notes

- All required arguments (`--output_dir`, `--usd_dir`, `--config_file`, `--writer`) must be provided correctly for the script to run successfully.
- The `--headless` flag is optional and useful in environments without a graphical UI.
- The functionality can be extended with additional writers defined in the `resumable_writers` plugin package. Any new writer must inherit from the abstract class `ResumableWriterInterface`.

## conveyor_belt_dataset_generator.py

The script `conveyor_belt_dataset_generator.py` generates training datasets for conveyor‑belt scenarios. 
It also uses a configuration file and USD models to build training scenes, captures data with writers, and writes the results to disk.

### Arguments

- --`headless`: Runs the script in headless mode (no default value).
- --`output_dir`: required. Output directory for the generated data (no default).
- --`usd_dir`: required. Directory containing the USD versions of the CAD models used for data generation (no default).
- --`config_file`: required. Path to the JSON configuration file describing the scenes to generate (no default).
- --`writer`: required. Configures a writer from the `resumable_writers` plugin package. A writer is loaded only when its class name is spelled correctly. Writer arguments must follow the format `argument=(true|false)`; any value other than `true` is interpreted as `False`. This argument can be supplied multiple times.
- --`keep_cad_materials`: Renders the objects with the materials extracted from the CAD models instead of the randomized materials from the configuration file (see [Object materials](#object-materials)).

### Example command

```bash
python conveyor_belt_dataset_generator.py \
  --headless \
  --output_dir "/path/to/output_directory" \
  --usd_dir "/path/to/usd_directory" \
  --config_file "/path/to/config.json" \
  --writer Writer1 writer1_arg=true writer1_arg2=false \
  --writer Writer2
```

### Important notes

- All required arguments (`--output_dir`, `--usd_dir`, `--config_file`, `--writer`) must be provided correctly for the script to run successfully.
- The `--headless` flag is optional and useful in environments without a graphical UI.
- The functionality can be extended with additional writers defined in the `resumable_writers` plugin package. Any new writer must inherit from the abstract class `ResumableWriterInterface`.

## Object materials

Both generators support two ways of shading the objects:

- **Randomized materials (default)**: The materials listed under `materials` in the configuration file are created on the stage and assigned per frame according to the `material_idx` of each object. This is the domain randomization used for training.
- **CAD materials (`--keep_cad_materials`)**: No material is assigned by the generator, so the objects keep the material their USD model carries. That material originates from the colour `step_to_obj.py` extracts from the STEP file and writes into the accompanying MTL file. Use this to render the objects in their real appearance.

The two options are mutually exclusive because `apply_visual_materials()` binds a material as *stronger than descendants*, which overrides the material bound inside the USD model. With `--keep_cad_materials` the `materials` section of the configuration file is ignored.

## Implemented Writers

### ResumableBasicWriter

During dataset generation the simulation is reset multiple times, which would cause NVIDIA’s BasicWriter to overwrite previously generated training data.

ResumableBasicWriter wraps NVIDIA’s BasicWriter and prevents overwriting of already generated data. 
Only boolean‑type arguments are supported for initializing the BasicWriter via command‑line arguments. 
The BasicWriter arguments are documented in the corresponding [API documentation](https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/omni.replicator.core/docs/API.html#basicwriter).

### WorldPoseWriter

WorldPoseWriter records the position and orientation of every visible object in the camera image together with its semantic label. 
It does not accept any command‑line arguments.

