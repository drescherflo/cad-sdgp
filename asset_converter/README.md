# asset_converter

The scripts contained here can be used to convert CAD models from STEP to OBJ or from OBJ to USD.

## step_to_obj.py

The script `step_to_obj.py` converts STEP files (`.stp` or `.step`) into OBJ files (`.obj`).

### Arguments

- `--input_dir`: **required**. Directory that contains the STEP files.  
- `--output_dir`: **required**. Destination directory for the converted OBJ files.  
- `--scale_factor`: Optional scaling factor for the OBJ files. The default value is `0.001`.

### Example command

```bash
python step_to_obj.py --input_dir "/path/to/input-directory" --output_dir "/path/to/output-directory" --scale_factor 0.01
```

### Important notes

- The paths to the directories must exist and be specified correctly.
- During conversion temporary STL files are created as part of the process; they are automatically deleted once the conversion finishes.
- The default scaling factor of 0.001 is typical when the original model was designed in millimetres Adjust the scaling factor as needed.

## obj_to_usd.py

The script `obj_to_usd.py` converts OBJ files (.obj) into USD files (.usd).

### Arguments

- `--input_dir`: **required**. Directory that contains the OBJ files.
- `--output_dir`: **required**. Destination directory for the converted USD files.

### Example command

```bash
python obj_to_usd.py --input_dir "/path/to/input-directory" --output_dir "/path/to/output-directory"
```

### Important notes

- The paths to the directories must exist and be specified correctly.

## Converting multiple obj files

- activate Isaac Sim Python environment and run command (adjust paths)

```bash
~/isaacsim/python.sh \
  ~/ros2_ws/src/synthetic-data-generation-pipeline/asset_converter/obj_to_usd_recursive.py \
  --input_root  ~/ros2_ws/resource/EGAD/train \
  --output_root ~/ros2_ws/resource/EGAD_usd/train
```

the script will automatically launch isaac sim in headless mode and convert all .obj files in folder input_root to .usd files and place them in output_root