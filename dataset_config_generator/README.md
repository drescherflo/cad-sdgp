# dataset_config_generator

Contains scripts for generating JSON configuration files. These configuration files precisely define how a frame in the synthetic dataset to be generated should look. They specify, among other things, the position and intensity of lights.

## 6_dof_config_generator.py

The script `6_dof_config_generator.py` generates a configuration file for a generic dataset where objects float freely in space in front of the camera.

### Arguments

- `--usd_dir`: **required**. Path to the directory containing USD files. No default.
- `--out_path`: Output path for the configuration file. Default is `"config.json"`.
- `--frame_width`: Width of the camera frame in pixels. Default is `480`.
- `--frame_height`: Height of the camera frame in pixels. Default is `360`.
- `--sub_frames_per_frame`: Number of sub‑frames per frame. Default is `32`.
- `--num_random_materials`: Number of random materials. Default is `100`.
- `--probability_of_glass_material`: Probability of selecting a glass material. Default is `0.5`.
- `--num_frames_per_object`: Number of frames per object. Default is `1000`.
- `--num_objects_per_frame`: Number of objects in the scene. Default is `20`.
- `--num_sphere_lights`: Number of spherical lights. Default is `5`.
- `--sphere_min_intensity`: Minimum intensity of the spherical lights. Default is `5000`.
- `--sphere_max_intensity`: Maximum intensity of the spherical lights. Default is `50000`.
- `--train_val_split`: Fraction of data reserved for validation. Default is `0.2`.
- `--min_x`: Minimum X coordinate for object placement. Default is `-2`.
- `--max_x`: Maximum X coordinate for object placement. Default is `2`.
- `--min_y`: Minimum Y coordinate for object placement. Default is `-1`.
- `--max_y`: Maximum Y coordinate for object placement. Default is `1`.
- `--cam_distance_to_background`: Distance of the camera to the background plane. Default is `5`.
- `--dome_light_min_intensity`: Minimum intensity of the dome lighting. Default is `5000`.
- `--dome_light_max_intensity`: Maximum intensity of the dome lighting. Default is `50000`.

### Example command

```bash
python 6_dof_config_generator.py \
  --usd_dir "/path/to/directory" \
  --out_path "/path/to/config.json" \
  --frame_width 1920 \
  --frame_height 1080 \
  --sub_frames_per_frame 5 \
  --num_frames_per_object 10 \
  --num_objects_per_frame 3 \
  --num_random_materials 20 \
  --num_sphere_lights 5 \
  --sphere_min_intensity 5000.0 \
  --sphere_max_intensity 50000.0 \
  --train_val_split 0.2 \
  --probability_of_glass_material 0.3 \
  --cam_distance_to_background 10.0 \
  --min_x -5.0 \
  --max_x 5.0 \
  --min_y -5.0 \
  --max_y 5.0 \
  --dome_light_min_intensity 5000.0 \
  --dome_light_max_intensity 50000.0
```

### Important notes

- The script checks that the supplied usd_dir exists and contains USD files.
- The `out_path` must point to a valid location; if the file already exists, the script aborts.
- `cam_distance_to_background` must be greater than 0.
- `train_val_split` should be between 0 and 1.

## conveyor_config_generator.py

The script `conveyor_config_generator.py` creates a configuration file for generating a dataset in a conveyor‑belt scenario, covering object, camera, lighting, and material settings.

### Arguments

- `--usd_dir`: **required**. Directory with the USD files of the CAD models. No default.
- `--out_path`: Output path for the configuration file. Default is `config.json`.
- `--frame_width`: Width of the generated images in pixels. Default is `480`.
- `--frame_height`: Height of the generated images in pixels. Default is `360`.
- `--sub_frames_per_frame`: Frames inserted between saved frames to avoid artifacts. Default is `32`.
- `--num_random_materials`: Number of randomly generated materials. Default is `1000`.
- `--probability_of_glass_material`: Probability of generating glass materials. Default is `0.5`.
- `--num_frames_per_scene`: Frames per scene. The default is computed automatically.
- `--num_objects_per_scene`: Objects per scene. Default is `100`.
- `--num_scenes_per_object`: Scenes per object in the USD directory. Default is `10`.
- `--num_sphere_lights`: Spherical lights with random colour per frame. Default is `5`.
- `--train_val_split`: Fraction of the dataset used for validation. Default is `0.2`.
- `--object_init_min_x`: Minimum X coordinate for the initial object position. Default is `-2.0`.
- `--object_init_max_x`: Maximum X coordinate for the initial object position. Default is `-1.5`.
- `--object_init_min_y`: Minimum Y coordinate for the initial object position. Default is `-0.4`.
- `--object_init_max_y`: Maximum Y coordinate for the initial object position. Default is `0.4`.
- `--object_init_min_z`: Minimum Z coordinate for the initial object position. Default is `3`.
- `--object_init_max_z`: Maximum Z coordinate for the initial object position. Default is `6`.
- `--sphere_min_x`: Minimum X coordinate for the spherical lights. Default is `-2.5`.
- `--sphere_max_x`: Maximum X coordinate for the spherical lights. Default is `2.5`.
- `--sphere_min_y`: Minimum Y coordinate for the spherical lights. Default is `-0.5`.
- `--sphere_max_y`: Maximum Y coordinate for the spherical lights. Default is `0.5`.
- `--sphere_min_z`: Minimum Z coordinate for the spherical lights. Default is `2.3`.
- `--sphere_max_z`: Maximum Z coordinate for the spherical lights. Default is `2.7`.
- `--sphere_min_intensity`: Minimum intensity of the spherical lights. Default is `1000`.
- `--sphere_max_intensity`: Maximum intensity of the spherical lights. Default is `5000`.
- `--camera_pos_min_x`: Minimum X coordinate for the camera position. Default is `-0.5`.
- `--camera_pos_max_x`: Maximum X coordinate for the camera position. Default is `0.5`.
- `--camera_pos_min_y`: Minimum Y coordinate for the camera position. Default is `-0.3`.
- `--camera_pos_max_y`: Maximum Y coordinate for the camera position. Default is `0.3`.
- `--camera_pos_min_z`: Minimum Z coordinate for the camera position. Default is `3`.
- `--camera_pos_max_z`: Maximum Z coordinate for the camera position. Default is `6`.
- `--camera_rot_min_x`: Minimum rotation of the camera around the X‑axis. Default is `-180`.
- `--camera_rot_max_x`: Maximum rotation of the camera around the X‑axis. Default is `180`.
- `--camera_rot_min_y`: Minimum rotation of the camera around the Y‑axis. Default is `-120`.
- `--camera_rot_max_y`: Maximum rotation of the camera around the Y‑axis. Default is `-60`.
- `--camera_rot_min_z`: Minimum rotation of the camera around the Z‑axis. Default is `-180`.
- `--camera_rot_max_z`: Maximum rotation of the camera around the Z‑axis. Default is `180`.
- `--distant_light_min_rot_x`: Minimum rotation of the distant light around the X‑axis. Default is `-90`.
- `--distant_light_max_rot_x`: Maximum rotation of the distant light around the X‑axis. Default is `90`.
- `--distant_light_min_rot_y`: Minimum rotation of the distant light around the Y‑axis. Default is `-90`.
- `--distant_light_max_rot_y`: Maximum rotation of the distant light around the Y‑axis. Default is `90`.
- `--distant_light_min_rot_z`: Minimum rotation of the distant light around the Z‑axis. Default is `-180`.
- `--distant_light_max_rot_z`: Maximum rotation of the distant light around the Z‑axis. Default is `180`.
- `--distant_light_min_intensity`: Minimum intensity of the distant light. Default is `200`.
- `--distant_light_max_intensity`: Maximum intensity of the distant light. Default is `1000`.
- `--conveyor_belt_speed`: Speed of the conveyor belt in the simulation (default derived from preset values).
- `--min_x_pos_for_record_start`: Minimum X coordinate at which data recording starts (default derived from preset values).
- `--render_frequency`: Rendering frequency in Hz. Default is `60`.
- `--physics_frequency`: Frequency at which physics calculations are performed. Default is `360`.

### Example command

```bash
python conveyor_config_generator.py \
  --usd_dir "/path/to/usd_models" \
  --out_path "my_config.json"
```

### Important notes

- The script verifies that the provided `usd_dir` exists and contains USD files.
- The `out_path` must point to a writable location; if the file already exists, the script aborts.
- `train_val_split` should be a value between 0 and 1.
