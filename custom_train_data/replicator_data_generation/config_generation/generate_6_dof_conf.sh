#!/bin/bash

python3 data_generation_config_generator.py --usd_dir "../../../CAD Models/OBJ_converted" --writer ResumableBasicWriter rgb=True camera_params=True --writer WorldPoseWriter --out_path generated_configs/6_dof_only_simple_object.json --num_frames_per_object 10
