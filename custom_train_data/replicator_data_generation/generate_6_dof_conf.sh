#!/bin/bash

python3 data_generation_config_generator.py --usd_dir "../../CAD Models/OBJ_converted" --writer BasicWriter output_dir=out_dir rgb=True camera_params=True image_output_format="png" --out_path generated_configs/6_dof_only_simple_object.json
