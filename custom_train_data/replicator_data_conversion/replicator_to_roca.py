import shutil
import sys
import os
import glob
import json
import argparse
from PIL import Image
import numpy as np


def get_scene_nrs(rep_data_path: str) -> list[str]:
    """
    Get scene numbers from replicator dataset

    Args:
    - rep_data_path : Path to the replicator dataset containing camera_params_*.json files

    Returns: list of scene numbers (list[str])
    """

    # Find all JSON files matching the pattern camera_params_*.json
    json_files = glob.glob(os.path.join(rep_data_path, "camera_params_*.json"))

    # Create and return list of scene numbers
    return [os.path.basename(json_file).split('_')[2].split('.')[0] for json_file in json_files]


def replicator_intrinsics_to_scannet(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
    """
    Generate intrinsics_color.txt files from camera parameters JSON files.

    Args:
    - rep_data_path (str): Path to the folder containing camera_params_*.json files.
    - roca_data_path (str): Path to the output folder where ScanNet25k/tasks/scannet_frames_25k/scene{NR}/intrinsics_color.txt files will be saved.
    - scene_numbers (list[str]): List of scene numbers
    """

    # Iterate through all scene_numbers
    for nr in scene_numbers:
        # Read the contents of the JSON file matching the pattern camera_params_*.json
        with open(os.path.join(rep_data_path, f"camera_params_{nr}.json"), 'r') as file:
            camera_params = json.load(file)

        # Extract needed parameters
        camera_focal_length = camera_params['cameraFocalLength']
        render_product_resolution = camera_params['renderProductResolution']
        camera_aperture_width = camera_params['cameraAperture'][0]

        # Calculate f_x, f_y, c_x, c_y
        f_x = f_y = (camera_focal_length * render_product_resolution[0]) / camera_aperture_width
        c_x = render_product_resolution[0] / 2
        c_y = render_product_resolution[1] / 2

        # Prepare the output directory
        output_dir = os.path.join(roca_data_path, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}")
        os.makedirs(output_dir, exist_ok=True)

        # Write to intrinsics_color.txt
        intrinsics_file_path = os.path.join(output_dir, "intrinsics_color.txt")
        with open(intrinsics_file_path, 'w') as intrinsics_file:
            intrinsics_file.write(f"{f_x} 0 {c_x} 0\n0 {f_y} {c_y} 0\n0 0 1 0\n0 0 0 1\n")


def replicator_image_to_scannet(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
    """
    Copy and convert images from PNG to JPG format and rename them.

    Args:
    - rep_data_path (str): Path to the folder containing rgb_{nr}.png files.
    - roca_data_path (str): Path to the output folder where images will be saved in the format ScanNet25k/tasks/scannet_frames_25k/scene{NR}/color/000000.jpg.
    - scene_numbers (list[str]): List of scene numbers
    """

    # Iterate through all scene_numbers
    for nr in scene_numbers:
        # Prepare the output directory
        output_dir = os.path.join(roca_data_path, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}/color")
        os.makedirs(output_dir, exist_ok=True)

        # Set the new file path
        new_file_path = os.path.join(output_dir, "000000.jpg")

        # Open the PNG file and convert it to JPG
        with Image.open(os.path.join(rep_data_path, f"rgb_{nr}.png")) as img:
            img.convert('RGB').save(new_file_path, 'JPEG')


def copy_obj_files(roca_data_path: str, obj_path: str) -> None:
    """
    Copy .obj files to the roca data path respecting the ShapeNet data structure.

    Args:
    - rep_data_path (str): Not used in this function, but included for consistency.
    - roca_data_path (str): ROCA data path where .obj files will be copied to.
    - obj_path (str): Path to the folder containing [ModelName].obj files.
    """

    # Find all .obj files in the obj_path
    obj_files = glob.glob(os.path.join(obj_path, "*.obj"))

    for obj_file in obj_files:
        # Extract the model name from the file name
        model_name = os.path.basename(obj_file).split('.')[0]

        # Prepare the output directory
        output_dir = os.path.join(roca_data_path, f"ShapeNetCore.v2/{model_name}/0/models")
        os.makedirs(output_dir, exist_ok=True)

        # Set the new file path
        new_file_path = os.path.join(output_dir, "model_normalized.obj")

        # Copy the .obj file
        shutil.copy(obj_file, new_file_path)


def main(args: list[str]) -> None:
    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts the generated training data from NVIDIA Replicator to the format required by ROCA")
    parser.add_argument("--rep_dir", help="Input directory containing the files in the Replicator format", required=True)
    parser.add_argument("--roca_dir", help="Directory that ROCA will use for training data generation", required=True)
    parser.add_argument("--obj_dir", help="Directory with all CAD Models in OBJ format", required=True)

    args = parser.parse_args(args)

    # Test if rep_dir and obj_dir exist
    if not os.path.isdir(args.rep_dir):
        print(f"The NVIDIA replicator directory {args.rep_dir} does not exist. Existing...")
        exit(-1)
    if not os.path.isdir(args.obj_dir):
        print(f"The OBJ model directory {args.rep_dir} does not exist. Existing...")
        exit(-1)

    # Test if roca_dir exists and create if necessary
    if not os.path.isdir(args.roca_dir):
        os.makedirs(args.roca_dir)

    # Convert training data
    replicator_dir = args.rep_dir
    obj_dir = args.obj_dir
    roca_dir = args.roca_dir
    scene_numbers = get_scene_nrs(replicator_dir)
    print("Converting camera intrinsics...")
    replicator_intrinsics_to_scannet(replicator_dir, roca_dir, scene_numbers)
    print("Converting images...")
    replicator_image_to_scannet(replicator_dir, roca_dir, scene_numbers)
    print("Copying CAD models...")
    copy_obj_files(roca_dir, obj_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
