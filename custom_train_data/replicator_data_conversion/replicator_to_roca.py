import sys
import os
import glob
import json
import argparse
from PIL import Image
import numpy as np


def rep_intrinsics_to_scannet(rep_data_path: str, roca_data_path: str) -> None:
    """
    Generate intrinsics_color.txt files from camera parameters JSON files.

    Args:
    - rep_data_path (str): Path to the folder containing camera_params_*.json files.
    - roca_data_path (str): Path to the output folder where ScanNet25k/tasks/scannet_frames_25k/scene{NR}/intrinsics_color.txt files will be saved.
    """

    # Find all JSON files matching the pattern camera_params_*.json
    json_files = glob.glob(os.path.join(rep_data_path, "camera_params_*.json"))

    for json_file in json_files:
        # Extract the number (NR) from the file name
        nr = os.path.basename(json_file).split('_')[2].split('.')[0]

        # Read the contents of the JSON file
        with open(json_file, 'r') as file:
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


def rep_image_to_scannet(rep_data_path: str, roca_data_path: str) -> None:
    """
    Copy and convert images from PNG to JPG format and rename them.

    Args:
    - rep_data_path (str): Path to the folder containing rgb_{nr}.png files.
    - roca_data_path (str): Path to the output folder where images will be saved in the format ScanNet25k/tasks/scannet_frames_25k/scene{NR}/color/000000.jpg.
    """

    # Find all PNG files matching the pattern rgb_*.png
    png_files = glob.glob(os.path.join(rep_data_path, "rgb_*.png"))

    for png_file in png_files:
        # Extract the number (NR) from the file name
        nr = os.path.basename(png_file).split('_')[1].split('.')[0]

        # Prepare the output directory
        output_dir = os.path.join(roca_data_path, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}/color")
        os.makedirs(output_dir, exist_ok=True)

        # Set the new file path
        new_file_path = os.path.join(output_dir, "000000.jpg")

        # Open the PNG file and convert it to JPG
        with Image.open(png_file) as img:
            img.convert('RGB').save(new_file_path, 'JPEG')


def main(args: list[str]) -> None:
    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts the generated training data from NVIDIA Replicator to the format required by ROCA")
    parser.add_argument("--rep_dir", help="Input directory containing the files in the Replicator format", required=True)
    parser.add_argument("--roca_dir", help="Directory that ROCA will use for training data generation", required=True)

    args = parser.parse_args(args)

    # Test if rep_dir exists
    if not os.path.isdir(args.rep_dir):
        print(f"The NVIDIA replicator directory {args.rep_dir} does not exist. Existing...")
        exit(-1)

    # Test if roca_dir exists and create if necessary
    if not os.path.isdir(args.roca_dir):
        os.makedirs(args.roca_dir)

    # Convert training data
    print("Converting camera intrinsics...")
    rep_intrinsics_to_scannet(args.rep_dir, args.roca_dir)
    print("Converting images...")
    rep_image_to_scannet(args.rep_dir, args.roca_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
