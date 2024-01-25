import argparse
import os
import shutil

from OCC.Extend.DataExchange import read_step_file, write_stl_file
from stl import mesh
import numpy as np


def get_step_files_in_dir(directory):
    """
    Retrieves a list of STEP file names from a specified directory.

    :param directory: The directory to search for STEP files.
    :type directory: str
    :return: A list of STEP file names found in the directory.
    :rtype: list
    """

    files = [f for f in os.listdir(directory) if
             (os.path.isfile(os.path.join(directory, f)) and f.endswith((".stp", ".step")))]
    return files


def convert_exponential_to_decimal(input_strings):
    """
    Converts strings containing exponential numbers to decimal format.

    :param input_strings: A list of strings containing exponential numbers.
    :type input_strings: list
    :return: A list of strings with numbers converted to decimal format.
    :rtype: list
    """

    converted_strings = []
    for string in input_strings:
        numbers = string.split()  # Split at space
        converted_numbers = ['{:.8f}'.format(float(num)) for num in numbers]  # Convert string to float
        converted_string = '  '.join(converted_numbers)  # Convert back to string
        converted_strings.append(converted_string)
    return converted_strings


def main():
    """
    Main function to initiate the conversion process from STEP to OBJ format.
    Handles command-line arguments and orchestrates the conversion.
    """

    # Set vars
    tmp_dir = "tmp/"

    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts STEP files to STL files")
    parser.add_argument("--input_dir", help="Input directory containing .stp files or .step files")
    parser.add_argument("--output_dir", help="Output directory for .obj files")
    parser.add_argument("--scale_factor", type=float, default=0.001,
                        help="Scales the .obj file by this factor.\n"
                             "This is required when the object was designed with some other unit than meters.\n"
                             "The default value of 0.001 is used when the object was designed in millimeters.")

    # Parse args
    args = parser.parse_args()
    if args.input_dir is None:
        print("No input directory provided via --input_dir. Exiting...")
        exit(-1)
    if args.output_dir is None:
        print("No output directory provided via --output_dir. Exiting...")
        exit(-1)

    # Test if input dir, tmp dir and output dir exist
    if not os.path.isdir(args.input_dir):
        print("Input directory does not exist. Exiting...")
        exit(-1)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    # Convert stp files to stl files
    files = get_step_files_in_dir(args.input_dir)
    for file in files:
        print("Processing file: " + file)
        # Prepare paths
        input_file_path = os.path.join(args.input_dir, file)
        tmp_file_path = os.path.join(tmp_dir, file).replace(".stp", ".stl").replace(".step", ".stl")
        output_file_path = os.path.join(args.output_dir, file).replace(".stp", ".obj").replace(".step", ".obj")

        # Convert stp to (high quality) stl file in tmp
        shape = read_step_file(input_file_path)
        write_stl_file(shape, tmp_file_path, mode="ascii", linear_deflection=0.001, angular_deflection=0.001)

        # Convert tmp stl to obj
        stl_mesh = mesh.Mesh.from_file(tmp_file_path)
        with open(output_file_path, 'w') as f:
            # Write vertices
            for v in np.vstack(stl_mesh.vectors):
                # scale by provided factor (default 0.001)
                v = v * args.scale_factor
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')

            # Write faces
            for i in range(len(stl_mesh.vectors)):
                f.write(f'f {3 * i + 1} {3 * i + 2} {3 * i + 3}\n')

    # Remove tmp dir
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
    print("Done!")
