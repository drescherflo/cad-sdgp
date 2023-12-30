import argparse
import os

from OCC.Extend.DataExchange import read_step_file, write_obj_file


def get_step_files_in_dir(directory):
    files = [f for f in os.listdir(directory) if
             (os.path.isfile(os.path.join(directory, f)) and f.endswith((".stp", ".step")))]
    return files


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts STEP files to OBJ files")
    parser.add_argument("--input_dir", help="Input directory containing .stp files or .step files")
    parser.add_argument("--output_dir", help="Output directory for .obj files")

    # Parse args
    args = parser.parse_args()
    if args.input_dir is None:
        print("No input directory provided via --input_dir. Exiting...")
        exit(-1)
    if args.output_dir is None:
        print("No output directory provided via --output_dir. Exiting...")
        exit(-1)

    # Test if input dir and output dir exist
    if not os.path.isdir(args.input_dir):
        print("Input directory does not exist. Exiting...")
        exit(-1)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Convert stp files to obj files
    files = get_step_files_in_dir(args.input_dir)
    for file in files:
        print("Processing file: " + file)
        # Prepare paths
        input_file_path = os.path.join(args.input_dir, file)
        output_file_path = os.path.join(args.output_dir, file).replace(".stp", ".obj").replace(".step", ".obj")

        # Read stp file
        shape = read_step_file(input_file_path)

        # Convert and save as obj file
        write_obj_file(shape, output_file_path)


if __name__ == "__main__":
    main()
