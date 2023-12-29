import argparse
import os

from OCC.Extend.DataExchange import read_step_file, write_stl_file


def get_step_files_in_dir(directory):
    files = [f for f in os.listdir(directory) if
             (os.path.isfile(os.path.join(directory, f)) and f.endswith((".stp", ".step")))]
    return files


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Konvertiert STEP-Dateien zu STL-Dateien")
    parser.add_argument("--input_dir", help="Pfad zum Eingabe-Verzeichnis mit .stp-Dateien")
    parser.add_argument("--output_dir", help="Pfad zum Ausgabe-Verzeichnis für .stl-Dateien")

    # Parse args
    args = parser.parse_args()
    if args.input_dir is None:
        print("Kein Eingabe-Verzeichnis via --input_dir angegeben. Programm wird beendet")
        exit(-1)
    if args.output_dir is None:
        print("Kein Ausgabe-Verzeichnis via --output_dir angegeben. Programm wird beendet")
        exit(-1)

    # Test if input dir and output dir exist
    if not os.path.isdir(args.input_dir):
        print("Eingabe-Verzeichnis existiert nicht. Programm wird beendet")
        exit(-1)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Convert stp files to stl files
    files = get_step_files_in_dir(args.input_dir)
    for file in files:
        print("Processing file: " + file)
        # Prepare paths
        input_file_path = os.path.join(args.input_dir, file)
        output_file_path = os.path.join(args.output_dir, file).replace(".stp", ".stl").replace(".step", ".stl")

        # Read stp file
        shape = read_step_file(input_file_path)

        # Convert and save as (high quality) stl file
        write_stl_file(shape, output_file_path, mode="binary", linear_deflection=0.001, angular_deflection=0.001)


if __name__ == "__main__":
    main()
