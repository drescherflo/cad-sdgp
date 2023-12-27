import argparse
import os

from OCC.STEPControl import STEPControl_Reader
from OCC.StlAPI import StlAPI_Writer


def get_files_in_dir(directory):
    """
    Listet alle Dateien im angegebenen Verzeichnis auf.
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
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
    files = get_files_in_dir(args.input_dir)
    for file in files:
        # Prepare paths
        input_file_path = os.path.join(args.input_dir, file)
        output_file_path = os.path.join(args.output_dir, file).replace(".stp", ".stl")

        # Read stp file
        step_reader = STEPControl_Reader()
        step_reader.ReadFile(input_file_path)
        step_reader.TransferRoot()
        shape = step_reader.Shape()

        # Convert and save as stl file
        stl_writer = StlAPI_Writer()
        stl_writer.SetASCIIMode(True)
        stl_writer.Write(shape, output_file_path)


if __name__ == "__main__":
    main()
