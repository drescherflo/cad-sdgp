import argparse
import os
import shutil

from OCC.Extend.DataExchange import read_step_file, write_stl_file
from src.stl_to_obj.stl_reader import StlReader
from src.stl_to_obj.stl_obj_convertor import StlVertexConvertor, StlNormalConvertor
from src.stl_to_obj.obj_writer import ObjWriter


def get_step_files_in_dir(directory):
    files = [f for f in os.listdir(directory) if
             (os.path.isfile(os.path.join(directory, f)) and f.endswith((".stp", ".step")))]
    return files


def main():
    # Set vars
    tmp_dir = "tmp/"

    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts STEP files to STL files")
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
        # Code from https://github.com/shivamkadukar/stl_obj_convertor/blob/main/src/stl_to_obj/run_stl_to_obj.py (08.01.2024) with minor modifications
        reader = StlReader()
        vertex_convertor = StlVertexConvertor()
        normal_convertor = StlNormalConvertor()
        writer = ObjWriter()

        with open(tmp_file_path, 'r') as read_file:
            for line in read_file:
                reader.search_vertices(line)
                reader.search_normals(line)

        vertex_convertor.vertex_index(reader.searched_vertices)
        vertex_convertor.unique_vertex_index(reader.searched_vertices)

        normal_convertor.normal_index(reader.searched_normals)
        normal_convertor.unique_normal_index(reader.searched_normals)

        with open(output_file_path, 'w') as write_file:
            writer.write(
                write_file,
                vertex_convertor.v_list,
                normal_convertor.vn_list,
                vertex_convertor.v_index,
                normal_convertor.vn_index
            )

    # Remove tmp dir
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
