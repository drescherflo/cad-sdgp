import argparse
import os
import re
import shutil

from OCC.Extend.DataExchange import read_step_file, write_stl_file
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Quantity import Quantity_Color
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool
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


def linear_to_srgb(channel):
    """
    Converts a single colour channel from linear RGB to sRGB.

    OpenCASCADE reports colours in linear space, while the Kd entry of an MTL file is
    interpreted as sRGB by the consuming renderers (Isaac Sim included). Without this
    conversion the material ends up noticeably too dark.

    :param channel: Colour channel in linear space.
    :type channel: float
    :return: Colour channel in sRGB space.
    :rtype: float
    """

    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * (channel ** (1 / 2.4)) - 0.055


def sanitize_material_name(name):
    """
    Turns an arbitrary STEP label into a name that is safe to use inside an MTL file.

    :param name: The raw name taken from the STEP file.
    :type name: str
    :return: A name without whitespace or special characters.
    :rtype: str
    """

    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", name.strip())
    return sanitized if sanitized else "step_material"


def extract_step_color(step_file_path):
    """
    Extracts the surface colour of a STEP file via the XDE/XCAF layer.

    The plain reader used for the geometry discards all presentation data, so the file is
    read a second time through STEPCAFControl_Reader. Only a single colour per file is
    resolved, because the STL intermediate merges all sub shapes into one mesh and the
    per face assignment would be lost anyway.

    :param step_file_path: Path to the STEP file.
    :type step_file_path: str
    :return: Tuple of (name, (r, g, b)) with the colour in sRGB, or None if the file carries no colour.
    :rtype: tuple | None
    """

    doc = TDocStd_Document("step-colour-import")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    if reader.ReadFile(step_file_path) != IFSelect_RetDone:
        return None
    reader.Transfer(doc)

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    if roots.Length() == 0:
        return None
    name = roots.Value(1).GetLabelName()

    # The label based overloads of GetColor are not callable from python, so the colour is
    # resolved through the shape a label refers to. Sub shapes are checked as well because
    # assemblies usually carry the colour on their components rather than on the root.
    for root_idx in range(1, roots.Length() + 1):
        label = roots.Value(root_idx)
        candidates = [shape_tool.GetShape(label)]

        sub_shapes = TDF_LabelSequence()
        shape_tool.GetSubShapes(label, sub_shapes)
        candidates += [shape_tool.GetShape(sub_shapes.Value(i)) for i in range(1, sub_shapes.Length() + 1)]

        for candidate in candidates:
            for channel in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
                color = Quantity_Color()
                if color_tool.GetColor(candidate, channel, color):
                    return name, tuple(linear_to_srgb(c) for c in (color.Red(), color.Green(), color.Blue()))

    return None


def write_mtl_file(mtl_file_path, material_name, rgb):
    """
    Writes a Wavefront MTL file containing a single diffuse material.

    :param mtl_file_path: Destination path of the .mtl file.
    :type mtl_file_path: str
    :param material_name: Name the material is registered under.
    :type material_name: str
    :param rgb: Diffuse colour as (r, g, b) in sRGB.
    :type rgb: tuple
    """

    with open(mtl_file_path, 'w') as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Kd {rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
        f.write("Ka 0.000000 0.000000 0.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")


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

        # Convert stp to stl file in tmp
        shape = read_step_file(input_file_path)
        write_stl_file(shape, tmp_file_path, mode="ascii")

        # The stl intermediate cannot carry the material, so read it from the step file directly
        material = extract_step_color(input_file_path)
        if material is None:
            print("  No colour found in step file, writing obj without material")

        # Convert tmp stl to obj
        stl_mesh = mesh.Mesh.from_file(tmp_file_path)
        with open(output_file_path, 'w') as f:
            # Write material reference
            if material is not None:
                material_name = sanitize_material_name(material[0])
                mtl_file_path = os.path.splitext(output_file_path)[0] + ".mtl"
                write_mtl_file(mtl_file_path, material_name, material[1])
                print(f"  Extracted material '{material_name}' -> {os.path.basename(mtl_file_path)}")
                f.write(f"mtllib {os.path.basename(mtl_file_path)}\n")
                f.write(f"usemtl {material_name}\n")

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
