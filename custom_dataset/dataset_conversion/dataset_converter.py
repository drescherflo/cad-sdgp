import sys
import argparse
import os
import importlib
from typing import Type

from converter_plugins.converter_interface import ConverterInterface


def load_converter_plugins(plugin_dir: str, plugin_package_name) -> list[Type[ConverterInterface]]:
    """
    Loads converter plugins from a specified directory.

    :param plugin_dir: Directory containing the plugin files.
    :param plugin_package_name: Name of the package where plugins are located.
    :return: A list of types derived from the ConverterInterface class.
    """

    converter_plugins = []
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            module = importlib.import_module('.' + module_name, package=plugin_package_name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, ConverterInterface) and attribute is not ConverterInterface:
                    converter_plugins.append(attribute)
    return converter_plugins


def main(args: list[str]) -> None:
    """
    Main function to handle command-line arguments and initiate the conversion process.

    :param args: List of command-line arguments.
    """

    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts the generated training data from NVIDIA Replicator to other formats using the plugins in the converter_plugins directory")
    parser.add_argument("--obj_dir", help="Directory with all CAD Models in OBJ format", required=True)
    parser.add_argument("--replicator_data_dir", action="append", help="Input directory containing the files in the Replicator format. Argument can be added multiple times", required=True)
    parser.add_argument("--output_dir", help="Directory in which the plugins should create the converted data. Each plugin gets its own subdirectory", required=True)
    args = parser.parse_args(args)

    # Test if replicator_data_dirs and obj_dir exist
    replicator_dataset_dirs = args.replicator_data_dir
    for replicator_dataset_dir in replicator_dataset_dirs:
        if not os.path.isdir(replicator_dataset_dir):
            print(f"The NVIDIA replicator directory {args.rep_dir} does not exist. Existing...")
            exit(-1)
    if not os.path.isdir(args.obj_dir):
        print(f"The OBJ model directory {args.rep_dir} does not exist. Existing...")
        exit(-1)

    # Load converter plugins
    plugin_package_name = "converter_plugins"
    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(script_location_dir,
                              plugin_package_name)  # plugin_dir has to be relative to the script. Depending on the execution method, this is not always the case
    converter_plugins = load_converter_plugins(plugin_dir, plugin_package_name)

    # Convert every dataset
    for replicator_dataset_dir in replicator_dataset_dirs:
        # Get base dir for building correct output dir
        base_dir = os.path.basename(os.path.dirname(replicator_dataset_dir))

        # Run conversion process for each converter plugin
        for converter_plugin in converter_plugins:
            print("Converting with plugin", converter_plugin.__name__, "...")

            # Create plugin output dir
            plugin_out_dir = os.path.join(args.output_dir, base_dir, converter_plugin.__name__)
            os.makedirs(plugin_out_dir, exist_ok=True)

            # Check for empty output dir
            if len(os.listdir(plugin_out_dir)) != 0:
                print(f"Output directory {plugin_out_dir} is not empty. Exiting...")
                exit(-1)

            # Convert data with plugin
            converter_plugin.convert(replicator_dataset_dir, args.obj_dir, plugin_out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
