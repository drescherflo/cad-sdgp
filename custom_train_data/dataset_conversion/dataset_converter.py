import sys
import argparse
import os
import importlib
from typing import Type

from converter_plugins.converter_interface import ConverterInterface


def load_converter_plugins(plugin_dir: str, plugin_package_name) -> list[Type[ConverterInterface]]:
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
    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts the generated training data from NVIDIA Replicator to other formats using the plugins in the converter_plugins directory")
    parser.add_argument("--obj_dir", help="Directory with all CAD Models in OBJ format", required=True)
    parser.add_argument("--replicator_data_dir", help="Input directory containing the files in the Replicator format", required=True)
    parser.add_argument("--output_dir", help="Directory in which the plugins should create the converted data. Each plugin gets its own subdirectory", required=True)
    args = parser.parse_args(args)

    # Test if replicator_data_dir and obj_dir exist
    if not os.path.isdir(args.replicator_data_dir):
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

    # Run conversion process for each converter plugin
    for converter_plugin in converter_plugins:
        print("Converting with plugin", converter_plugin.__name__, "...")

        # Create plugin output dir
        plugin_out_dir = os.path.join(args.output_dir, str(converter_plugin.__name__))
        os.makedirs(plugin_out_dir, exist_ok=True)

        # Convert data with plugin
        converter_plugin.convert(args.replicator_data_dir, args.obj_dir, plugin_out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
