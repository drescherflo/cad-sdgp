import sys
import argparse
import os

from converter_plugins import load_converter_plugins


def __parse_converter_init_args(converter_init_args: list[str]) -> dict:
    """
    Parses initialization arguments for a converter.

    Converts argument strings into a dictionary, interpreting 'true'/'false' as booleans and
    numeric values as int or float, so that a converter can take more than just flags.
    E.g. ['split=train_pbr', 'amodal_masks=false'] becomes {'split': 'train_pbr', 'amodal_masks': False}.

    :param converter_init_args: A list of string arguments.
    :type converter_init_args: list[str]
    :return: A dictionary mapping argument names to their parsed values.
    :rtype: dict
    """

    args_dict = {}
    for arg in converter_init_args:
        if '=' not in arg:
            print(f"Ignoring converter argument {arg}, it is not a key=value pair")
            continue
        key, value = arg.split('=', 1)
        if value.lower() in ('true', 'false'):
            args_dict[key] = value.lower() == 'true'
            continue
        try:
            args_dict[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            args_dict[key] = float(value)
            continue
        except ValueError:
            pass
        args_dict[key] = value
    return args_dict


def parse_converter_args(converter_args: list[list[str]]) -> list[dict]:
    """
    Parses arguments for multiple converters.

    :param converter_args: A list of lists, each containing the name of a converter followed by its
           arguments as key=value pairs.
    :type converter_args: list[list[str]]
    :return: A list of configurations, one per converter.
    :rtype: list[dict]
    """

    converters = []
    for converter_arg in converter_args:
        if not converter_arg:
            print("Ignoring empty --converter argument")
            continue
        converters.append({"name": converter_arg[0], "args": __parse_converter_init_args(converter_arg[1:])})

    return converters


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
    parser.add_argument("--converter", nargs="*", action="append", help="Configures a converter plugin from the 'converter_plugins' package as 'Name key=value ...'. Argument can be added multiple times", required=True)
    args = parser.parse_args(args)

    # Parse converter config
    converter_configs = parse_converter_args(args.converter)

    # Test if replicator_data_dirs and obj_dir exist
    replicator_dataset_dirs = args.replicator_data_dir
    for replicator_dataset_dir in replicator_dataset_dirs:
        if not os.path.isdir(replicator_dataset_dir):
            print(f"The NVIDIA replicator directory {replicator_dataset_dir} does not exist. Existing...")
            exit(-1)
    if not os.path.isdir(args.obj_dir):
        print(f"The OBJ model directory {args.obj_dir} does not exist. Existing...")
        exit(-1)

    # Load converter plugins
    converter_plugins = load_converter_plugins()

    # Only use specified converters
    converter_args_by_name = {converter_config["name"]: converter_config["args"] for converter_config in converter_configs}
    converter_plugins = [converter_plugin for converter_plugin in converter_plugins if converter_plugin.__name__ in converter_args_by_name]
    if len(converter_plugins) == 0:
        print(f"No converters matching the class names in the converter_plugins module. Exiting...")
        exit(-1)

    # Convert every dataset
    for replicator_dataset_dir in replicator_dataset_dirs:
        # Get base dir for building correct output dir
        if replicator_dataset_dir.endswith("/"):
            base_dir = os.path.basename(os.path.dirname(replicator_dataset_dir))
        else:
            base_dir = os.path.basename(replicator_dataset_dir)

        # Run conversion process for each converter plugin
        for converter_plugin in converter_plugins:
            print("Converting with plugin", converter_plugin.__name__, "...")

            # Create plugin output dir
            plugin_out_dir = os.path.join(args.output_dir, base_dir, converter_plugin.__name__)
            os.makedirs(plugin_out_dir, exist_ok=True)

            # Check for empty output dir
            # if len(os.listdir(plugin_out_dir)) != 0:
            #     print(f"Output directory {plugin_out_dir} is not empty. Exiting...")
            #     exit(-1)

            # Convert data with plugin
            converter_plugin.convert(replicator_dataset_dir, args.obj_dir, plugin_out_dir, **converter_args_by_name[converter_plugin.__name__])


if __name__ == '__main__':
    main(sys.argv[1:])
