import argparse
import os
import asyncio
import logging

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp

def get_obj_files_in_dir(directory):
    """
    Retrieves a list of OBJ file names from a specified directory.

    :param directory: The directory to search for OBJ files.
    :type directory: str
    :return: A list of OBJ file names found in the directory.
    :rtype: list
    """

    files = [f for f in os.listdir(directory) if
             (os.path.isfile(os.path.join(directory, f)) and f.lower().endswith((".obj")))]
    return files

async def convert(input_file_path, output_file_path):    
    import omni.kit.asset_converter
    # https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html (05.06.2025)
    # converter_context = omni.kit.asset_converter.AssetConverterContext()  # can be used for config, we use default config
    converter_instance = omni.kit.asset_converter.get_instance()
    converter_task = converter_instance.create_converter_task(input_file_path, output_file_path)
    success = await converter_task.wait_until_finished()
    if not success:
        logger = logging.getLogger(__name__)
        detailed_status_error_string = converter_task.get_error_message()
        logger.error(detailed_status_error_string)
    


if __name__ == "__main__":
    """
    Main function to initiate the conversion process from OBJ to USD format.
    Handles command-line arguments and orchestrates the conversion.
    """

    # Get logger
    logger = logging.getLogger(__name__)
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts STEP files to STL files")
    parser.add_argument("--input_dir", help="Input directory containing .obj files")
    parser.add_argument("--output_dir", help="Output directory for .usd files")

    # Parse args
    args = parser.parse_args()
    if args.input_dir is None:
        logger.error("No input directory provided via --input_dir. Exiting...")
        exit(-1)
    if args.output_dir is None:
        logger.error("No output directory provided via --output_dir. Exiting...")
        exit(-1)
    
    # Test if input dir and output dir exist
    if not os.path.isdir(args.input_dir):
        logger.error("Input directory does not exist. Exiting...")
        exit(-1)
    if not os.path.isdir(args.output_dir):
        logger.debug("Creating ouput dir f{args.output_dir}, because it does not exist yet")
        os.makedirs(args.output_dir)

    # Convert obj to usd files
    kit = SimulationApp()
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")

    files = get_obj_files_in_dir(args.input_dir)
    for file in files:
        # Build paths
        input_file_path = os.path.join(args.input_dir, file)
        output_file_path = os.path.join(args.output_dir, file).replace(".obj", ".usd").replace(".OBJ", ".usd")

        # Convert obj to usd
        logger.info("Converting f{file}...")
        asyncio.get_event_loop().run_until_complete(convert(input_file_path, output_file_path))
    
    logger.info("Conversion to .usd finished")
    kit.close()
