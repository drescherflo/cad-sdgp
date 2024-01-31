# Regular imports
import argparse
import os
import json
import sys
import numpy as np
from utils.config import parse_writer_args

# Launch Isaac Sim
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}# args.headless}
simulation_app = SimulationApp(launch_config=CONFIG)

# Omniverse imports and omniverse related imports need to be done after the simulation has been started
import omni.replicator.core as rep
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils.stage import create_new_stage, open_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from resumable_writers import load_resumable_writer_plugins
from utils.scene_setup import generate_materials
from utils import quit_on_error


# Enable conveyor belt extension
extensions.enable_extension("omni.isaac.conveyor")


def main() -> None:
    """
    Main function to handle the dataset generation process.

    :param conf_path: Path to the configuration file.
    :param usd_dir: Directory containing USD files.
    :param out_dir: Output directory where the dataset will be generated.
    :type conf_path: str
    :type usd_dir: str
    :type out_dir: str
    """

    # Load resumable writers
    load_resumable_writer_plugins()

    # Load stage
    if not open_stage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaac_worlds/conveyor.usd")):
        print("Could not open world. Exiting...")
        quit_on_error(simulation_app)

    while simulation_app.is_running():
        simulation_app.update()
    

if __name__ == '__main__':
    main()
    simulation_app.close()
