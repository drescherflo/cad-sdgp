import os
print(os.environ["ISAAC_PATH"])

# Launch Isaac Sim
from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(launch_config=CONFIG)

# Main code starts here
import signal

# Register signal handler for Ctrl+C for exiting
def signal_handler(signal, frame):
    print("Received Ctrl+C, exiting...")
    simulation_app.close()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)




# All omniverse imports need to be done after the simulation has been created (https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_offline_generation.html#running-as-a-simulationapp 20.12.2023)
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_current_stage, create_new_stage


# Helper function to find the assets server
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path

# Run simulation
def main():
    ## Setup simulation
    # Create new stage
    stage = create_new_stage()

    # Create world
    world = World()
    world.scene.add_default_ground_plane()

    while(True):
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()

