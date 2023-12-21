import os
print(os.environ["CARB_APP_PATH"])
print(os.environ["ISAAC_PATH"])
print(os.environ["EXP_PATH"])

# Launch Isaac Sim
import numpy as np
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(launch_config=CONFIG)


# Register signal handler for Ctrl+C for exiting
import signal
def signal_handler(signal, frame):
    print("Received Ctrl+C, exiting...")
    simulation_app.close()
    exit(0)
signal.signal(signal.SIGINT, signal_handler)


# Main code starts here
# All omniverse imports need to be done after the simulation has been created (https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_offline_generation.html#running-as-a-simulationapp 20.12.2023)
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_current_stage, create_new_stage
from omni.isaac.core.objects import DynamicCuboid



# def register_random_cubes():
#     def place_cubes():
#         # Register cubes with replicator
#         cubes = [rep.create.cube(scale=(0.1, 0.1, 0.1)) for _ in range(2)]
#         cubes_grp = rep.create.group(cubes, semantics=[("class", "cone")])
#         with cubes_grp:
#             rep.modify.pose(position=rep.distribution.uniform((-1, -1, 0.5), (1, 1, 0.5)))
#         return cubes_grp.node
#
#     rep.randomizer.register(place_cubes)
#

# Run simulation
def main():
    ## Setup simulation
    # Create new stage
    create_new_stage()

    # Create world
    world = World()
    scene = world.scene

    simulation_app.update()
#    scene.add_default_ground_plane()

    rep.create.cone(count=100, position=rep.distribution.uniform((-100, -100, -100), (100, 100, 100)), semantics=[("class", "cube")])

    # Add cubes with physics
    #cubes = []
    #for i in range(5):
    #    cubes.append(DynamicCuboid(prim_path=f"/cubes/cube{i}", name=f"cube{i}", position=np.array([0, 0, 0.5 + i]), scale=np.array([0.1, 0.1, 0.1])))
    #    scene.add(cubes[i])

    # Run Simulation for 100000 steps
    #for i in range(100000):
    #    world.step(render=True, step_sim=True)



    # Register replicator graph nodes
    #register_random_cubes()

    # Run replicator on every frame
    #with rep.trigger.on_frame():
    #    rep.randomizer.place_cubes()

    while (True):
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()

