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
from omni.isaac.core.utils.semantics import add_update_semantics



def register_random_replicator_cubes():
    def place_replicator_cubes():
        # Register cubes with replicator
        cubes = [rep.create.cube(scale=(0.1, 0.1, 0.1)) for _ in range(2)]
        cubes_grp = rep.create.group(cubes, semantics=[("class", "cube")])
        with cubes_grp:
            rep.modify.pose(position=rep.distribution.uniform((-1, -1, 0.5), (1, 1, 0.5)))
        return cubes_grp.node

    # Register randomization function
    rep.randomizer.register(place_replicator_cubes)


def register_random_isaac_sim_cubes(scene):
    # Create cubes
    cubes = []
    for i in range(5):
        cubes.append(DynamicCuboid(prim_path=f"/cubes/cube{i}", name=f"cube{i}", scale=np.array([0.1, 0.1, 0.1]))) #, position=np.array([0, 0, 0.5 + i]))
        scene.add(cubes[i])

    # Apply semantics
    for cube in cubes:
        add_update_semantics(cube.prim, semantic_label="cube", type_label="class")

    # Define randomization function
    def place_isaac_sim_cubes():
        cubes_as_rep_cubes = rep.get.prims(path_pattern="/cubes/cube*") #, semantics=[("class", "cube")])
        with cubes_as_rep_cubes:
            rep.modify.pose(position=rep.distribution.uniform((-1, -1, 0.5), (1, 1, 0.5)))
        return cubes_as_rep_cubes

    # Register randomization function
    rep.randomizer.register(place_isaac_sim_cubes)


# Run simulation
def main():
    #### Setup simulation
    ### Create new stage
    create_new_stage()

    ### Create world
    world = World()
    scene = world.scene

    ### Add ground plane to scene
    scene.add_default_ground_plane()

    #### Code snippets
    ### Create 100 cones with replicator
    #rep.create.cone(count=100, position=rep.distribution.uniform((-100, -100, -100), (100, 100, 100)), semantics=[("class", "cone")])


    ### Add cubes from replicator with registration function and semantic labels and randomized position
    ## Register replicator graph nodes
    #register_random_replicator_cubes()

    ## Register randomizer functions to run on every frame
    #with rep.trigger.on_frame():
    #    rep.randomizer.place_replicator_cubes()


    ### Add 5 cubes with physics (so called cuboids) from isaac core objects
    register_random_isaac_sim_cubes(scene)

    ## Register randomizer functions to run on every frame
    with rep.trigger.on_frame():
        rep.randomizer.place_isaac_sim_cubes()

    #### Generate replicator graphs without triggering writing
    rep.orchestrator.preview()

    #### Run replicator for 100 frames
    #rep.orchestrator.run(num_frames=100)

    #### Run replicator for one step/frame
    #rep.orchestrator.step()

    #### Run (Physics) Simulation for 100000 steps
    # for i in range(100000):
    #    world.step(render=True, step_sim=True)

    #### Replicator run-loop (only for testing, do NOT use this when generating data)
    while (True):
        rep.orchestrator.step()

    #### Isaac Sim run-loop (only for testing, do NOT use this when generating data)
    #while (True):
    #    simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()

