import os

print(os.environ["CARB_APP_PATH"])
print(os.environ["ISAAC_PATH"])
print(os.environ["EXP_PATH"])
print(os.getcwd())

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
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.scenes import Scene
from omni.isaac.core.prims import RigidPrim, XFormPrim, GeometryPrim


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
        cubes.append(DynamicCuboid(prim_path=f"/cubes/cube{i}", name=f"cube{i}",
                                   scale=np.array([0.1, 0.1, 0.1])))  # , position=np.array([0, 0, 0.5 + i]))
        scene.add(cubes[i])

    # Apply semantics
    for cube in cubes:
        add_update_semantics(cube.prim, semantic_label="cube", type_label="class")

    # Define randomization function
    def place_isaac_sim_cubes():
        cubes_as_rep_cubes = rep.get.prims(path_pattern="/cubes/cube*")  # , semantics=[("class", "cube")])
        with cubes_as_rep_cubes:
            rep.modify.pose(position=rep.distribution.uniform((-1, -1, 0.5), (1, 1, 0.5)))
        return cubes_as_rep_cubes

    # Register randomization function
    rep.randomizer.register(place_isaac_sim_cubes)


# Run simulation
def register_random_replicator_usd_cad_model():
    def place_replicator_models():
        # Load cad models with replicator
        models = rep.create.from_usd(usd="CAD Models/STL_converted/MA Simple Object_stl.usd", semantics=[("class", "simple_object")], count=2)
        with models:
            rep.modify.pose(position=rep.distribution.uniform((-1, -1, 0.5), (1, 1, 0.5)))
            rep.modify._scale(scale=(0.001, 0.001, 0.001))  # Set correct scale (unit of step file is mm, but isaac sim uses m)
        return models

    # Register randomization function
    rep.randomizer.register(place_replicator_models)


def register_random_isaac_sim_usd_cad_model(world: World) -> None:
    # Create prims
    xform_prims = []
    for i in range(2):
        prim_name = f"simple_object{i}"
        prim_path = "/simple_objects/" + prim_name
        xform_prim = prims.create_prim(prim_path=prim_path, usd_path="CAD Models/STL_converted/MA Simple Object_stl.usd", scale=[0.001, 0.001, 0.001], position=[0, 0, i])  # XFormPrim for rendering
        xform_prims.append(xform_prim)
        rigid_prim = RigidPrim(prim_path=prim_path, name=prim_name + "_rigid")  # RigidPrim for Physics
        geometry_prim = GeometryPrim(prim_path=prim_path, name=prim_name + "_geometry", collision=True)  # GeometryPrim for Collisions

        world.scene.add(rigid_prim)  # Register in world's scene to run physics simulation
        world.scene.add(geometry_prim)  # Register in world's scene to run physics simulation
    # Reset the world to handle the physics of the newly created rigid prims
    world.reset()

    # Apply semantics
    for xform_prim in xform_prims:
        add_update_semantics(xform_prim, semantic_label="simple_object", type_label="class")

    # Define randomization function
    def place_isaac_sim_cad_models():
        prims_as_rep_prims = rep.get.xform(path_pattern="/simple_objects/simple_object*")  # !!! get.xform needs to be used, because else replicator will also return subdirs, materials, etc of the matching prims
        with prims_as_rep_prims:
            rep.modify.pose(position=rep.distribution.uniform((-1, -1, 0), (1, 1, 0)))
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
        return prims_as_rep_prims

    # Register randomization function
    rep.randomizer.register(place_isaac_sim_cad_models)


def main():
    #### Setup simulation
    ### Create new stage
    create_new_stage()

    ### Create world
    world = World()
    scene = world.scene

    ### Add ground plane to scene
    scene.add_default_ground_plane()

    ### Rerender simulation
    simulation_app.update()

    #### Code snippets

    ####################################################################
    ### Create 100 cones with replicator
    # rep.create.cone(count=100, position=rep.distribution.uniform((-100, -100, -100), (100, 100, 100)), semantics=[("class", "cone")])

    ####################################################################
    ### Add cubes from replicator with registration function and semantic labels and randomized position
    ## Register replicator graph nodes
    # register_random_replicator_cubes()

    ## Register randomizer functions to run on every frame
    # with rep.trigger.on_frame():
    #    rep.randomizer.place_replicator_cubes()

    ####################################################################
    ### Add 5 cubes with physics (so called cuboids) from isaac core objects
    # register_random_isaac_sim_cubes(scene)

    ## Register randomizer functions to run on every frame
    # with rep.trigger.on_frame():
    #     rep.randomizer.place_isaac_sim_cubes()

    ####################################################################
    ### Add 2 CAD Models with replicator with semantic labels and randomized position
    # register_random_replicator_usd_cad_model()

    ## Register randomizer functions to run on every frame
    # with rep.trigger.on_frame():
    #     rep.randomizer.place_replicator_models()

    ####################################################################
    ### Add 2 CAD Models with isaac sim with pyhsics, collisions, semantic labels, randomized position and randomized color
    register_random_isaac_sim_usd_cad_model(world)

    ## Register randomizer functions to run on every frame
    with rep.trigger.on_frame():
        rep.randomizer.place_isaac_sim_cad_models()

    ####################################################################
    #### Generate replicator graphs without triggering writing
    # rep.orchestrator.preview()

    ####################################################################
    #### Run replicator for 100 frames
    # rep.orchestrator.run(num_frames=100)

    ####################################################################
    #### Run replicator for one step/frame
    # rep.orchestrator.step()

    ####################################################################
    #### Run (Physics) Simulation for 1000 steps
    for i in range(1000):
       world.step(render=True, step_sim=True)

    ####################################################################
    #### Replicator run-loop (only for testing, do NOT use this when generating data)
    while True:
        rep.orchestrator.step()

    ####################################################################
    #### Isaac Sim run-loop (only for testing, do NOT use this when generating data)
    while True:
       simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
