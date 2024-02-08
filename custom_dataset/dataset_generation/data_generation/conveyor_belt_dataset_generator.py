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
from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils.stage import create_new_stage, open_stage
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.prims import RigidPrim, XFormPrim, GeometryPrim

from resumable_writers import load_resumable_writer_plugins
from utils.scene_setup import generate_materials
from utils import quit_on_error


# Enable conveyor belt extension
extensions.enable_extension("omni.isaac.conveyor")


def get_conveyor_node_prims():
    return rep.get.prims(path_pattern="\/World\/ConveyorTrack(.)*\/ConveyorBeltGraph\/ConveyorNode")


def main(conf_path: str, usd_dir: str, out_dir: str) -> None:
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

    # Load json config
    with open(conf_path, "r") as f:
        config = json.load(f)

    # Parse general config
    image_width = config["camera_frame_config"]["frame_width"]
    image_height = config["camera_frame_config"]["frame_height"]
    sub_frames_per_frame = config["sub_frames_per_frame"]
    conveyor_belt_speed = config["conveyor_belt_speed"]
    min_x_pos_for_record_start = config["min_x_pos_for_record_start"]

    # Parse writer config
    # writer_configs = parse_writer_args(args.writer)  TODO uncomment

    # Write train val split
    train_val_split_config = {
        "train_scenes": config["train_frames"],
        "val_scenes": config["val_frames"]
    }

    # Create out_dir if necessary
    os.makedirs(out_dir, exist_ok=True)

    # Check for empty out_dir
    #if len(os.listdir(out_dir)) != 0:
    #    print("Output directory is not empty. Exiting...")
    #    quit_on_error(simulation_app)
    # TODO uncomment and move to separate function

    with open(os.path.join(out_dir, "train_val_scenes.json"), "w") as f:
        json.dump(train_val_split_config, f, indent=4)

    # Set conveyor speed to 1 m/s
    #conveyor_nodes = rep.get.prims(path_pattern="\/World\/ConveyorTrack(.)*\/ConveyorBeltGraph\/ConveyorNode")
    #with conveyor_nodes:
    #    rep.modify.attribute("velocity", 1.0)

    # Scene generation loop
    frame_number = 0
    num_frames = len(config["scenes"]) * config["num_frames_per_scene"]
    for scene_nr, scene_config in enumerate(config["scenes"]):
        print("Setting up scene", scene_nr + 1, "of", len(config["scenes"]))

        # Reset simulation
        create_new_stage()  # Create new stage and delete previous contents
        
        # Load stage
        if not open_stage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaac_worlds/conveyor.usd")):
            print("Could not open world. Exiting...", file=sys.stderr)
            quit_on_error(simulation_app)

        #FIXME: Isaac Sim adds a default light, because there is none in the usd stage

        # Get world
        world = World()

        # Create scene objects
        num_objects_per_scene = len(scene_config["objects"])
        object_prims = []
        object_rigid_prims = []
        for i, object in enumerate(scene_config["objects"]):
            usd_path = os.path.abspath(os.path.join(usd_dir, object["usd_model"]))
            if not os.path.isfile(usd_path):
                print(f"USD file at path '{usd_path}' could not be found. Exiting...", file=sys.stderr)
                quit_on_error(simulation_app)

            prim_name = f"object_{i:0{len(str(num_objects_per_scene))}}"
            prim_path = f"/objects/{prim_name}"
            object_prims.append(prims.create_prim(prim_path=prim_path,
                                                  usd_path=usd_path,
                                                  semantic_label=object["semantic_class_label"],
                                                  position=object["object_init_pose"]["position"],
                                                  orientation=euler_angles_to_quat(np.array(object["object_init_pose"]["rotation"]), degrees=True, extrinsic=False)))
            rigid_prim = RigidPrim(prim_path=prim_path, name=prim_name + "_rigid")  # RigidPrim for Physics
            geometry_prim = GeometryPrim(prim_path=prim_path, name=prim_name + "_geometry",
                                         collision=True)  # GeometryPrim for Collisions

            world.scene.add(rigid_prim)  # Register in world's scene to enable physics simulation
            world.scene.add(geometry_prim)  # Register in world's scene to enable collision calculations
            object_rigid_prims.append(rigid_prim)

        # Reset the world to handle the physics of the newly created rigid prims
        world.reset()

        # Set conveyor belt speed to 0 (until all objects stopped falling to prevent lower objects already moving on the conveyor belt)
        conveyor_nodes = get_conveyor_node_prims()
        with conveyor_nodes:
           rep.modify.attribute("velocity", 0.0)

        # Run simulation until all objects stopped falling
        objects_stopped_falling = False
        while not objects_stopped_falling:
            # Run simulation for one step and check linear velocity
            world.step(render=True, step_sim=True)
            objects_stopped_falling = all([np.linalg.norm(object_prim.get_linear_velocity()) < 0.001 for object_prim in object_rigid_prims])

        # Set conveyor belt speed to specified value
        conveyor_nodes = get_conveyor_node_prims()
        with conveyor_nodes:
           rep.modify.attribute("velocity", conveyor_belt_speed)

        # Wait for min one object to pass min_x_pos_for_record_start before start of recording
        min_x_pos_for_record_start_passed = False
        while not min_x_pos_for_record_start_passed:
            # Run simulation for one step and check x coordinates
            world.step(render=True, step_sim=True)
            min_x_pos_for_record_start_passed = any(
                [np.linalg.norm(object_prim.get_world_pose()[0][0]) > min_x_pos_for_record_start for object_prim in object_rigid_prims])

        while simulation_app.is_running():
            simulation_app.update()
    

if __name__ == '__main__':
    conf_path = "custom_dataset/dataset_generation/config_generation/config.json"
    usd_dir = "CAD Models/OBJ_converted"
    out_dir = "temp_replicator_out"

    main(conf_path, usd_dir, out_dir)
    simulation_app.close()
