# Regular imports
import argparse
import os
import json
import sys
import numpy as np
import collections
from utils.config import parse_writer_args

# Launch Isaac Sim
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}# args.headless}
simulation_app = SimulationApp(launch_config=CONFIG)

# Omniverse imports and omniverse related imports need to be done after the simulation has been started
import omni.graph.core as og
import omni.replicator.core as rep
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.prims import RigidPrim, XFormPrim, GeometryPrim
from omni.isaac.core.materials import OmniPBR

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
    num_frames_per_scene = config["num_frames_per_scene"]

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

    # Scene generation loop
    frame_number = 0
    num_frames = len(config["scenes"]) * num_frames_per_scene
    for scene_nr, scene_config in enumerate(config["scenes"]):
        print("Setting up scene", scene_nr + 1, "of", len(config["scenes"]))

        # Parse per frame config
        per_frame_config = scene_config["per_frame_config"]
        camera_positions = [pose["position"] for pose in per_frame_config["camera_poses"]]
        camera_orientations = [pose["rotation"] for pose in per_frame_config["camera_poses"]]
        sphere_light_configs = per_frame_config["sphere_light_configs"]
        num_sphere_lights = len(sphere_light_configs[0])
        distant_light_orientations = [conf["rotation"] for conf in per_frame_config["distant_light_configs"]]
        distant_light_colors = [conf["color"] for conf in per_frame_config["distant_light_configs"]]
        distant_light_intensities = [conf["intensity"] for conf in per_frame_config["distant_light_configs"]]
        ground_plane_colors = per_frame_config["ground_plane_colors"]
        object_material_assignments = per_frame_config["object_material_assignments"]
        conveyor_belt_colors = per_frame_config["conveyor_belt_colors"]
        conveyor_frame_colors = per_frame_config["conveyor_frame_colors"]

        # Reset simulation by creating new stage
        stage_utils.create_new_stage()

        # Add conveyor environment (don't use stage_utils.open_stage(); reopening leads to simulation crash)
        stage_utils.add_reference_to_stage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaac_worlds/conveyor.usd"), "/World")

        # Get world
        world = World()

        # Generate object materials
        materials = generate_materials(config["materials"])

        # Add camera
        camera = rep.create.camera()
        render_product = rep.create.render_product(camera=camera, resolution=(image_width, image_height))

        # Add material for custom colors of ground plane
        plane_material = rep.create.material_omnipbr(roughness=1)
        plane = rep.get.prims(path_match="/World/GroundPlane")
        with plane:
            rep.modify.material(plane_material)

        # Add material for custom colors of conveyor belt belts
        conveyor_belt_material_path = "/materials/conveyor_belt"
        conveyor_belt_material = OmniPBR(conveyor_belt_material_path)
        conveyor_belt_material.set_reflection_roughness(1.0)
        conveyor_belt_prim_paths = prim_utils.find_matching_prim_paths(
            "/World/ConveyorTrack(.)*/Belt/SM_ConveyorBelt_A09_Belt_02")
        for conveyor_belt_prim_path in conveyor_belt_prim_paths:
            xform_conveyor_belt_prim = XFormPrim(conveyor_belt_prim_path)
            xform_conveyor_belt_prim.apply_visual_material(conveyor_belt_material)
        rep_conveyor_belt_material = rep.get.material(conveyor_belt_material_path)

        # Add material for custom color of conveyor belt frame
        conveyor_frame_material_path = "/materials/conveyor_frame"
        conveyor_frame_material = OmniPBR(conveyor_frame_material_path)
        conveyor_frame_material.set_reflection_roughness(1.0)
        conveyor_frame_prim_paths = prim_utils.find_matching_prim_paths(
            "/World/ConveyorTrack(.)*/SM_ConveyorBelt_A09_02")
        for conveyor_frame_prim_path in conveyor_frame_prim_paths:
            xform_conveyor_frame_prim = XFormPrim(conveyor_frame_prim_path)
            xform_conveyor_frame_prim.apply_visual_material(conveyor_frame_material)
        rep_conveyor_frame_material = rep.get.material(conveyor_frame_material_path)

        # Add distant light
        distant_light = rep.create.light(light_type="distant")

        # Add sphere lights
        sphere_lights = [rep.create.light(light_type="sphere") for _ in range(num_sphere_lights)]

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

            # Enable physics and collision
            # Don't use rep.physics.rigid_body() and rep.physics.collider() or else either the simulation or PhysX will crash!
            rigid_prim = RigidPrim(prim_path=prim_path, name=prim_name + "_rigid")  # RigidPrim for Physics
            geometry_prim = GeometryPrim(prim_path=prim_path, name=prim_name + "_geometry",
                                         collision=True)  # GeometryPrim for Collisions
            geometry_prim.set_collision_approximation("convexDecomposition")  # Use Convex Decomposition for a more fine granular collision calculation at cost of simulation performance

            world.scene.add(rigid_prim)  # Register in world's scene to enable physics simulation
            world.scene.add(geometry_prim)  # Register in world's scene to enable collision calculations
            object_rigid_prims.append(rigid_prim)

        # Reset the world to handle the physics of the newly created prims
        world.reset()

        # Set conveyor belt speed to 0 (until all objects stopped falling to prevent lower objects already moving on the conveyor belt)
        # Use OmniGraph for this since replicator does not work since world is not loaded with open_stage()
        conveyor_node_prim_paths = prim_utils.find_matching_prim_paths("/World/ConveyorTrack(.)*/ConveyorBeltGraph/ConveyorNode")
        for conveyor_node_prim_path in conveyor_node_prim_paths:
            assert og.Controller.set(og.Controller.attribute(conveyor_node_prim_path + ".inputs:velocity"), 0.0)

        # Run simulation until all objects stopped falling
        num_velocities_to_check = 10
        max_lin_velocity_for_finished_falling = 0.001
        last_max_velocities = collections.deque(maxlen=num_velocities_to_check)
        while True:
            # Run simulation for one step and check linear velocity
            world.step(render=True, step_sim=True)

            # If linear velocity of all objects is small, they stopped falling
            if all([np.linalg.norm(object_prim.get_linear_velocity()) < max_lin_velocity_for_finished_falling for object_prim in object_rigid_prims]):
                break

            # Sometimes objects fall through the conveyor belt
            # Then PhysX / Isaac Sim don't calculate the correct velocity which results in objects_stopped_falling not beeing set to True
            # To prevent an endless loop the last 10 max velocities are checked if they are in a specified delta
            # If so one can assume that the objects don't move anymore are not falling anymore
            # Else the max velocity would have large changes due to gravity or the collision with the belt / ground plane / etc
            max_lin_vel = max([np.linalg.norm(object_prim.get_linear_velocity()) for object_prim in object_rigid_prims])
            last_max_velocities.append(max_lin_vel)
            if len(last_max_velocities) < num_velocities_to_check:
                continue
            max_last_lin_vel = max(last_max_velocities)
            min_last_lin_vel = min(last_max_velocities)
            if max_last_lin_vel - min_last_lin_vel < max_lin_velocity_for_finished_falling:
                break

        # Set conveyor belt speed to specified value
        for conveyor_node_prim_path in conveyor_node_prim_paths:
            assert og.Controller.set(og.Controller.attribute(conveyor_node_prim_path + ".inputs:velocity"), conveyor_belt_speed)

        # Wait for min one object to pass min_x_pos_for_record_start before start of recording
        min_x_pos_for_record_start_passed = False
        while not min_x_pos_for_record_start_passed:
            # Run simulation for one step and check x coordinates
            world.step(render=True, step_sim=True)
            min_x_pos_for_record_start_passed = any(
                [object_prim.get_world_pose()[0][0] > min_x_pos_for_record_start for object_prim in object_rigid_prims])

        # Configure replicator "randomization"
        def randomize_sphere_light(sphere_lights, sphere_light_idx, sphere_light_configs):
            sphere_light = sphere_lights[sphere_light_idx]
            sphere_light_config = [sphere_light_per_frame_config[sphere_light_idx] for sphere_light_per_frame_config in sphere_light_configs]
            sphere_light_positions = [config["position"] for config in sphere_light_config]
            sphere_light_colors = [config["color"] for config in sphere_light_config]
            sphere_light_intensities = [config["intensity"] if "intensity" in config else 1000 for config in sphere_light_config]  # if expression required for compatibility with older configs. 1000 is default value according to https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.create.light (08.02.2024)
            with sphere_light:
                rep.modify.attribute("color", rep.distribution.sequence(sphere_light_colors))
                rep.modify.attribute("intensity", rep.distribution.sequence(sphere_light_intensities))
                rep.modify.pose(position=rep.distribution.sequence(sphere_light_positions))
            return sphere_light

        rep.randomizer.register(randomize_sphere_light)

        with rep.trigger.on_frame():  # Change on every rendered frame
            # Change camera position and orientation
            with camera:
                rep.modify.pose(position=rep.distribution.sequence(camera_positions), rotation=rep.distribution.sequence(camera_orientations))

            # Change ground plane color
            with plane_material:
                rep.modify.attribute("diffuse_color_constant", rep.distribution.sequence(ground_plane_colors))

            # Change distant light color, intensity and orientation
            with distant_light:
                rep.modify.attribute("color", rep.distribution.sequence(distant_light_colors))
                rep.modify.attribute("intensity", rep.distribution.sequence(distant_light_intensities))
                rep.modify.pose(rotation=rep.distribution.sequence(distant_light_orientations))

            # Change sphere light color, position and intensities
            for i in range(len(sphere_lights)):
                rep.randomizer.randomize_sphere_light(sphere_lights, i, sphere_light_configs)

            # Change conveyor belt color
            with rep_conveyor_belt_material:
                rep.modify.attribute("diffuse_color_constant", rep.distribution.sequence(conveyor_belt_colors))

            # Change conveyor frame color
            with rep_conveyor_frame_material:
                rep.modify.attribute("diffuse_color_constant", rep.distribution.sequence(conveyor_frame_colors))

        # Initialize writers
        writer_configs = [{"name": "ResumableBasicWriter", "args": {"rgb": True}}]  # TODO remove me
        writers = []
        for writer_config in writer_configs:
            writer = rep.WriterRegistry.get(writer_config["name"])
            writer.initialize(output_dir=out_dir, init_frame_nr=frame_number, **writer_config["args"])
            writer.attach(render_product)
            writers.append(writer)

        # Generate replicator graphs
        rep.orchestrator.preview()

        # Capture data
        for scene_frame_nr in range(num_frames_per_scene):
            if not simulation_app.is_running():
                print("Simulation has been stopped. Exiting...", file=sys.stderr)
                quit_on_error(simulation_app)

            print(f"Writing scene frame {scene_frame_nr + 1} of {num_frames_per_scene} (total frame {frame_number + 1} of {num_frames})")

            # Assign material to object manually since replicator does not support sequential assignment
            for obj_idx, object_prim in enumerate(object_prims):
                material_idx = object_material_assignments[scene_frame_nr][obj_idx]["material_idx"]
                xform_object_prim = XFormPrim(object_prim.GetPrimPath().pathString)
                xform_object_prim.apply_visual_material(materials[material_idx])

            # Run simulation for one step, but don't render, since rendering is done by replicator
            #TODO world.step(render=False, step_sim=True)?  render False?

            # Generate multiple sub-frames for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))
            rep.orchestrator.step(rt_subframes=sub_frames_per_frame)

            # Increase frame_number count
            frame_number += 1

    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()
    exit(0)
    

if __name__ == '__main__':
    conf_path = "custom_dataset/dataset_generation/config_generation/config.json"
    usd_dir = "CAD Models/OBJ_converted"
    out_dir = os.path.abspath("temp_replicator_out")

    main(conf_path, usd_dir, out_dir)
    simulation_app.close()
