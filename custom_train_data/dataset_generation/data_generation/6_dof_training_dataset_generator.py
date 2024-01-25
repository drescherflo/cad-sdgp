# Regular imports
import argparse
import os
import json
import sys
import numpy as np
import importlib


def parse_writer_init_args(writer_init_args: list[str]) -> dict:
    """
    Parses initialization arguments for a writer.

    Converts argument strings into a dictionary format, interpreting values as booleans if they match 'true'.
    E.g. ['rgb=true', 'depth=true'] is converted to {'rgb': True, 'depth': True}.

    :param writer_init_args: A list of string arguments.
    :type writer_init_args: list[str]
    :return: A dictionary mapping argument names to their parsed boolean values.
    :rtype: dict
    """

    args_dict = {}
    for arg in writer_init_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args_dict[key] = value.lower() == 'true'
    return args_dict


def parse_writer_args(writer_args: list[list[str]]) -> list[dict]:
    """
    Parses arguments for multiple writers.

    Converts a list of argument lists into a dictionary format, suitable for initializing multiple writers.

    :param writer_args: A list of lists, each containing arguments for a specific writer.
    :type writer_args: list[list[str]]
    :return: A dictionary containing configurations for each writer.
    :rtype: list[dict]
    """

    writers = []
    for writer_arg in writer_args:
        writer_conf = {"name": writer_arg[0], "args": parse_writer_init_args(writer_arg[1:])}
        writers.append(writer_conf)

    return writers


# Parse args
parser = argparse.ArgumentParser(description="Generates training data for 6-DOF Alignment neural networks with Omniverse Replicator")
parser.add_argument("--headless", help="Run in headless mode", action="store_true")
parser.add_argument("--output_dir", help="Output directory", required=True)
parser.add_argument("--usd_dir", help="Directory containing the USD versions of the CAD models to be used for data generation", required=True)
parser.add_argument("--config_file", help="Path to the JSON configuration file describing the to be generated scenes", required=True)
parser.add_argument("--writer", nargs="*", action="append", help="Configures writers from the resumable_writers plugin package. Argument can be added multiple times", required=True)
args = parser.parse_args(sys.argv[1:])

# Launch Isaac Sim
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless}
simulation_app = SimulationApp(launch_config=CONFIG)


# Omniverse imports
import omni
import omni.replicator.core as rep
import omni.graph.core as og
from omni.isaac.core.utils.stage import create_new_stage
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import prims, extensions
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.materials import OmniPBR, OmniGlass
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from resumable_writers.writer_interface import ResumableWriterInterface


def generate_materials(materials_config) -> list[OmniGlass | OmniPBR]:
    materials = []
    for material_config in materials_config:
        if material_config["is_glass"]:
            materials.append(OmniGlass(f"/obj_materials/material_{material_config['material_idx']}", color=np.array(material_config["color"])))
        else:
            material = OmniPBR(f"/obj_materials/material_{material_config['material_idx']}", color=np.array(material_config["color"]))
            material.set_reflection_roughness(material_config["surface_roughness"])
            materials.append(material)

    return materials


def extract_per_scene_config(scenes: list[dict], config_key: str) -> list:
    """
    Extracts a specific configuration value for each scene from a list of scenes.

    :param scenes: A list of dictionaries, each representing a scene.
    :param config_key: The key for the configuration value to be extracted.
    :type scenes: list[dict]
    :type config_key: str
    :return: A list of configuration values extracted for each scene.
    :rtype: list
    """

    return [scene[config_key] for scene in scenes]


def load_resumable_writer_plugins(plugin_dir: str, plugin_package_name) -> None:
    """
    Loads resumable writer plugins from a specified directory.

    :param plugin_dir: Directory containing the writer plugin files.
    :param plugin_package_name: Name of the package where writer plugins are located.
    :type plugin_dir: str
    :type plugin_package_name: str
    """

    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            module = importlib.import_module('.' + module_name, package=plugin_package_name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, ResumableWriterInterface) and attribute is not ResumableWriterInterface:
                    rep.WriterRegistry.register(attribute)


def __quit_on_error() -> None:
    """
    Terminates the program execution in case of an error.
    This function should be called when an unrecoverable error is encountered.
    """

    simulation_app.close()
    exit(-1)


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
    plugin_package_name = "resumable_writers"
    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(script_location_dir, plugin_package_name) # plugin_dir has to be relative to the script. This is not always the case, e.g. when debugging with VSCode
    load_resumable_writer_plugins(plugin_dir, plugin_package_name)

    # Load json config
    with open(conf_path, "r") as f:
        config = json.load(f)

    # Parse general config
    image_width = config["camera_config"]["frame_width"]
    image_height = config["camera_config"]["frame_height"]
    camera_position = config["camera_config"]["pose"]["position"]
    camera_orientation = config["camera_config"]["pose"]["orientation"]
    sub_frames_per_frame = config["sub_frames_per_frame"]

    # Parse writer config
    writer_configs = parse_writer_args(args.writer)

    # Write train val split
    train_val_split_config = {
        "train_scenes": config["train_scenes"],
        "val_scenes": config["val_scenes"]
    }

    # Create out_dir if necessary
    os.makedirs(out_dir, exist_ok=True)

    # Check for empty out_dir
    if len(os.listdir(out_dir)) != 0:
        print("Output directory is not empty. Exiting...")
        __quit_on_error()

    with open(os.path.join(out_dir, "train_val_scenes.json"), "w") as f:
        json.dump(train_val_split_config, f, indent=4)

    # Scene generation loop
    frame_number = 0
    num_frames = np.sum(np.fromiter((len(scene_configs) for scene_configs in config["scenes"]), int))
    for object_type_specific_scene_configs in config["scenes"]:
        # Parse scene configs
        background_colors = extract_per_scene_config(object_type_specific_scene_configs, "background_color")
        dome_light_colors = extract_per_scene_config(object_type_specific_scene_configs, "dome_light_color")
        sphere_light_configs = extract_per_scene_config(object_type_specific_scene_configs, "sphere_light_configs")
        num_sphere_lights = len(sphere_light_configs[0])
        object_configs = extract_per_scene_config(object_type_specific_scene_configs, "object_configs")

        # Reset simulation
        # We need to reset every num_frames_per_object because the usd models change after num_frames_per_object
        create_new_stage()  # Create new stage and delete previous contents

        # Generate object materials
        materials = generate_materials(config["materials"])

        # Add camera
        camera = rep.create.camera(position=camera_position, rotation=camera_orientation)
        render_product = rep.create.render_product(camera=camera, resolution=(image_width, image_height))

        # Initialize writers
        writers = []
        for writer_config in writer_configs:
            writer = rep.WriterRegistry.get(writer_config["name"])
            writer.initialize(output_dir=out_dir, init_frame_nr=frame_number, **writer_config["args"])
            writer.attach(render_product)
            writers.append(writer)

        # Add dome light
        dome_light = rep.create.light(light_type="Dome")

        # Add sphere_lights
        sphere_lights = [rep.create.light(light_type="Sphere") for _ in range(num_sphere_lights)]

        # Add ground plane to scene as background
        plane_material = rep.create.material_omnipbr(roughness=1)
        plane = rep.create.plane(scale=10, visible=True, material=plane_material)

        # Add objects
        num_objects_per_scene = len(object_configs[0])
        object_prims = []
        for i, object_config in enumerate(object_configs[0]):
            usd_path = os.path.join(usd_dir, object_config["usd_model"])
            if not os.path.isfile(usd_path):
                print(f"USD file at path '{usd_path}' could not be found. Exiting...", file=sys.stderr)
                __quit_on_error()
                
            object_prims.append(prims.create_prim(prim_path=f"/objects/object_{i:0{len(str(num_objects_per_scene))}}", usd_path=usd_path, semantic_label=object_config["semantic_class_label"]))

        # Configure replicator "randomization"
        def randomize_sphere_light(sphere_lights, sphere_light_idx, sphere_light_configs):
            sphere_light = sphere_lights[sphere_light_idx]
            sphere_light_config = [sphere_light_per_frame_config[sphere_light_idx] for sphere_light_per_frame_config in sphere_light_configs]
            sphere_light_positions = [config["position"] for config in sphere_light_config]
            sphere_light_colors = [config["color"] for config in sphere_light_config]
            with sphere_light:
                rep.modify.attribute("color", rep.distribution.sequence(sphere_light_colors))
                rep.modify.pose(position=rep.distribution.sequence(sphere_light_positions))
            return sphere_light

        rep.randomizer.register(randomize_sphere_light)

        with rep.trigger.on_frame():  # Change on every rendered frame
            # Change background color
            with plane_material:
                rep.modify.attribute("diffuse_color_constant", rep.distribution.sequence(background_colors))

            # Change dome_light color
            with dome_light:
                rep.modify.attribute("color", rep.distribution.sequence(dome_light_colors))

            # Change sphere light color and position
            for i in range(len(sphere_lights)):
                rep.randomizer.randomize_sphere_light(sphere_lights, i, sphere_light_configs)
        
        # Capture training data
        for current_scene_config in object_type_specific_scene_configs:
            if not simulation_app.is_running():
                print("Simulation has been stopped. Exiting...")
                __quit_on_error()

            print(f"Writing frame {str(frame_number + 1)} of {num_frames}")

            # Assign material to object manually since replicator does not support sequential assignment
            # Modify pose of object manually since replicator throws error "WritePrimAttribute Error: cannot reshape array of size 3 into shape (2,newaxis)"
            for obj_idx, object_prim in enumerate(object_prims):
                material_idx = current_scene_config["object_configs"][obj_idx]["material_idx"]
                xform_object_prim = XFormPrim(object_prim.GetPrimPath().pathString)
                xform_object_prim.apply_visual_material(materials[material_idx])
                position = current_scene_config["object_configs"][obj_idx]["pose"]["position"]
                orientation = current_scene_config["object_configs"][obj_idx]["pose"]["orientation"]
                orientation_quaternion = euler_angles_to_quat(np.array(orientation), degrees=True, extrinsic=False)
                xform_object_prim.set_world_pose(position=position, orientation=orientation_quaternion)

            # Generate multiple sub-frames for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))
            rep.orchestrator.step(rt_subframes=sub_frames_per_frame)

            # Increase frame_number count
            frame_number += 1
        

if __name__ == '__main__':
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.getcwd(), args.output_dir)
    main(args.config_file, args.usd_dir, out_dir)
    simulation_app.close()
