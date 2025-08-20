# Regular imports
import argparse
import os
import json
import sys
import numpy as np
from utils.config import parse_writer_args


# Parse args
parser = argparse.ArgumentParser(description="Generates training data for 6-DOF Alignment neural networks with Omniverse Replicator")
parser.add_argument("--headless", help="Run in headless mode", action="store_true")
parser.add_argument("--output_dir", help="Output directory", required=True)
parser.add_argument("--usd_dir", help="Directory containing the USD versions of the CAD models to be used for data generation", required=True)
parser.add_argument("--config_file", help="Path to the JSON configuration file describing the to be generated scenes", required=True)
parser.add_argument("--writer", nargs="*", action="append", help="Configures writers from the resumable_writers plugin package. Argument can be added multiple times", required=True)
args = parser.parse_args(sys.argv[1:])

# Launch Isaac Sim
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless}
simulation_app = SimulationApp(launch_config=CONFIG)


# Omniverse imports and omniverse related imports need to be done after the simulation has been started
import omni.replicator.core as rep
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat

from resumable_writers import load_resumable_writer_plugins
from utils.scene_setup import generate_materials, randomize_sphere_light
from utils.simulation import quit_on_error
from utils.io import quit_if_out_dir_not_empty


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
    # quit_if_out_dir_not_empty(out_dir, simulation_app)

    # Write train val split config
    with open(os.path.join(out_dir, "train_val_scenes.json"), "w") as f:
        json.dump(train_val_split_config, f, indent=4)

    # Scene generation loop
    frame_number = 0
    num_frames = np.sum(np.fromiter((len(scene_configs) for scene_configs in config["scenes"]), int))
    for object_type_specific_scene_configs in config["scenes"]:
        # Parse scene configs
        background_colors = extract_per_scene_config(object_type_specific_scene_configs, "background_color")
        sphere_light_configs = extract_per_scene_config(object_type_specific_scene_configs, "sphere_light_configs")
        num_sphere_lights = len(sphere_light_configs[0])
        object_configs = extract_per_scene_config(object_type_specific_scene_configs, "object_configs")
        if "dome_light_color" in object_type_specific_scene_configs:  # if condition required for compatibility of with old configs
            dome_light_colors = extract_per_scene_config(object_type_specific_scene_configs, "dome_light_color")
            dome_light_intensities = [1000 for _ in range(len(dome_light_colors))]  # 1000 is default value according to
        else:
            dome_light_configs = extract_per_scene_config(object_type_specific_scene_configs, "dome_light_configs")
            dome_light_colors = [dome_light_config["color"] for dome_light_config in dome_light_configs]
            dome_light_intensities = [dome_light_config["intensity"] for dome_light_config in dome_light_configs]

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
        rep.create.plane(scale=10, visible=True, material=plane_material)

        # Add objects
        num_objects_per_scene = len(object_configs[0])
        object_prims = []
        for i, object_config in enumerate(object_configs[0]):
            usd_path = os.path.abspath(os.path.join(usd_dir, object_config["usd_model"]))
            if not os.path.isfile(usd_path):
                quit_on_error(f"USD file at path '{usd_path}' could not be found. Exiting...", simulation_app)
                
            object_prims.append(prims.create_prim(prim_path=f"/objects/object_{i:0{len(str(num_objects_per_scene))}}", usd_path=usd_path, semantic_label=object_config["semantic_class_label"]))

        # Configure replicator "randomization"
        rep.randomizer.register(randomize_sphere_light)

        with rep.trigger.on_frame():  # Change on every rendered frame
            # Change background color
            with plane_material:
                rep.modify.attribute("inputs:diffuse_color_constant", rep.distribution.sequence(background_colors))

            # Change dome_light color
            with dome_light:
                rep.modify.attribute("inputs:color", rep.distribution.sequence(dome_light_colors))
                rep.modify.attribute("inputs:intensity", rep.distribution.sequence(dome_light_intensities))

            # Change sphere light color and position
            for i in range(len(sphere_lights)):
                rep.randomizer.randomize_sphere_light(sphere_lights, i, sphere_light_configs)
        
        # Capture training data
        for current_scene_config in object_type_specific_scene_configs:
            if not simulation_app.is_running():
                quit_on_error("Simulation has been stopped. Exiting...", simulation_app)

            print(f"Writing frame {str(frame_number + 1)} of {num_frames}")

            # Assign material to object manually since replicator does not support sequential assignment
            # Modify pose of object manually since replicator throws error "WritePrimAttribute Error: cannot reshape array of size 3 into shape (2,newaxis)"
            for obj_idx, object_prim in enumerate(object_prims):
                material_idx = current_scene_config["object_configs"][obj_idx]["material_idx"]
                xform_object_prim = XFormPrim(object_prim.GetPrimPath().pathString)
                xform_object_prim.apply_visual_materials(materials[material_idx])
                position = current_scene_config["object_configs"][obj_idx]["pose"]["position"]
                orientation = current_scene_config["object_configs"][obj_idx]["pose"]["orientation"]
                orientation_quaternion = euler_angles_to_quat(np.array(orientation), degrees=True, extrinsic=False)
                xform_object_prim.set_world_poses(positions=np.array([position]), orientations=np.array([orientation_quaternion]))

            # Generate multiple sub-frames for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))
            rep.orchestrator.step(rt_subframes=sub_frames_per_frame)

            # Increase frame_number count
            frame_number += 1
        

if __name__ == '__main__':
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.getcwd(), args.output_dir)
    main(args.config_file, args.usd_dir, out_dir)
    simulation_app.close()
