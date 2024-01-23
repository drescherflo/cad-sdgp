# Regular imports
import argparse
import os
import json
import sys
import numpy as np
import importlib

# Parse args
parser = argparse.ArgumentParser()
args = parser.parse_args(sys.argv[1:])

# TODO parse headless
# TODO parse out_dir
# TODO parse usd_dir
# TODO parse conf_path


# Launch Isaac Sim
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
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
    return [scene[config_key] for scene in scenes]


def load_resumable_writer_plugins(plugin_dir: str, plugin_package_name) -> None:
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            module = importlib.import_module('.' + module_name, package=plugin_package_name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, ResumableWriterInterface) and attribute is not ResumableWriterInterface:
                    rep.WriterRegistry.register(attribute)


def main(conf_path: str, usd_dir: str, out_dir: str) -> None:
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
    usd_models = config["usd_models"]
    writer_configs = config["writer_configs"]
    num_frames_per_object = config["num_frames_per_object"]

    # Write train val split
    train_val_split_config = {
        "train_scenes": config["train_scenes"],
        "val_scenes": config["val_scenes"]
    }

    os.makedirs(out_dir, exist_ok=True)
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
                simulation_app.close()
                exit(-1)
                
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

        # TODO: remove following 4 lines
        # Isaac Sim run-loop (only for testing do NOT use this when generating data)
        #rep.orchestrator.preview()
        #while simulation_app.is_running():
        #    simulation_app.update()
        
        # Capture training data
        for current_scene_config in object_type_specific_scene_configs:
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

            # Generate multiple subframes for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))
            rep.orchestrator.step(rt_subframes=sub_frames_per_frame)

            # Increase frame_number count
            frame_number += 1
        

if __name__ == '__main__':
    conf_path = "custom_train_data/replicator_data_generation/config_generation/generated_configs/6_dof_only_simple_object.json"
    usd_dir = "CAD Models/OBJ_converted"
    out_dir = "temp_replicator_out"
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(os.getcwd(), out_dir)
    main(conf_path, usd_dir, out_dir)
    simulation_app.close()
