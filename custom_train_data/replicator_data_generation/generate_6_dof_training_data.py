# Regular imports
import argparse
import os
import json
import sys
import numpy as np

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


def main(conf_path: str, usd_dir: str, out_dir: str) -> None:
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

    # Scene generation loop
    for object_type_specific_scene_configs in config["scenes"]:
        # Parse scene configs
        background_colors = extract_per_scene_config(object_type_specific_scene_configs, "background_color")
        dome_light_colors = extract_per_scene_config(object_type_specific_scene_configs, "dome_light_color")
        sphere_light_configs = extract_per_scene_config(object_type_specific_scene_configs, "sphere_light_configs")
        num_sphere_lights = len(sphere_light_configs[0])
        object_configs = extract_per_scene_config(object_type_specific_scene_configs, "object_configs")
        num_objects_per_scene = len(object_configs[0])

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
            writer.initialize(output_dir=out_dir, **writer_config["args"])
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
        object_prims = []
        for i, object_config in enumerate(object_configs[0]):
            usd_path = os.path.join(usd_dir, object_config["usd_model"])
            if not os.path.isfile(usd_path):
                print(f"USD file at path '{usd_path}' could not be found. Exiting...", file=sys.stderr)
                simulation_app.close()
                exit(-1)
                
            object_prims.append(prims.create_prim(prim_path=f"/objects/object_{i}", usd_path=usd_path, semantic_label=object_config["semantic_class_label"]))

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

        def randomize_objects(object_prims, object_idx, object_configs, materials):
            object_prim = object_prims[object_idx]
            rep_object_prim = rep.get.xform(path_pattern=object_prim.GetPrimPath().pathString)
            object_config = [object_per_frame_config[object_idx] for object_per_frame_config in object_configs]
            object_positions = [config["pose"]["position"] for config in object_config]
            object_orientations = [config["pose"]["orientation"] for config in object_config]
            object_material_indices = [config["material_idx"] for config in object_config]
            object_materials = [materials[index] for index in object_material_indices]
            rep_object_materials = [rep.get.material(path_pattern=material.prim_path) for material in object_materials]
            with rep_object_prim:
                rep.modify.material(rep.distribution.sequence(rep_object_materials))
                rep.modify.pose(position=rep.distribution.sequence(object_positions), rotation=rep.distribution.sequence(object_orientations))
            return rep_object_prim
        
        rep.randomizer.register(randomize_objects)

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
            
            # Change object position, orientation and material
            for i in range(len(object_prims)):
                rep.randomizer.randomize_objects(object_prims, i, object_configs, materials)

        # TODO: remove following 4 lines
        # Isaac Sim run-loop (only for testing do NOT use this when generating data)
        rep.orchestrator.preview()
        while simulation_app.is_running():
            simulation_app.update()
        
        # Capture training data
        num_scenes = len(object_type_specific_scene_configs)
        for scene_nr in range(num_scenes):
            print(f"Writing frame {str(scene_nr + 1)} of {num_scenes}")

            # Generate multiple subframes for 1 frame for better quality (see https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples (08.01.2024))
            rep.orchestrator.step(rt_subframes=sub_frames_per_frame)

        

if __name__ == '__main__':
    conf_path = "custom_train_data/replicator_data_generation/generated_configs/6_dof_only_simple_object.json"
    usd_dir = "CAD Models/OBJ_converted"
    out_dir = os.path.join(os.getcwd(), "temp_replicator_out")
    main(conf_path, usd_dir, out_dir)
    simulation_app.close()
