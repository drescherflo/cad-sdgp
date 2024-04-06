"""
Script to create a config files for the evaluation dataset generation.
The term scene means one data generation run, before the simulator reset.
In one scene multiple frames are generated.
TODO: README and doc comments
"""
import copy
import sys
import json
import os
import argparse

from utils import io, config
from utils.conveyor_config import generate_multiple_object_scenes
from utils.scene_randomization import usd_model_to_semantic_class_label


def generate_scenes(
                    out_dir: str,
                    usd_models: list[str],
                    camera_pos_x: float, camera_pos_y: float, camera_pos_z: float,
                    camera_rot_x: float, camera_rot_y: float, camera_rot_z: float,
                    conveyor_belt_speed: float,
                    distant_light_color_r: float, distant_light_color_g: float, distant_light_color_b: float,
                    distant_light_intensity: float,
                    distant_light_rot_x: float, distant_light_rot_y: float, distant_light_rot_z: float,
                    frame_height: int, frame_width: int,
                    ground_plane_color_r: float, ground_plane_color_g: float, ground_plane_color_b: float,
                    material_names: list[str], materials: list[dict],
                    min_x_pos_for_record_start: float,
                    num_frames_per_scene: int,
                    num_objects_per_scene: int,
                    num_scenes: int,
                    num_sphere_lights: int,
                    sphere_min_intensity: float, sphere_max_intensity: float,
                    sphere_min_x: float, sphere_max_x: float,
                    sphere_min_y: float, sphere_max_y: float,
                    sphere_min_z: float, sphere_max_z: float,
                    object_init_min_x: float, object_init_max_x: float,
                    object_init_min_y: float, object_init_max_y: float,
                    object_init_min_z: float, object_init_max_z: float,
                    sub_frames_per_frame: int,
                    physics_frequency: int,
                    render_frequency: int,
                    args
                    ):
    # Create out dir
    os.makedirs(os.path.join(out_dir), exist_ok=True)
    # Use multiple object scenes as base
    base_scenes = [generate_multiple_object_scenes(
        usd_models, len(materials),
        num_frames_per_scene, num_objects_per_scene, 1,
        num_sphere_lights,
        object_init_min_x, object_init_max_x,
        object_init_min_y, object_init_max_y,
        object_init_min_z, object_init_max_z,
        sphere_min_x, sphere_max_x,
        sphere_min_y, sphere_max_y,
        sphere_min_z, sphere_max_z,
        sphere_min_intensity, sphere_max_intensity,
        distant_light_rot_x, distant_light_rot_x,
        distant_light_rot_y, distant_light_rot_y,
        distant_light_rot_z, distant_light_rot_z,
        camera_pos_x, camera_pos_x,
        camera_pos_y, camera_pos_y,
        camera_pos_z, camera_pos_z,
        camera_rot_x, camera_rot_x,
        camera_rot_y, camera_rot_y,
        camera_rot_z, camera_rot_z,
        distant_light_intensity, distant_light_intensity)[0] for _ in range(num_scenes)]
    for scene in base_scenes:
        # Replace distant light configs
        for distant_light_config in scene["per_frame_config"]["distant_light_configs"]:
            distant_light_config["color"] = [distant_light_color_r, distant_light_color_g, distant_light_color_b]

        # Replace ground plane color
        scene["per_frame_config"]["ground_plane_colors"] = [
            [ground_plane_color_r, ground_plane_color_g, ground_plane_color_b] for _ in
            range(len(scene["per_frame_config"]["ground_plane_colors"]))]

        # Remove conveyor belt colors and conveyor frame colors
        scene["per_frame_config"]["conveyor_belt_colors"] = []
        scene["per_frame_config"]["conveyor_frame_colors"] = []

    for scene_idx, scene in enumerate(base_scenes):
        print("Generating scene {} of {}".format(scene_idx + 1, num_scenes))
        for usd_model in usd_models:
            for mat_idx, material in enumerate(materials):
                modified_scene = copy.deepcopy(scene)
                # Replace all USD models with usd_model
                for object_conf in modified_scene["objects"]:
                    object_conf["usd_model"] = usd_model
                    object_conf["semantic_class_label"] = usd_model_to_semantic_class_label(usd_model)

                # Create missing config and replace materials with current material
                config_materials = [material for _ in range(len(materials))]
                scene_config = {
                    "camera_frame_config": {"frame_height": frame_height, "frame_width": frame_width},
                    "sub_frames_per_frame": sub_frames_per_frame,
                    "materials": config_materials,
                    "scenes": [modified_scene],
                    "conveyor_belt_speed": conveyor_belt_speed,
                    "min_x_pos_for_record_start": min_x_pos_for_record_start,
                    "num_frames_per_scene": num_frames_per_scene,
                    "render_frequency": render_frequency,
                    "physics_frequency": physics_frequency,
                    "eval_dataset": True,
                    "generation_script_args": vars(args)
                }

                # Save config
                file_name = f"{usd_model_to_semantic_class_label(usd_model)}_{scene_idx + 1}_{material_names[mat_idx]}.json"
                file_path = os.path.join(out_dir, file_name)
                with open(file_path, "w") as f:
                    json.dump(scene_config, f, indent=4)

                # Restore mixed object configuration
                modified_scene["objects"] = scene["objects"]

                # Save config with mixed objects
                file_name = f"mixed_{scene_idx + 1}_{material_names[mat_idx]}.json"
                file_path = os.path.join(out_dir, file_name)
                with open(file_path, "w") as f:
                    json.dump(scene_config, f, indent=4)


def main(argv: list[str]) -> None:
    """
    Main function to handle command-line arguments and execute the configuration generation process.

    :param argv: List of command-line arguments.
    :type argv: list[str]
    """

    # Calculate default num frames per scene
    # These are the number of frames the object needs to travel from recording start to the end of the conveyor belt
    default_render_frequency = 60
    default_conveyor_belt_speed = 0.2
    default_min_x_pos_for_record_start = -0.5
    end_x_of_recording = 0.5
    distance_to_capture = end_x_of_recording - default_min_x_pos_for_record_start
    time_for_recording = distance_to_capture / default_conveyor_belt_speed
    default_num_frames_per_scene = int(time_for_recording * default_render_frequency)  # Sim runs at 60 FPS (https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#module-omni.isaac.core.world)

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generates a configuration for the training data generation script")
    parser.add_argument("--usd_dir", help="Directory containing the converted CAD models as USD files", required=True)
    parser.add_argument("--out_dir", default="configs", help="Output directory for the generated configurations")
    parser.add_argument("--frame_width", default=480, type=int, help="Width of the generated frames")
    parser.add_argument("--frame_height", default=360, type=int, help="Width of the generated frames")
    parser.add_argument("--sub_frames_per_frame", default=32, type=int,
                        help="Number of frames to render before saving the frame to avoid artifacts after fast object movement")
    parser.add_argument("--num_frames_per_scene", default=default_num_frames_per_scene, type=int,
                        help="Number of frames to record per simulation run / simulation scene")
    parser.add_argument("--num_objects_per_scene", default=100, type=int,
                        help="Specifies the number of objects in the scene")
    parser.add_argument("--num_objects_per_cluttered_scene", default=100, type=int,
                        help="Specifies the number of objects in the cluttered scene")
    parser.add_argument("--num_scenes", default=5, type=int,
                        help="Specifies the number of uncluttered scenes to generate")
    parser.add_argument("--num_cluttered_scenes", default=5, type=int,
                        help="Specifies the number of cluttered scenes to generate")
    parser.add_argument("--num_sphere_lights", default=0, type=int,
                        help="Specifies the number of sphere lights with random light color in the scene")
    parser.add_argument("--object_init_min_x", default=-2.0, type=float,
                        help="The minimum initial x coordinate of the object in the scene")
    parser.add_argument("--object_init_max_x", default=2.0, type=float,
                        help="The maximum initial x coordinate of the object in the scene")
    parser.add_argument("--object_init_min_y", default=-0.4, type=float,
                        help="The minimum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_max_y", default=0.4, type=float,
                        help="The maximum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_min_z", default=3, type=float,
                        help="The minimum initial z coordinate of the object in the scene")
    parser.add_argument("--object_init_max_z", default=6, type=float,
                        help="The maximum initial z coordinate of the object in the scene")
    parser.add_argument("--cluttered_scene_object_init_min_x", default=-0.25, type=float,
                        help="The minimum initial x coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_max_x", default=0.25, type=float,
                        help="The maximum initial x coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_min_y", default=-0.4, type=float,
                        help="The minimum initial y coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_max_y", default=0.4, type=float,
                        help="The maximum initial y coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_min_z", default=3, type=float,
                        help="The minimum initial z coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_max_z", default=20, type=float,
                        help="The maximum initial z coordinate of the object in the cluttered scene")
    parser.add_argument("--sphere_min_x", default=-2.5, type=float,
                        help="The minimum x coordinate of light spheres in the scene")
    parser.add_argument("--sphere_max_x", default=2.5, type=float,
                        help="The maximum x coordinate of light spheres in the scene")
    parser.add_argument("--sphere_min_y", default=-0.5, type=float,
                        help="The minimum y coordinate of light spheres in the scene")
    parser.add_argument("--sphere_max_y", default=0.5, type=float,
                        help="The maximum y coordinate of light spheres in the scene")
    parser.add_argument("--sphere_min_z", default=2.3, type=float,
                        help="The minimum z coordinate of light spheres in the scene")
    parser.add_argument("--sphere_max_z", default=2.7, type=float,
                        help="The maximum z coordinate of light spheres in the scene")
    parser.add_argument("--sphere_min_intensity", default=1000, type=float,
                        help="The minimum light intensity of a sphere light")
    parser.add_argument("--sphere_max_intensity", default=5000, type=float,
                        help="The maximum light intensity of a sphere light")
    parser.add_argument("--camera_pos_x", default=0, type=float,
                        help="The x coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_y", default=0, type=float,
                        help="The y coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_z", default=3.5, type=float,
                        help="The z coordinate of the camera in the scene")
    parser.add_argument("--camera_rot_x", default=0, type=float,
                        help="The rotation of the camera around the x axis in degrees")
    parser.add_argument("--camera_rot_y", default=-90, type=float,
                        help="The rotation of the camera around the y axis in degrees")
    parser.add_argument("--camera_rot_z", default=0, type=float,
                        help="The rotation of the camera around the z axis in degrees")
    parser.add_argument("--distant_light_rot_x", default=0, type=float,
                        help="The rotation around x of the direct light in the scene")
    parser.add_argument("--distant_light_rot_y", default=0, type=float,
                        help="The rotation around y of the direct light in the scene")
    parser.add_argument("--distant_light_rot_z", default=0, type=float,
                        help="The z coordinate of the direct light in the scene")
    parser.add_argument("--distant_light_color_r", default=1, type=float,
                        help="The direct light light color in rgb (r value, value should be between 0 and 1)")
    parser.add_argument("--distant_light_color_g", default=1, type=float,
                        help="The direct light light color in rgb (g value, value should be between 0 and 1)")
    parser.add_argument("--distant_light_color_b", default=1, type=float,
                        help="The direct light light color in rgb (b value, value should be between 0 and 1))")
    parser.add_argument("--distant_light_intensity", default=1000, type=float,
                        help="The minimum light intensity of the direct light")
    parser.add_argument("--conveyor_belt_speed", default=default_conveyor_belt_speed, type=float,
                        help="The speed of the conveyor belt in the simulation")
    parser.add_argument("--min_x_pos_for_record_start", default=default_min_x_pos_for_record_start, type=float,
                        help="Defines the minimum x coordinate at least one object needs to have passed to start writing the dataset")
    parser.add_argument("--render_frequency", default=default_render_frequency, type=int,
                        help="The render frequency in Hz")
    parser.add_argument("--physics_frequency", default=360, type=int,
                        help="The frequency at which the physics are calculated")
    parser.add_argument("--ground_plane_color_r", default=1, type=float,
                        help="The ground plane color in rgb (r value, value should be between 0 and 1)")
    parser.add_argument("--ground_plane_color_g", default=1, type=float,
                        help="The ground plane color in rgb (g value, value should be between 0 and 1)")
    parser.add_argument("--ground_plane_color_b", default=1, type=float,
                        help="The ground plane color in rgb (b value, value should be between 0 and 1))")

    # Parse args
    args = parser.parse_args(argv)

    # Assign variables
    usd_dir = args.usd_dir
    out_dir = args.out_dir
    frame_width = args.frame_width
    frame_height = args.frame_height
    sub_frames_per_frame = args.sub_frames_per_frame
    num_frames_per_scene = args.num_frames_per_scene
    num_objects_per_scene = args.num_objects_per_scene
    num_objects_per_cluttered_scene = args.num_objects_per_cluttered_scene
    num_scenes = args.num_scenes
    num_cluttered_scenes = args.num_cluttered_scenes
    num_sphere_lights = args.num_sphere_lights
    object_init_min_x = args.object_init_min_x
    object_init_max_x = args.object_init_max_x
    object_init_min_y = args.object_init_min_y
    object_init_max_y = args.object_init_max_y
    object_init_min_z = args.object_init_min_z
    object_init_max_z = args.object_init_max_z
    cluttered_scene_object_init_min_x = args.cluttered_scene_object_init_min_x
    cluttered_scene_object_init_max_x = args.cluttered_scene_object_init_max_x
    cluttered_scene_object_init_min_y = args.cluttered_scene_object_init_min_y
    cluttered_scene_object_init_max_y = args.cluttered_scene_object_init_max_y
    cluttered_scene_object_init_min_z = args.cluttered_scene_object_init_min_z
    cluttered_scene_object_init_max_z = args.cluttered_scene_object_init_max_z
    sphere_min_x = args.sphere_min_x
    sphere_max_x = args.sphere_max_x
    sphere_min_y = args.sphere_min_y
    sphere_max_y = args.sphere_max_y
    sphere_min_z = args.sphere_min_z
    sphere_max_z = args.sphere_max_z
    sphere_min_intensity = args.sphere_min_intensity
    sphere_max_intensity = args.sphere_max_intensity
    camera_pos_x = args.camera_pos_x
    camera_pos_y = args.camera_pos_y
    camera_pos_z = args.camera_pos_z
    camera_rot_x = args.camera_rot_x
    camera_rot_y = args.camera_rot_y
    camera_rot_z = args.camera_rot_z
    distant_light_rot_x = args.distant_light_rot_x
    distant_light_rot_y = args.distant_light_rot_y
    distant_light_rot_z = args.distant_light_rot_z
    distant_light_intensity = args.distant_light_intensity
    distant_light_color_r = args.distant_light_color_r
    distant_light_color_g = args.distant_light_color_g
    distant_light_color_b = args.distant_light_color_b
    conveyor_belt_speed = args.conveyor_belt_speed
    min_x_pos_for_record_start = args.min_x_pos_for_record_start
    render_frequency = args.render_frequency
    physics_frequency = args.physics_frequency
    ground_plane_color_r = args.ground_plane_color_r
    ground_plane_color_g = args.ground_plane_color_g
    ground_plane_color_b = args.ground_plane_color_b

    # Check range args for plausibility
    config.check_range_plausibility(object_init_min_x, object_init_max_x)
    config.check_range_plausibility(object_init_min_y, object_init_max_y)
    config.check_range_plausibility(object_init_min_z, object_init_max_z)
    config.check_range_plausibility(sphere_min_x, sphere_max_x)
    config.check_range_plausibility(sphere_min_y, sphere_max_y)
    config.check_range_plausibility(sphere_min_z, sphere_max_z)
    config.check_range_plausibility(sphere_min_intensity, sphere_max_intensity)

    # Check for existing config at out_path
    if os.path.exists(out_dir):
        print(f"Output directory at '{out_dir}' already exists. Exiting...")
        exit(-1)

    # Test usd_dir
    if not os.path.isdir(usd_dir):
        print("USD directory does not exist. Exiting...")
        exit(-1)

    # Create out dir
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    usd_models = io.get_usd_models(usd_dir)

    # Create eval materials config
    materials = [
        {
            "material_idx": 0,
            "is_glass": False,
            "color": [0.2, 0.2, 0.2],
            "surface_roughness": 0.5
        },  # default material (default values from Isaac Sim OmniPBR)
        {
            "material_idx": 1,
            "is_glass": False,
            "color": [0.2, 0.2, 0.2],
            "surface_roughness": 0
        },  # metallic, high reflective
        {
            "material_idx": 2,
            "is_glass": False,
            "color": [0.2, 0.2, 0.2],
            "surface_roughness": 1
        },  # metallic, rough
        {
            "material_idx": 3,
            "is_glass": True,
            "color": [1, 1, 1]
        },  # default white glass / plastic (default values from Isaac Sim OmniGlass)
        {
            "material_idx": 4,
            "is_glass": False,
            "color": [0, 0, 0],
            "surface_roughness": 0.5
        },  # black material
        {
            "material_idx": 5,
            "is_conveyor": True
        },  # conveyor belt material
        {
            "material_idx": 6,
            "is_glass": False,
            "color": [0, 1, 0],
            "surface_roughness": 0.5
        },   # green material
    ]

    material_names = ["default", "metal-reflective", "metal-rough", "glass", "black", "conveyor", "green"]

    print("Generating uncluttered scenes...")
    uncluttered_out_dir = os.path.join(out_dir, "uncluttered")
    generate_scenes(uncluttered_out_dir,
                    usd_models,
                    camera_pos_x, camera_pos_y, camera_pos_z,
                    camera_rot_x, camera_rot_y, camera_rot_z,
                    conveyor_belt_speed,
                    distant_light_color_r, distant_light_color_g, distant_light_color_b,
                    distant_light_intensity,
                    distant_light_rot_x, distant_light_rot_y, distant_light_rot_z,
                    frame_height, frame_width,
                    ground_plane_color_r, ground_plane_color_g, ground_plane_color_b,
                    material_names, materials,
                    min_x_pos_for_record_start,
                    num_frames_per_scene,
                    num_objects_per_scene,
                    num_scenes,
                    num_sphere_lights,
                    sphere_min_intensity, sphere_max_intensity,
                    sphere_min_x, sphere_max_x,
                    sphere_min_y, sphere_max_y,
                    sphere_min_z, sphere_max_z,
                    object_init_min_x, object_init_max_x,
                    object_init_min_y, object_init_max_y,
                    object_init_min_z, object_init_max_z,
                    sub_frames_per_frame,
                    physics_frequency,
                    render_frequency,
                    args)

    print("Generating cluttered scenes...")
    cluttered_out_dir = os.path.join(out_dir, "cluttered")
    generate_scenes(cluttered_out_dir,
                    usd_models,
                    camera_pos_x, camera_pos_y, camera_pos_z,
                    camera_rot_x, camera_rot_y, camera_rot_z,
                    conveyor_belt_speed,
                    distant_light_color_r, distant_light_color_g, distant_light_color_b,
                    distant_light_intensity,
                    distant_light_rot_x, distant_light_rot_y, distant_light_rot_z,
                    frame_height, frame_width,
                    ground_plane_color_r, ground_plane_color_g, ground_plane_color_b,
                    material_names, materials,
                    min_x_pos_for_record_start,
                    num_frames_per_scene,
                    num_objects_per_cluttered_scene,
                    num_cluttered_scenes,
                    num_sphere_lights,
                    sphere_min_intensity, sphere_max_intensity,
                    sphere_min_x, sphere_max_x,
                    sphere_min_y, sphere_max_y,
                    sphere_min_z, sphere_max_z,
                    cluttered_scene_object_init_min_x, cluttered_scene_object_init_max_x,
                    cluttered_scene_object_init_min_y, cluttered_scene_object_init_max_y,
                    cluttered_scene_object_init_min_z, cluttered_scene_object_init_max_z,
                    sub_frames_per_frame,
                    physics_frequency,
                    render_frequency,
                    args)


if __name__ == '__main__':
    main(sys.argv[1:])
