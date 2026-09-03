"""
Script to create a config file for the conveyor dataset generation.
The term scene means one data generation run, before the simulator reset.
In one scene multiple frames are generated.
"""

import sys
import json
import os
import argparse

from utils import config, io, camera_intrinsics
from utils.conveyor_config import generate_config


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
    default_min_x_pos_for_record_start = -1.5
    end_x_of_conveyor_in_simulation = 2.0
    distance_to_capture = end_x_of_conveyor_in_simulation - default_min_x_pos_for_record_start
    time_for_recording = distance_to_capture / default_conveyor_belt_speed
    default_num_frames_per_scene = int(time_for_recording * default_render_frequency)  # Sim runs at 60 FPS (https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#module-omni.isaac.core.world)

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generates a configuration for the training data generation script")
    parser.add_argument("--usd_dir", help="Directory containing the converted CAD models as USD files", required=True)
    parser.add_argument("--out_path", default="config.json", help="Output path for the generated configuration")
    parser.add_argument("--frame_width", default=480, type=int, help="Width of the generated frames")
    parser.add_argument("--frame_height", default=360, type=int, help="Width of the generated frames")
    parser.add_argument("--sub_frames_per_frame", default=32, type=int,
                        help="Number of frames to render before saving the frame to avoid artifacts after fast object movement")
    parser.add_argument("--num_random_materials", default=1000, type=int,
                        help="Defines the number of random materials to generate")
    parser.add_argument("--probability_of_glass_material", default=0.5, type=float,
                        help="Defines the probability of generating glass objects")
    parser.add_argument("--num_frames_per_scene", default=default_num_frames_per_scene, type=int,
                        help="Number of frames to record per simulation run / simulation scene")
    parser.add_argument("--num_objects_per_scene", default=100, type=int,
                        help="Specifies the number of objects in the scene")
    parser.add_argument("--num_scenes_per_object", default=10, type=int,
                        help="Specifies the number of scenes to generate for each object in the USD directory. Additionally n * num_scenes_per_object scenes will be generated with all objects in the scene. (n is num_frames_per_object times objects in the USD directory)")
    parser.add_argument("--num_sphere_lights", default=5, type=int,
                        help="Specifies the number of sphere lights with random light color in the scene")
    parser.add_argument("--train_val_split", default=0.2, type=float,
                        help="Sets the train and validation split of the generated dataset. The default value of 0.2 means that 20 percent of the dataset are assigned to the validation dataset")
    parser.add_argument("--object_init_min_x", default=-2.0, type=float,
                        help="The minimum initial x coordinate of the object in the scene")
    parser.add_argument("--object_init_max_x", default=-1.5, type=float,
                        help="The maximum initial x coordinate of the object in the scene")
    parser.add_argument("--object_init_min_y", default=-0.4, type=float,
                        help="The minimum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_max_y", default=0.4, type=float,
                        help="The maximum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_min_z", default=3, type=float,
                        help="The minimum initial z coordinate of the object in the scene")
    parser.add_argument("--object_init_max_z", default=6, type=float,
                        help="The maximum initial z coordinate of the object in the scene")
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
    parser.add_argument("--camera_pos_min_x", default=-0.5, type=float,
                        help="The minimum x coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_max_x", default=0.5, type=float,
                        help="The maximum x coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_min_y", default=-0.3, type=float,
                        help="The minimum y coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_max_y", default=0.3, type=float,
                        help="The maximum y coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_min_z", default=3, type=float,
                        help="The minimum z coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_max_z", default=6, type=float,
                        help="The maximum z coordinate of the camera in the scene")
    parser.add_argument("--camera_rot_min_x", default=-180, type=float,
                        help="The minimum rotation of the camera around the x axis in degrees")
    parser.add_argument("--camera_rot_max_x", default=180, type=float,
                        help="The maximum rotation of the camera around the x axis in degrees")
    parser.add_argument("--camera_rot_min_y", default=-120, type=float,
                        help="The minimum rotation of the camera around the y axis in degrees")
    parser.add_argument("--camera_rot_max_y", default=-60, type=float,
                        help="The maximum rotation of the camera around the y axis in degrees")
    parser.add_argument("--camera_rot_min_z", default=-180, type=float,
                        help="The minimum rotation of the camera around the z axis in degrees")
    parser.add_argument("--camera_rot_max_z", default=180, type=float,
                        help="The maximum rotation of the camera around the z axis in degrees")
    parser.add_argument("--distant_light_min_rot_x", default=-90, type=float,
                        help="The minimum rotation around x of the direct light in the scene")
    parser.add_argument("--distant_light_max_rot_x", default=90, type=float,
                        help="The maximum rotation around x of the direct light in the scene")
    parser.add_argument("--distant_light_min_rot_y", default=-90, type=float,
                        help="The minimum rotation around y of the direct light in the scene")
    parser.add_argument("--distant_light_max_rot_y", default=90, type=float,
                        help="The maximum rotation around y of the direct light in the scene")
    parser.add_argument("--distant_light_min_rot_z", default=-180, type=float,
                        help="The minimum rotation around z of the direct light in the scene")
    parser.add_argument("--distant_light_max_rot_z", default=180, type=float,
                        help="The maximum rotation around z of the direct light in the scene")
    parser.add_argument("--distant_light_min_intensity", default=200, type=float,
                        help="The minimum light intensity of the direct light")
    parser.add_argument("--distant_light_max_intensity", default=1000, type=float,
                        help="The minimum light intensity of the direct light")
    parser.add_argument("--conveyor_belt_speed", default=default_conveyor_belt_speed, type=float,
                        help="The speed of the conveyor belt in the simulation")
    parser.add_argument("--min_x_pos_for_record_start", default=default_min_x_pos_for_record_start, type=float,
                        help="Defines the minimum x coordinate at least one object needs to have passed to start writing the dataset")
    parser.add_argument("--render_frequency", default=default_render_frequency, type=int,
                        help="The render frequency in Hz")
    parser.add_argument("--physics_frequency", default=360, type=int,
                        help="The frequency at which the physics are calculated")
    camera_intrinsics.add_camera_intrinsics_args(parser)

    # Parse args
    args = parser.parse_args(argv)

    # Assign variables
    usd_dir = args.usd_dir
    out_path = args.out_path
    frame_width = args.frame_width
    frame_height = args.frame_height
    sub_frames_per_frame = args.sub_frames_per_frame
    num_random_materials = args.num_random_materials
    probability_of_glass_material = args.probability_of_glass_material
    if probability_of_glass_material < 0:
        probability_of_glass_material = 0
    if probability_of_glass_material > 1:
        probability_of_glass_material = 1
    num_frames_per_scene = args.num_frames_per_scene
    num_objects_per_scene = args.num_objects_per_scene
    num_scenes_per_object = args.num_scenes_per_object
    num_sphere_lights = args.num_sphere_lights
    val_dataset_share = args.train_val_split
    object_init_min_x = args.object_init_min_x
    object_init_max_x = args.object_init_max_x
    object_init_min_y = args.object_init_min_y
    object_init_max_y = args.object_init_max_y
    object_init_min_z = args.object_init_min_z
    object_init_max_z = args.object_init_max_z
    sphere_min_x = args.sphere_min_x
    sphere_max_x = args.sphere_max_x
    sphere_min_y = args.sphere_min_y
    sphere_max_y = args.sphere_max_y
    sphere_min_z = args.sphere_min_z
    sphere_max_z = args.sphere_max_z
    sphere_min_intensity = args.sphere_min_intensity
    sphere_max_intensity = args.sphere_max_intensity
    camera_pos_min_x = args.camera_pos_min_x
    camera_pos_max_x = args.camera_pos_max_x
    camera_pos_min_y = args.camera_pos_min_y
    camera_pos_max_y = args.camera_pos_max_y
    camera_pos_min_z = args.camera_pos_min_z
    camera_pos_max_z = args.camera_pos_max_z
    camera_rot_min_x = args.camera_rot_min_x
    camera_rot_max_x = args.camera_rot_max_x
    camera_rot_min_y = args.camera_rot_min_y
    camera_rot_max_y = args.camera_rot_max_y
    camera_rot_min_z = args.camera_rot_min_z
    camera_rot_max_z = args.camera_rot_max_z
    distant_light_min_rot_x = args.distant_light_min_rot_x
    distant_light_max_rot_x = args.distant_light_max_rot_x
    distant_light_min_rot_y = args.distant_light_min_rot_y
    distant_light_max_rot_y = args.distant_light_max_rot_y
    distant_light_min_rot_z = args.distant_light_min_rot_z
    distant_light_max_rot_z = args.distant_light_max_rot_z
    distant_light_min_intensity = args.distant_light_min_intensity
    distant_light_max_intensity = args.distant_light_max_intensity
    conveyor_belt_speed = args.conveyor_belt_speed
    min_x_pos_for_record_start = args.min_x_pos_for_record_start
    render_frequency = args.render_frequency
    physics_frequency = args.physics_frequency

    # Check range args for plausibility
    config.check_range_plausibility(object_init_min_x, object_init_max_x)
    config.check_range_plausibility(object_init_min_y, object_init_max_y)
    config.check_range_plausibility(object_init_min_z, object_init_max_z)
    config.check_range_plausibility(sphere_min_x, sphere_max_x)
    config.check_range_plausibility(sphere_min_y, sphere_max_y)
    config.check_range_plausibility(sphere_min_z, sphere_max_z)
    config.check_range_plausibility(sphere_min_intensity, sphere_max_intensity)
    config.check_range_plausibility(camera_pos_min_x, camera_pos_max_x)
    config.check_range_plausibility(camera_pos_min_y, camera_pos_max_y)
    config.check_range_plausibility(camera_pos_min_z, camera_pos_max_z)
    config.check_range_plausibility(camera_rot_min_x, camera_rot_max_x)
    config.check_range_plausibility(camera_rot_min_y, camera_rot_max_y)
    config.check_range_plausibility(camera_rot_min_z, camera_rot_max_z)
    config.check_range_plausibility(distant_light_min_rot_x, distant_light_max_rot_x)
    config.check_range_plausibility(distant_light_min_rot_y, distant_light_max_rot_y)
    config.check_range_plausibility(distant_light_min_rot_z, distant_light_max_rot_z)
    config.check_range_plausibility(distant_light_min_intensity, distant_light_max_intensity)

    # Check for existing config at out_path
    if os.path.exists(out_path):
        print(f"Config file at '{out_path}' already exists. Exiting...")
        exit(-1)

    # Test usd_dir
    if not os.path.isdir(usd_dir):
        print("USD directory does not exist. Exiting...")
        exit(-1)

    usd_models = io.get_usd_models(usd_dir)

    # Build final config
    data_generation_config = generate_config(
                    usd_models,
                    frame_height, frame_width,
                    sub_frames_per_frame,
                    num_random_materials,
                    probability_of_glass_material,
                    num_frames_per_scene,
                    num_objects_per_scene,
                    num_scenes_per_object,
                    num_sphere_lights,
                    sphere_min_intensity, sphere_max_intensity,
                    sphere_min_x, sphere_max_x,
                    sphere_min_y, sphere_max_y,
                    sphere_min_z, sphere_max_z,
                    object_init_min_y, object_init_max_x,
                    object_init_min_x, object_init_max_y,
                    object_init_min_z, object_init_max_z,
                    val_dataset_share,
                    camera_pos_min_x, camera_pos_max_x,
                    camera_pos_min_y, camera_pos_max_y,
                    camera_pos_min_z, camera_pos_max_z,
                    camera_rot_min_x, camera_rot_max_x,
                    camera_rot_min_y, camera_rot_max_y,
                    camera_rot_min_z, camera_rot_max_z,
                    distant_light_min_intensity, distant_light_max_intensity,
                    distant_light_min_rot_x, distant_light_max_rot_x,
                    distant_light_min_rot_y, distant_light_max_rot_y,
                    distant_light_min_rot_z, distant_light_max_rot_z,
                    conveyor_belt_speed,
                    min_x_pos_for_record_start,
                    render_frequency,
                    physics_frequency,
                    camera_intrinsics.build_camera_intrinsics_conf(args, frame_width, frame_height),
                    args)

    # Write config
    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data_generation_config, f, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])
