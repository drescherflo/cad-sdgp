"""
Script to create a config file for the 6 DOF dataset generation.
The term scene is equivalent to one to be generated frame.
"""

import sys
import json
import os
import argparse
import random

import numpy as np
from sklearn.model_selection import train_test_split

from utils import io, config
from utils.scene_randomization import generate_random_vec_3, generate_random_rgb_color, generate_materials_conf, generate_sphere_light_confs_for_one_frame, usd_model_to_semantic_class_label


def generate_camera_conf(frame_width: int, frame_height: int, cam_distance_to_background: float) -> dict:
    """
    Generates a configuration dictionary for a camera setup.

    This function creates a configuration with specified frame height and width, and sets a default pose for the camera.
    The default pose is set to look at the origin (0, 0, 0) with the x-axis oriented to the right.

    :param frame_width: The width of the frame in pixels.
    :type frame_height: int
    :param frame_height: The height of the frame in pixels.
    :type frame_width: int
    :param cam_distance_to_background: Camera distance to the background plane
    :type cam_distance_to_background: float
    :return: A dictionary containing the camera configuration, including frame size and pose.
    :rtype: dict
    """

    return {
        "frame_height": frame_height,
        "frame_width": frame_width,
        "pose": {
            "position": [0, 0, cam_distance_to_background],
            "orientation": [-90, -90, 0]  # Look at (0, 0, 0) with x-axis to the right
        }
    }


def generate_obj_conf(num_random_materials: int, usd_model: str, min_x: float, max_x: float, min_y: float, max_y: float,
                      max_z: float) -> dict:
    """
    Generates a configuration for an object using a specified USD model.

    The configuration includes a random position, orientation, material index, and a semantic class label derived from the USD model file name.

    :param num_random_materials: The number of available random materials.
    :type num_random_materials: int
    :param usd_model: The file name of the USD model.
    :type usd_model: str
    :param min_x: Minimum x coordinate for an object.
    :type min_x: float
    :param max_x: Maximum x coordinate for an object.
    :type max_x: float
    :param min_y: Minimum y coordinate for an object.
    :type min_y: float
    :param max_y: Maximum y coordinate for an object.
    :type max_y: float
    :param max_z: Maximum z coordinate for an object.
    :type max_z: float
    :return: A dictionary containing the object's configuration.
    :rtype: dict
    """

    return {
        "usd_model": usd_model,
        "pose": {
            "position": generate_random_vec_3(min_x, max_x, min_y, max_y, 0, max_z),
            "orientation": generate_random_vec_3(-180, 180, -180, 180, -180, 180) # all axes from (-180° to 180°) (including)
        },
        "material_idx": random.randint(0, num_random_materials - 1),
        "semantic_class_label": usd_model_to_semantic_class_label(usd_model)
    }


def generate_single_object_scene_confs(num_frames_per_object: int, num_objects_per_frame: int, usd_model: str,
                                       num_random_materials: int, min_x: float, max_x: float, min_y: float,
                                       max_y: float, max_z: float) -> list[dict]:
    """
    Generates a list of scene configurations, each containing configurations for a single object.

    :param num_frames_per_object: Number of frames to generate for each object.
    :type num_frames_per_object: int
    :param num_objects_per_frame: Number of objects in each frame.
    :type num_objects_per_frame: int
    :param usd_model: The file name of the USD model to be used for each object.
    :type usd_model: str
    :param num_random_materials: The number of available random materials.
    :type num_random_materials: int
    :param min_x: Minimum x coordinate for an object.
    :type min_x: float
    :param max_x: Maximum x coordinate for an object.
    :type max_x: float
    :param min_y: Minimum y coordinate for an object.
    :type min_y: float
    :param max_y: Maximum y coordinate for an object.
    :type max_y: float
    :param max_z: Maximum z coordinate for an object.
    :type max_z: float
    :return: A list of dictionaries, each representing a scene configuration.
    :rtype: list[dict]
    """

    return [
        {
            "object_configs": [generate_obj_conf(num_random_materials, usd_model, min_x, max_x, min_y, max_y, max_z) for
                               _ in range(num_objects_per_frame)]
        } for _ in range(num_frames_per_object)
    ]


def generate_multiple_object_scene_confs(num_frames_per_object: int, num_objects_per_frame: int, usd_models: list[str],
                                         num_random_materials: int, min_x: float, max_x: float, min_y: float,
                                         max_y: float, max_z: float) -> list[dict]:
    """
    Generates a list of scene configurations with multiple objects.

    Each scene configuration contains a random selection of objects based on the provided USD models.

    :param num_frames_per_object: Number of frames to generate for each object type.
    :type num_frames_per_object: int
    :param num_objects_per_frame: Number of objects in each frame.
    :type num_objects_per_frame: int
    :param usd_models: A list of USD model file names to be used.
    :type usd_models: list[str]
    :param num_random_materials: The number of available random materials.
    :type num_random_materials: int
    :param min_x: Minimum x coordinate for an object.
    :type min_x: float
    :param max_x: Maximum x coordinate for an object.
    :type max_x: float
    :param min_y: Minimum y coordinate for an object.
    :type min_y: float
    :param max_y: Maximum y coordinate for an object.
    :type max_y: float
    :param max_z: Maximum z coordinate for an object.
    :type max_z: float
    :return: A list of dictionaries, each representing a scene configuration.
    :rtype: list[dict]
    """

    num_scenes = len(usd_models) * num_frames_per_object
    scene_confs = []
    for _ in range(num_scenes):
        obj_configs = []
        for _ in range(num_objects_per_frame):
            usd_model = random.choice(usd_models)
            obj_configs.append(generate_obj_conf(num_random_materials, usd_model, min_x, max_x, min_y, max_y, max_z))

        scene_confs.append({"object_configs": obj_configs})

    return scene_confs


def generate_scenes_conf(num_frames_per_object: int, num_objects_per_frame: int, usd_models: list[str],
                         num_random_materials: int, num_sphere_lights: int, sphere_min_intensity: float, sphere_max_intensity: float,
                         min_x: float, max_x: float, min_y: float,
                         max_y: float, max_z: float,
                         dome_light_min_intensity, dome_light_max_intensity) -> list[list[dict]]:
    """
    Generates configurations for a variety of scenes.

    This includes single and multiple object scenes, each with randomized background, dome light colors, and sphere light configurations.

    :param num_frames_per_object: Number of frames to generate for each object type.
    :type num_frames_per_object: int
    :param num_objects_per_frame: Number of objects in each frame.
    :type num_objects_per_frame: int
    :param usd_models: A list of USD model file names to be used.
    :type usd_models: list[str]
    :param num_random_materials: The number of available random materials.
    :type num_random_materials: int
    :param num_sphere_lights: Number of sphere lights in each scene.
    :type num_sphere_lights: int
    :param sphere_min_intensity: Minimum intensity for a sphere light
    :type sphere_min_intensity: int
    :param sphere_max_intensity: Maximum intensity for a sphere light
    :type sphere_min_intensity: int
    :param min_x: Minimum x coordinate for a light sphere or an object.
    :type min_x: float
    :param max_x: Maximum x coordinate for a light sphere or an object.
    :type max_x: float
    :param min_y: Minimum y coordinate for a light sphere or an object.
    :type min_y: float
    :param max_y: Maximum y coordinate for a light sphere or an object.
    :type max_y: float
    :param max_z: Maximum z coordinate for a light sphere or an object.
    :type max_z: float
    :param dome_light_min_intensity: Minimum intensity for a dome light
    :type dome_light_min_intensity: int
    :param dome_light_max_intensity: Maximum intensity for a dome light
    :type dome_light_max_intensity: int
    :return: A list of a list of dictionaries, each representing a scene configuration.
    :rtype: list[list[dict]]
    """

    # Generate single object scenes
    scene_configs = []
    for usd_model in usd_models:
        scene_configs.append(generate_single_object_scene_confs(num_frames_per_object, num_objects_per_frame, usd_model,
                                                                num_random_materials, min_x, max_x, min_y, max_y,
                                                                max_z))

    # Generate multiple object scenes
    scene_configs.append(generate_multiple_object_scene_confs(num_frames_per_object, num_objects_per_frame, usd_models,
                                                              num_random_materials, min_x, max_x, min_y, max_y, max_z))

    for object_type_scene_configs in scene_configs:
        for scene_config in object_type_scene_configs:
            # Add randomized background / ground plane color
            scene_config["background_color"] = generate_random_rgb_color()

            # Add randomized dome light color
            scene_config["dome_light_configs"] = {"color": generate_random_rgb_color(), "intensity": np.random.uniform(dome_light_min_intensity, dome_light_max_intensity)}

            # Add randomized sphere lights
            scene_config["sphere_light_configs"] = generate_sphere_light_confs_for_one_frame(num_sphere_lights, min_x,
                                                                                             max_x, min_y, max_y, 0, max_z, sphere_min_intensity, sphere_max_intensity)

    return scene_configs


def generate_train_val_splits(scenes: list[list[dict]], val_share: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates training and validation splits from a list of scenes.

    :param scenes: A nested list of dictionaries, each representing a scene.
    :param val_share: The proportion of the dataset to include in the validation split.
    :type scenes: list[list[dict]]
    :type val_share: float
    :return: A tuple containing two numpy arrays with scene numbers, one for training and one for validation.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    num_scenes = np.sum(np.fromiter((len(s) for s in scenes), int))
    scene_indices = np.arange(num_scenes)
    return train_test_split(scene_indices, test_size=val_share)


def main(argv: list[str]) -> None:
    """
    Main function to handle command-line arguments and execute the configuration generation process.

    :param argv: List of command-line arguments.
    :type argv: list[str]
    """

    parser = argparse.ArgumentParser(description="Generates a configuration for the training data generation script")
    parser.add_argument("--usd_dir", help="Directory containing the converted CAD models as USD files", required=True)
    parser.add_argument("--out_path", default="config.json", help="Output path for the generated configuration")
    parser.add_argument("--frame_width", default=480, type=int, help="Width of the generated frames")
    parser.add_argument("--frame_height", default=360, type=int, help="Width of the generated frames")
    parser.add_argument("--sub_frames_per_frame", default=32, type=int,
                        help="Number of frames to render before saving the frame to avoid artifacts after fast object movement")
    parser.add_argument("--num_random_materials", default=100, type=int,
                        help="Defines the number of random materials to generate")
    parser.add_argument("--probability_of_glass_material", default=0.5, type=float,
                        help="Defines the probability of generating glass objects")
    parser.add_argument("--num_frames_per_object", default=1000, type=int,
                        help="Number of frames to generate per object. After the specified amount of frames was generated per object in the USD directory, n * num_frames_per_object will be generated with all objects in the scene. (n is num_frames_per_object times objects in the USD directory)")
    parser.add_argument("--num_objects_per_frame", default=20, type=int,
                        help="Specifies the number of objects in the scene")
    parser.add_argument("--num_sphere_lights", default=5, type=int,
                        help="Specifies the number of sphere lights with random light color in the scene")
    parser.add_argument("--sphere_min_intensity", default=5000, type=float,
                        help="The minimum light intensity of a sphere light")
    parser.add_argument("--sphere_max_intensity", default=50000, type=float,
                        help="The maximum light intensity of a sphere light")
    parser.add_argument("--train_val_split", default=0.2, type=float,
                        help="Sets the train and validation split of the generated dataset. The default value of 0.2 means that 20% of the dataset are assigned to the validation dataset")
    parser.add_argument("--min_x", default=-2, type=float, help="The minimum x coordinate of the object in the scene")
    parser.add_argument("--max_x", default=2, type=float, help="The maximum x coordinate of the object in the scene")
    parser.add_argument("--min_y", default=-1, type=float, help="The minimum y coordinate of the object in the scene")
    parser.add_argument("--max_y", default=1, type=float, help="The maximum y coordinate of the object in the scene")
    parser.add_argument("--cam_distance_to_background", default=5, type=float,
                        help="Defines the distance between the camera and the background plane")
    parser.add_argument("--dome_light_min_intensity", default=5000, type=float,
                        help="The minimum light intensity of the dome light")
    parser.add_argument("--dome_light_max_intensity", default=50000, type=float,
                        help="The maximum light intensity of the dome light")

    # Parse args
    args = parser.parse_args(argv)

    # Assign variables
    usd_dir = args.usd_dir
    out_path = args.out_path
    frame_width = args.frame_width
    frame_height = args.frame_height
    sub_frames_per_frame = args.sub_frames_per_frame
    num_frames_per_object = args.num_frames_per_object
    num_objects_per_frame = args.num_objects_per_frame
    num_random_materials = args.num_random_materials
    num_sphere_lights = args.num_sphere_lights
    sphere_min_intensity = args.sphere_min_intensity
    sphere_max_intensity = args.sphere_max_intensity
    val_dataset_share = args.train_val_split
    probability_of_glass_material = args.probability_of_glass_material
    if probability_of_glass_material < 0:
        probability_of_glass_material = 0
    if probability_of_glass_material > 1:
        probability_of_glass_material = 1
    min_x = args.min_x
    max_x = args.max_x
    min_y = args.min_y
    max_y = args.max_y
    cam_distance_to_background = args.cam_distance_to_background
    dome_light_min_intensity = args.dome_light_min_intensity
    dome_light_max_intensity = args.dome_light_max_intensity

    # Check for plausibility
    if cam_distance_to_background <= 0:
        print("cam_distance_to_background must be greater than 0. Exiting...")
        exit(-1)

    config.check_range_plausibility(min_x, max_x)
    config.check_range_plausibility(min_y, max_y)
    config.check_range_plausibility(sphere_min_intensity, sphere_max_intensity)
    config.check_range_plausibility(dome_light_min_intensity, dome_light_max_intensity)

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
    data_generation_config = {
        "camera_config": generate_camera_conf(frame_width, frame_height, cam_distance_to_background),
        "sub_frames_per_frame": sub_frames_per_frame,
        "num_frames_per_object": num_frames_per_object,
        "materials": generate_materials_conf(num_random_materials, probability_of_glass_material),
        "scenes": generate_scenes_conf(num_frames_per_object, num_objects_per_frame, usd_models, num_random_materials,
                                       num_sphere_lights, sphere_min_intensity, sphere_max_intensity, min_x, max_x, min_y, max_y, cam_distance_to_background, dome_light_min_intensity, dome_light_max_intensity),
        "usd_models": usd_models,
        "generation_script_args": vars(args)
    }

    (train_scenes, val_scenes) = generate_train_val_splits(data_generation_config["scenes"], val_dataset_share)
    data_generation_config["train_scenes"] = train_scenes.tolist()
    data_generation_config["val_scenes"] = val_scenes.tolist()

    # Write config
    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data_generation_config, f, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])
