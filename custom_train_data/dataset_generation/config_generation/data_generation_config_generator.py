import sys
import json
import os
import argparse
import random

import numpy as np
from sklearn.model_selection import train_test_split


def get_usd_models(usd_dir: str) -> list[str]:
    """
    List all files with the '.usd' extension in the specified directory.

    :param usd_dir: The path to the directory to search in.
    :type usd_dir: str
    :return: A list of file names with the '.usd' extension found in the specified directory.
    :rtype: list
    """
    return [file for file in os.listdir(usd_dir) if file.endswith('.usd')]


def generate_camera_conf(frame_width: int, frame_height: int) -> dict:
    """
    Generates a configuration dictionary for a camera setup.

    This function creates a configuration with specified frame height and width, and sets a default pose for the camera.
    The default pose is set to look at the origin (0, 0, 0) with the x-axis oriented to the right.

    :param frame_width: The width of the frame in pixels.
    :type frame_height: int
    :param frame_height: The height of the frame in pixels.
    :type frame_width: int
    :return: A dictionary containing the camera configuration, including frame size and pose.
    :rtype: dict
    """

    return {
        "frame_height": frame_height,
        "frame_width": frame_width,
        "pose": {
            "position": [0, 0, 5], "orientation": [-90, -90, 0]  # Look at (0, 0, 0) with x-axis to the right
        }
    }


def generate_random_rgb_color() -> list[float]:
    """
    Generates a random RGB color.

    Each color component is a float between 0 and 1, representing the intensity of red, green, and blue.

    :return: A list containing three floats, each representing the red, green, and blue color components.
    :rtype: list[float]
    """

    return [random.uniform(0, 1) for _ in range(3)]


def generate_materials_conf(num_random_materials: int, probability_of_glass_material: float) -> list[dict]:
    """
    Generates a list of random material configurations.

    The function randomly decides if a material is glass or metallic based on the provided probability, and assigns a random RGB color to each material. For metallic materials, a random surface roughness is also assigned.

    :param num_random_materials: Number of random materials to generate.
    :type num_random_materials: int
    :param probability_of_glass_material: Probability that a material is glass.
    :type probability_of_glass_material: float
    :return: A list of dictionaries, each containing the configuration of a material.
    :rtype: list[dict]
    """

    mat_configs = []
    for i in range(num_random_materials):
        if random.uniform(0, 1) < probability_of_glass_material:
            # Material is glass
            mat_configs.append({
                "material_idx": i,
                "is_glass": True,
                "color": generate_random_rgb_color()
            })
        else:
            # Material is metallic
            mat_configs.append({
                "material_idx": i,
                "is_glass": False,
                "color": generate_random_rgb_color(),
                "surface_roughness": random.uniform(0, 1)
            })

    return mat_configs


def generate_obj_conf(num_random_materials: int, usd_model: str) -> dict:
    """
    Generates a configuration for an object using a specified USD model.

    The configuration includes a random position, orientation, material index, and a semantic class label derived from the USD model file name.

    :param num_random_materials: The number of available random materials.
    :type num_random_materials: int
    :param usd_model: The file name of the USD model.
    :type usd_model: str
    :return: A dictionary containing the object's configuration.
    :rtype: dict
    """

    return {
        "usd_model": usd_model,
        "pose": {
            "position": [random.uniform(-2, 2), random.uniform(-1, 1), random.uniform(0, 3)],
            # from (-2, -1, 0) to (2, 1, 3) (including)
            "orientation": [random.randint(-180, 180), random.randint(-180, 180), random.randint(-180, 180)]
            # all axes from (-180° to 180°) (including)
        },
        "material_idx": random.randint(0, num_random_materials - 1),
        "semantic_class_label": usd_model.removesuffix("_obj.usd").lower().replace(" ", "_")
    }


def generate_single_object_scene_confs(num_frames_per_object: int, num_objects_per_frame: int, usd_model: str,
                                       num_random_materials: int) -> list[dict]:
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
    :return: A list of dictionaries, each representing a scene configuration.
    :rtype: list[dict]
    """

    return [
        {
            "object_configs": [generate_obj_conf(num_random_materials, usd_model) for _ in range(num_objects_per_frame)]
        } for _ in range(num_frames_per_object)
    ]


def generate_multiple_object_scene_confs(num_frames_per_object: int, num_objects_per_frame: int, usd_models: list[str],
                                         num_random_materials: int) -> list[dict]:
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
    :return: A list of dictionaries, each representing a scene configuration.
    :rtype: list[dict]
    """

    num_scenes = len(usd_models) * num_frames_per_object
    scene_confs = []
    for _ in range(num_scenes):
        obj_configs = []
        for _ in range(num_objects_per_frame):
            usd_model = random.choice(usd_models)
            obj_configs.append(generate_obj_conf(num_random_materials, usd_model))

        scene_confs.append({"object_configs": obj_configs})

    return scene_confs


def generate_sphere_light_confs_for_one_frame(num_sphere_lights: int) -> list[dict]:
    """
    Generates configurations for sphere lights in a single frame.

    Each sphere light configuration includes a random position and color.

    :param num_sphere_lights: Number of sphere lights to generate.
    :type num_sphere_lights: int
    :return: A list containing the configurations of sphere lights.
    :rtype: list[dict]
    """

    return [
        {
            "position": [random.uniform(-3, 3), random.uniform(-2, 2), random.uniform(0, 4)],
            # from (-3, -2, 0) to (3, 2, 4) (including)]}
            "color": generate_random_rgb_color()
        }
        for _ in range(num_sphere_lights)]


def generate_scenes_conf(num_frames_per_object: int, num_objects_per_frame: int, usd_models: list[str],
                         num_random_materials: int, num_sphere_lights: int) -> list[list[dict]]:
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
    :return: A list of a list of dictionaries, each representing a scene configuration.
    :rtype: list[list[dict]]
    """

    # Generate single object scenes
    scene_configs = []
    for usd_model in usd_models:
        scene_configs.append(generate_single_object_scene_confs(num_frames_per_object, num_objects_per_frame, usd_model,
                                                                num_random_materials))

    # Generate multiple object scenes
    scene_configs.append(generate_multiple_object_scene_confs(num_frames_per_object, num_objects_per_frame, usd_models,
                                                              num_random_materials))

    for object_type_scene_configs in scene_configs:
        for scene_config in object_type_scene_configs:
            # Add randomized background / ground plane color
            scene_config["background_color"] = generate_random_rgb_color()

            # Add randomized dome light color
            scene_config["dome_light_color"] = generate_random_rgb_color()

            # Add randomized sphere lights
            scene_config["sphere_light_configs"] = generate_sphere_light_confs_for_one_frame(num_sphere_lights)

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
    parser.add_argument("--train_val_split", default=0.2, type=float,
                        help="Sets the train and validation split of the generated dataset. The default value of 0.2 means that 20% of the dataset are assigned to the validation dataset")

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
    val_dataset_share = args.train_val_split
    probability_of_glass_material = args.probability_of_glass_material
    if probability_of_glass_material < 0:
        probability_of_glass_material = 0
    if probability_of_glass_material > 1:
        probability_of_glass_material = 1

    # Check for existing config at out_path
    if os.path.exists(out_path):
        print(f"Config file at '{out_path}' already exists. Exiting...")
        exit(-1)

    # Test usd_dir
    if not os.path.isdir(usd_dir):
        print("USD directory does not exist. Exiting...")
        exit(-1)

    usd_models = get_usd_models(usd_dir)

    # Build final config
    data_generation_config = {
        "camera_config": generate_camera_conf(frame_width, frame_height),
        "sub_frames_per_frame": sub_frames_per_frame,
        "num_frames_per_object": num_frames_per_object,
        "materials": generate_materials_conf(num_random_materials, probability_of_glass_material),
        "scenes": generate_scenes_conf(num_frames_per_object, num_objects_per_frame, usd_models, num_random_materials,
                                       num_sphere_lights),
        "usd_models": usd_models,
        "generation_script_args": vars(args)
    }

    (train_scenes, val_scenes) = generate_train_val_splits(data_generation_config["scenes"], val_dataset_share)
    data_generation_config["train_scenes"] = train_scenes.tolist()
    data_generation_config["val_scenes"] = val_scenes.tolist()

    # Write config
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data_generation_config, f, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])
