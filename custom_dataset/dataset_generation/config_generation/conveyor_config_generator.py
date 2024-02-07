import inspect
import sys
import json
import os
import argparse
import random
from typing import Any

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

    The function randomly decides if a material is glass or metallic based on the provided probability, and assigns a random RGB color to each material.
    For metallic materials, a random surface roughness is also assigned.

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


def generate_sphere_light_confs_for_one_frame(num_sphere_lights: int, min_x: float, max_x: float, min_y: float,
                                              max_y: float, min_z: float, max_z: float) -> list[dict]:
    """
    Generates configurations for sphere lights in a single frame.

    Each sphere light configuration includes a random position and color.

    :param num_sphere_lights: Number of sphere lights to generate.
    :type num_sphere_lights: int
    :param min_x: Minimum x coordinate for a light sphere.
    :type min_x: float
    :param max_x: Maximum x coordinate for a light sphere.
    :type max_x: float
    :param min_y: Minimum y coordinate for a light sphere.
    :type min_y: float
    :param max_y: Maximum y coordinate for a light sphere.
    :type max_y: float
    :param min_z: Minimum z coordinate for a light sphere.
    :type min_z: float
    :param max_z: Maximum z coordinate for a light sphere.
    :type max_z: float
    :return: A list containing the configurations of sphere lights.
    :rtype: list[dict]
    """

    return [
        {
            "position": generate_random_vec_3(min_x, max_x, min_y, max_y, min_z, max_z),
            "color": generate_random_rgb_color()
        }
        for _ in range(num_sphere_lights)]


def generate_random_vec_3(min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float):
    return [random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.uniform(min_z, max_z)]


def generate_scenes_conf(usd_models: list[str], num_random_materials: int,
                         num_frames_per_scene: int, num_objects_per_scene: int, num_scenes_per_object: int,
                         num_sphere_lights: int,
                         object_init_min_x: float, object_init_max_x: float,
                         object_init_min_y: float, object_init_max_y: float,
                         object_init_min_z: float, object_init_max_z: float,
                         sphere_min_x: float, sphere_max_x: float,
                         sphere_min_y: float, sphere_max_y: float,
                         sphere_min_z: float, sphere_max_z: float,
                         direct_light_min_x: float, direct_light_max_x: float,
                         direct_light_min_y: float, direct_light_max_y: float,
                         direct_light_min_z: float, direct_light_max_z: float,
                         camera_pos_min_x: float, camera_pos_max_x: float,
                         camera_pos_min_y: float, camera_pos_max_y: float,
                         camera_pos_min_z: float, camera_pos_max_z: float,
                         camera_rot_min_x: float, camera_rot_max_x: float,
                         camera_rot_min_y: float, camera_rot_max_y: float,
                         camera_rot_min_z: float, camera_rot_max_z: float
                         ) -> list:

    # TODO-List:
    # - ok Configurations-Liste für Camera-Posen pro Szene
    # - ok Configurations-Liste für Materialien pro Szene
    # - ok Configurations-Liste für Sphere-Lights Position und Farbe pro Szene
    # - ok Configurations-Liste für Direct-Light Position und Farbe pro Szene(Orientierung wird in Simulation auf look-at: (0, 0, 0) gesetzt)
    # - ok Configurations-Liste für Ground-Plane Farbe pro Szene
    # - ok Configurations-Liste für Init-Pose der Objekte pro Szene

    # Generate single object scenes
    scene_configs = []
    for usd_model in usd_models:
        for _ in range(num_scenes_per_object):
            scene_config = {}
            scene_config["objects"] = [{
                "model": usd_model,
                "object_init_pose": {
                    "position": generate_random_vec_3(object_init_min_x, object_init_max_x, object_init_min_y, object_init_max_y, object_init_min_z, object_init_max_z),
                    "rotation": generate_random_vec_3(-180, 180, -180, 180, -180, 180) # all axes from (-180° to 180°) (including)
                },
                "semantic_class_label": usd_model.removesuffix("_obj.usd").lower().replace(" ", "_")
            } for _ in range(num_objects_per_scene)]
            scene_config["per_frame_config"] = {
                "camera_poses": [{
                    "position": generate_random_vec_3(camera_pos_min_x, camera_pos_max_x, camera_pos_min_y, camera_pos_max_y, camera_pos_min_z, camera_pos_max_z),
                    "rotation": generate_random_vec_3(camera_rot_min_x, camera_rot_max_x, camera_rot_min_y, camera_rot_max_y, camera_rot_min_z, camera_rot_max_z)
                } for _ in range(num_objects_per_scene)],
                "sphere_light_configs": [generate_sphere_light_confs_for_one_frame(num_sphere_lights, sphere_min_x, sphere_max_x, sphere_min_y, sphere_max_y, sphere_min_z, sphere_max_z) for _ in range(num_frames_per_scene)],
                "direct_light_configs": [{
                    "position": generate_random_vec_3(direct_light_min_x, direct_light_max_x, direct_light_min_y, direct_light_max_y, direct_light_min_z, direct_light_max_z),
                    "color": generate_random_rgb_color()
                } for _ in range(num_frames_per_scene)],
                "ground_plane_colors": [generate_random_rgb_color() for _ in range(num_frames_per_scene)],
                "object_material_assignments": [
                    [{
                        "object_idx": i,
                        "material_idx": random.randint(0, num_random_materials - 1)
                    } for i in range(num_objects_per_scene)]
                    for _ in range(num_frames_per_scene)]
            }

            scene_configs.append(scene_config)

    # Generate multiple object scenes
    for _ in range(len(usd_models)):
        for _ in range(num_scenes_per_object):
            scene_config = {"objects": []}
            for _ in range(num_objects_per_scene):
                usd_model = random.choice(usd_models)
                scene_config["objects"].append({
                    "model": usd_model,
                    "object_init_pose": {
                        "position": generate_random_vec_3(object_init_min_x, object_init_max_x, object_init_min_y,
                                                          object_init_max_y, object_init_min_z, object_init_max_z),
                        "rotation": generate_random_vec_3(-180, 180, -180, 180, -180, 180)
                        # all axes from (-180° to 180°) (including)
                    },
                    "semantic_class_label": usd_model.removesuffix("_obj.usd").lower().replace(" ", "_")
                })
            scene_config["per_frame_config"] = {
                "camera_poses": [{
                    "position": generate_random_vec_3(camera_pos_min_x, camera_pos_max_x, camera_pos_min_y,
                                                      camera_pos_max_y, camera_pos_min_z, camera_pos_max_z),
                    "rotation": generate_random_vec_3(camera_rot_min_x, camera_rot_max_x, camera_rot_min_y,
                                                      camera_rot_max_y, camera_rot_min_z, camera_rot_max_z)
                } for _ in range(num_objects_per_scene)],
                "sphere_light_configs": [
                    generate_sphere_light_confs_for_one_frame(num_sphere_lights, sphere_min_x, sphere_max_x,
                                                              sphere_min_y, sphere_max_y, sphere_min_z, sphere_max_z)
                    for _ in range(num_frames_per_scene)],
                "direct_light_configs": [{
                    "position": generate_random_vec_3(direct_light_min_x, direct_light_max_x, direct_light_min_y,
                                                      direct_light_max_y, direct_light_min_z, direct_light_max_z),
                    "color": generate_random_rgb_color()
                } for _ in range(num_frames_per_scene)],
                "ground_plane_colors": [generate_random_rgb_color() for _ in range(num_frames_per_scene)],
                "object_material_assignments": [
                    [{
                        "object_idx": i,
                        "material_idx": random.randint(0, num_random_materials - 1)
                    } for i in range(num_objects_per_scene)]
                    for _ in range(num_frames_per_scene)]
            }

            scene_configs.append(scene_config)

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


def check_range_plausibility(min_val: float, max_val: float) -> None:
    if min_val > max_val:
        def get_var_name(var: Any) -> str:
            # Code from https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string (02.02.2024)
            callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

        raise ValueError(f"{get_var_name(min_val)} must be less or equal than {get_var_name(max_val)}")


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
    parser.add_argument("--num_frames_per_scene", default=1000, type=int,
                        help="Number of frames to record per simulation run / simulation scene")
    parser.add_argument("--num_objects_per_scene", default=20, type=int,
                        help="Specifies the number of objects in the scene")
    parser.add_argument("--num_scenes_per_object", default=5, type=int,
                        help="Specifies the number of scenes to generate for each object in the USD directory. Additionally n * num_scenes_per_object scenes will be generated with all objects in the scene. (n is num_frames_per_object times objects in the USD directory)")
    parser.add_argument("--num_sphere_lights", default=5, type=int,
                        help="Specifies the number of sphere lights with random light color in the scene")
    parser.add_argument("--train_val_split", default=0.2, type=float,
                        help="Sets the train and validation split of the generated dataset. The default value of 0.2 means that 20% of the dataset are assigned to the validation dataset")
    parser.add_argument("--object_init_min_x", default=-2, type=float,
                        help="The minimum initial x coordinate of the object in the scene")  # TODO adjust default values
    parser.add_argument("--object_init_max_x", default=2, type=float,
                        help="The maximum initial x coordinate of the object in the scene")
    parser.add_argument("--object_init_min_y", default=-1, type=float,
                        help="The minimum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_max_y", default=1, type=float,
                        help="The maximum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_min_z", default=-1, type=float,
                        help="The minimum initial z coordinate of the object in the scene")
    parser.add_argument("--object_init_max_z", default=1, type=float,
                        help="The maximum initial z coordinate of the object in the scene")
    parser.add_argument("--sphere_min_x", default=-2, type=float,
                        help="The minimum x coordinate of light spheres in the scene")
    parser.add_argument("--sphere_max_x", default=2, type=float,
                        help="The maximum x coordinate of light spheres in the scene")
    parser.add_argument("--sphere_min_y", default=-1, type=float,
                        help="The minimum y coordinate of light spheres in the scene")
    parser.add_argument("--sphere_max_y", default=1, type=float,
                        help="The maximum y coordinate of light spheres in the scene")
    parser.add_argument("--sphere_min_z", default=-1, type=float,
                        help="The minimum z coordinate of light spheres in the scene")
    parser.add_argument("--sphere_max_z", default=1, type=float,
                        help="The maximum z coordinate of light spheres in the scene")
    parser.add_argument("--camera_pos_min_x", default=-2, type=float,
                        help="The minimum x coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_max_x", default=2, type=float,
                        help="The maximum x coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_min_y", default=-1, type=float,
                        help="The minimum y coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_max_y", default=1, type=float,
                        help="The maximum y coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_min_z", default=-1, type=float,
                        help="The minimum z coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_max_z", default=1, type=float,
                        help="The maximum z coordinate of the camera in the scene")
    parser.add_argument("--camera_rot_min_x", default=-2, type=float,
                        help="The minimum rotation of the camera around the x axis in degrees")
    parser.add_argument("--camera_rot_max_x", default=2, type=float,
                        help="The maximum rotation of the camera around the x axis in degrees")
    parser.add_argument("--camera_rot_min_y", default=-1, type=float,
                        help="The minimum rotation of the camera around the y axis in degrees")
    parser.add_argument("--camera_rot_max_y", default=1, type=float,
                        help="The maximum rotation of the camera around the y axis in degrees")
    parser.add_argument("--camera_rot_min_z", default=-1, type=float,
                        help="The minimum rotation of the camera around the z axis in degrees")
    parser.add_argument("--camera_rot_max_z", default=1, type=float,
                        help="The maximum rotation of the camera around the z axis in degrees")
    parser.add_argument("--direct_light_min_x", default=-2, type=float,
                        help="The minimum x coordinate of the direct light in the scene")
    parser.add_argument("--direct_light_max_x", default=2, type=float,
                        help="The maximum x coordinate of the direct light in the scene")
    parser.add_argument("--direct_light_min_y", default=-1, type=float,
                        help="The minimum y coordinate of the direct light in the scene")
    parser.add_argument("--direct_light_max_y", default=1, type=float,
                        help="The maximum y coordinate of the direct light in the scene")
    parser.add_argument("--direct_light_min_z", default=-1, type=float,
                        help="The minimum z coordinate of the direct light in the scene")
    parser.add_argument("--direct_light_max_z", default=1, type=float,
                        help="The maximum z coordinate of the direct light in the scene")


    # TODO: evaulation-data-flag



    # TODO (falls möglich): Bandfarbe randomisieren

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
    num_scenes_per_object = args.num_scnenes_per_object
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
    direct_light_min_x = args.direct_light_min_x
    direct_light_max_x = args.direct_light_max_x
    direct_light_min_y = args.direct_light_min_y
    direct_light_max_y = args.direct_light_max_y
    direct_light_min_z = args.direct_light_min_z
    direct_light_max_z = args.direct_light_max_z

    # Check range args for plausibility
    check_range_plausibility(object_init_min_x, object_init_max_x)
    check_range_plausibility(object_init_min_y, object_init_max_y)
    check_range_plausibility(object_init_min_z, object_init_max_z)
    check_range_plausibility(sphere_min_x, sphere_max_x)
    check_range_plausibility(sphere_min_y, sphere_max_y)
    check_range_plausibility(sphere_min_z, sphere_max_z)
    check_range_plausibility(camera_pos_min_x, camera_pos_max_x)
    check_range_plausibility(camera_pos_min_y, camera_pos_max_y)
    check_range_plausibility(camera_pos_min_z, camera_pos_max_z)
    check_range_plausibility(camera_rot_min_x, camera_rot_max_x)
    check_range_plausibility(camera_rot_min_y, camera_rot_max_y)
    check_range_plausibility(camera_rot_min_z, camera_rot_max_z)
    check_range_plausibility(direct_light_min_x, direct_light_max_x)
    check_range_plausibility(direct_light_min_y, direct_light_max_y)
    check_range_plausibility(direct_light_min_z, direct_light_max_z)

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
        "camera_frame_config": {"frame_height": frame_height, "frame_width": frame_width},
        "sub_frames_per_frame": sub_frames_per_frame,
        "materials": generate_materials_conf(num_random_materials, probability_of_glass_material),
        "scenes": generate_scenes_conf(usd_models, num_random_materials,
                         num_frames_per_scene, num_objects_per_scene, num_scenes_per_object,
                         num_sphere_lights,
                         object_init_min_x, object_init_max_x,
                         object_init_min_y, object_init_max_y,
                         object_init_min_z, object_init_max_z,
                         sphere_min_x, sphere_max_x,
                         sphere_min_y, sphere_max_y,
                         sphere_min_z, sphere_max_z,
                         direct_light_min_x, direct_light_max_x,
                         direct_light_min_y, direct_light_max_y,
                         direct_light_min_z, direct_light_max_z,
                         camera_pos_min_x, camera_pos_max_x,
                         camera_pos_min_y, camera_pos_max_y,
                         camera_pos_min_z, camera_pos_max_z,
                         camera_rot_min_x, camera_rot_max_x,
                         camera_rot_min_y, camera_rot_max_y,
                         camera_rot_min_z, camera_rot_max_z),
        "usd_models": usd_models,
        "num_frames_per_scene": num_frames_per_scene,
        "num_scenes_per_object": num_scenes_per_object,
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
