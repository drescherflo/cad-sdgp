"""
Script to create a config file for the conveyor dataset generation.
The term scene means one data generation run, before the simulator reset.
In one scene multiple frames are generated.
"""

import sys
import json
import os
import argparse
import random

import numpy as np
from sklearn.model_selection import train_test_split

from utils import io, config
from utils.scene_randomization import generate_random_vec_3, generate_random_rgb_color, generate_materials_conf, generate_sphere_light_confs_for_one_frame


def generate_object_config(usd_model: str, object_init_min_x: float, object_init_max_x: float,
                         object_init_min_y: float, object_init_max_y: float,
                         object_init_min_z: float, object_init_max_z: float) -> dict:
    """
    Generates a configuration dictionary for an object to be used in a conveyor dataset scene.
    The configuration includes the USD model reference and the initial pose of the object,
    with a randomly generated position within specified ranges.

    :param usd_model: The USD model path or identifier for the object.
    :type usd_model: str
    :param object_init_min_x: Minimum x-coordinate for the initial position of the object.
    :type object_init_min_x: float
    :param object_init_max_x: Maximum x-coordinate for the initial position of the object.
    :type object_init_max_x: float
    :param object_init_min_y: Minimum y-coordinate for the initial position of the object.
    :type object_init_min_y: float
    :param object_init_max_y: Maximum y-coordinate for the initial position of the object.
    :type object_init_max_y: float
    :param object_init_min_z: Minimum z-coordinate for the initial position of the object.
    :type object_init_min_z: float
    :param object_init_max_z: Maximum z-coordinate for the initial position of the object.
    :type object_init_max_z: float
    :return: A dictionary containing the object's USD model path and initial pose.
    :rtype: dict
    """

    return {
                "usd_model": usd_model,
                "object_init_pose": {
                    "position": generate_random_vec_3(object_init_min_x, object_init_max_x, object_init_min_y, object_init_max_y, object_init_min_z, object_init_max_z),
                    "rotation": generate_random_vec_3(-180, 180, -180, 180, -180, 180) # all axes from (-180° to 180°) (including)
                },
                "semantic_class_label": usd_model.removesuffix("_obj.usd").lower().replace(" ", "_")
            }


def generate_per_frame_config(num_random_materials: int,
                         num_frames_per_scene: int, num_objects_per_scene: int,
                         num_sphere_lights: int,
                         sphere_min_x: float, sphere_max_x: float,
                         sphere_min_y: float, sphere_max_y: float,
                         sphere_min_z: float, sphere_max_z: float,
                         sphere_min_intensity: float, sphere_max_intensity: float,
                         distant_light_min_rot_x: float, distant_light_max_rot_x: float,
                         distant_light_min_rot_y: float, distant_light_max_rot_y: float,
                         distant_light_min_rot_z: float, distant_light_max_rot_z: float,
                         camera_pos_min_x: float, camera_pos_max_x: float,
                         camera_pos_min_y: float, camera_pos_max_y: float,
                         camera_pos_min_z: float, camera_pos_max_z: float,
                         camera_rot_min_x: float, camera_rot_max_x: float,
                         camera_rot_min_y: float, camera_rot_max_y: float,
                         camera_rot_min_z: float, camera_rot_max_z: float,
                         distant_light_min_intensity: float, distant_light_max_intensity: float) -> dict:
    """
    Generates a configuration dictionary for each frame in a scene, including camera poses,
    lighting configurations (sphere lights and distant lights), and material assignments for objects.

    :param num_random_materials: Number of random materials to choose from.
    :type num_random_materials: int
    :param num_frames_per_scene: Number of frames in each scene.
    :type num_frames_per_scene: int
    :param num_objects_per_scene: Number of objects in each scene.
    :type num_objects_per_scene: int
    :param num_sphere_lights: Number of sphere lights per frame.
    :type num_sphere_lights: int
    :param sphere_min_x: Minimum x-coordinate for sphere light positions.
    :type sphere_min_x: float
    :param sphere_max_x: Maximum x-coordinate for sphere light positions.
    :type sphere_max_x: float
    :param sphere_min_y: Minimum y-coordinate for sphere light positions.
    :type sphere_min_y: float
    :param sphere_max_y: Maximum y-coordinate for sphere light positions.
    :type sphere_max_y: float
    :param sphere_min_z: Minimum z-coordinate for sphere light positions.
    :type sphere_min_z: float
    :param sphere_max_z: Maximum z-coordinate for sphere light positions.
    :type sphere_max_z: float
    :param sphere_min_intensity: Minimum intensity for sphere lights.
    :type sphere_min_intensity: float
    :param sphere_max_intensity: Maximum intensity for sphere lights.
    :type sphere_max_intensity: float
    :param distant_light_min_rot_x: Minimum rotation angle for distant lights in the x-axis.
    :type distant_light_min_rot_x: float
    :param distant_light_max_rot_x: Maximum rotation angle for distant lights in the x-axis.
    :type distant_light_max_rot_x: float
    :param distant_light_min_rot_y: Minimum rotation angle for distant lights in the y-axis.
    :type distant_light_min_rot_y: float
    :param distant_light_max_rot_y: Maximum rotation angle for distant lights in the y-axis.
    :type distant_light_max_rot_y: float
    :param distant_light_min_rot_z: Minimum rotation angle for distant lights in the z-axis.
    :type distant_light_min_rot_z: float
    :param distant_light_max_rot_z: Maximum rotation angle for distant lights in the z-axis.
    :type distant_light_max_rot_z: float
    :param camera_pos_min_x: Minimum x-coordinate for camera positions.
    :type camera_pos_min_x: float
    :param camera_pos_max_x: Maximum x-coordinate for camera positions.
    :type camera_pos_max_x: float
    :param camera_pos_min_y: Minimum y-coordinate for camera positions.
    :type camera_pos_min_y: float
    :param camera_pos_max_y: Maximum y-coordinate for camera positions.
    :type camera_pos_max_y: float
    :param camera_pos_min_z: Minimum z-coordinate for camera positions.
    :type camera_pos_min_z: float
    :param camera_pos_max_z: Maximum z-coordinate for camera positions.
    :type camera_pos_max_z: float
    :param camera_rot_min_x: Minimum x-angle for camera rotation.
    :type camera_rot_min_x: float
    :param camera_rot_max_x: Maximum x-angle for camera rotation.
    :type camera_rot_max_x: float
    :param camera_rot_min_y: Minimum y-angle for camera rotation.
    :type camera_rot_min_y: float
    :param camera_rot_max_y: Maximum y-angle for camera rotation.
    :type camera_rot_max_y: float
    :param camera_rot_min_z: Minimum z-angle for camera rotation.
    :type camera_rot_min_z: float
    :param camera_rot_max_z: Maximum z-angle for camera rotation.
    :type camera_rot_max_z: float
    :param distant_light_min_intensity: Minimum intensity for distant lights.
    :type distant_light_min_intensity: float
    :param distant_light_max_intensity: Maximum intensity for distant lights.
    :type distant_light_max_intensity: float
    :return: Dictionary with configuration for per frame settings including camera poses, lighting, and material assignments.
    :rtype: dict
    """

    return {
        "camera_poses": [{
            "position": generate_random_vec_3(camera_pos_min_x, camera_pos_max_x, camera_pos_min_y, camera_pos_max_y,
                                              camera_pos_min_z, camera_pos_max_z),
            "rotation": generate_random_vec_3(camera_rot_min_x, camera_rot_max_x, camera_rot_min_y, camera_rot_max_y,
                                              camera_rot_min_z, camera_rot_max_z)
        } for _ in range(num_frames_per_scene)],
        "sphere_light_configs": [
            generate_sphere_light_confs_for_one_frame(num_sphere_lights, sphere_min_x, sphere_max_x, sphere_min_y,
                                                      sphere_max_y, sphere_min_z, sphere_max_z, sphere_min_intensity,
                                                      sphere_max_intensity) for _ in range(num_frames_per_scene)],
        "distant_light_configs": [{
            "rotation": generate_random_vec_3(distant_light_min_rot_x, distant_light_max_rot_x, distant_light_min_rot_y,
                                              distant_light_max_rot_y, distant_light_min_rot_z, distant_light_max_rot_z),
            "color": generate_random_rgb_color(),
            "intensity": random.uniform(distant_light_min_intensity, distant_light_max_intensity)
        } for _ in range(num_frames_per_scene)],
        "ground_plane_colors": [generate_random_rgb_color() for _ in range(num_frames_per_scene)],
        "object_material_assignments": [
            [{
                "material_idx": random.randint(0, num_random_materials - 1)
            } for _ in range(num_objects_per_scene)]
            for _ in range(num_frames_per_scene)],
        "conveyor_belt_colors": [generate_random_rgb_color() for _ in range(num_frames_per_scene)],
        "conveyor_frame_colors": [generate_random_rgb_color() for _ in range(num_frames_per_scene)]
    }


def generate_scenes_conf(usd_models: list[str], num_random_materials: int,
                         num_frames_per_scene: int, num_objects_per_scene: int, num_scenes_per_object: int,
                         num_sphere_lights: int,
                         object_init_min_x: float, object_init_max_x: float,
                         object_init_min_y: float, object_init_max_y: float,
                         object_init_min_z: float, object_init_max_z: float,
                         sphere_min_x: float, sphere_max_x: float,
                         sphere_min_y: float, sphere_max_y: float,
                         sphere_min_z: float, sphere_max_z: float,
                         sphere_min_intensity: float, sphere_max_intensity: float,
                         distant_light_min_rot_x: float, distant_light_max_rot_x: float,
                         distant_light_min_rot_y: float, distant_light_max_rot_y: float,
                         distant_light_min_rot_z: float, distant_light_max_rot_z: float,
                         camera_pos_min_x: float, camera_pos_max_x: float,
                         camera_pos_min_y: float, camera_pos_max_y: float,
                         camera_pos_min_z: float, camera_pos_max_z: float,
                         camera_rot_min_x: float, camera_rot_max_x: float,
                         camera_rot_min_y: float, camera_rot_max_y: float,
                         camera_rot_min_z: float, camera_rot_max_z: float,
                         distant_light_min_intensity: float, distant_light_max_intensity: float
                         ) -> list:
    """
    Generates a list of scene configurations, each containing object configurations and per-frame configurations
    for lighting, camera positions, and materials based on input parameters.

    :param usd_models: List of USD model paths or identifiers for the objects.
    :type usd_models: list[str]
    :param num_random_materials: Number of random materials to choose from.
    :type num_random_materials: int
    :param num_frames_per_scene: Number of frames in each scene.
    :type num_frames_per_scene: int
    :param num_objects_per_scene: Number of objects in each scene.
    :type num_objects_per_scene: int
    :param num_scenes_per_object: Number of scenes to generate per object.
    :type num_scenes_per_object: int
    :param num_sphere_lights: Number of sphere lights per frame.
    :type num_sphere_lights: int
    :param object_init_min_x: Minimum x-coordinate for object initial positions.
    :type object_init_min_x: float
    :param object_init_max_x: Maximum x-coordinate for object initial positions.
    :type object_init_max_x: float
    :param object_init_min_y: Minimum y-coordinate for object initial positions.
    :type object_init_min_y: float
    :param object_init_max_y: Maximum y-coordinate for object initial positions.
    :type object_init_max_y: float
    :param object_init_min_z: Minimum z-coordinate for object initial positions.
    :type object_init_min_z: float
    :param object_init_max_z: Maximum z-coordinate for object initial positions.
    :type object_init_max_z: float
    :param sphere_min_x: Minimum x-coordinate for sphere light positions.
    :type sphere_min_x: float
    :param sphere_max_x: Maximum x-coordinate for sphere light positions.
    :type sphere_max_x: float
    :param sphere_min_y: Minimum y-coordinate for sphere light positions.
    :type sphere_min_y: float
    :param sphere_max_y: Maximum y-coordinate for sphere light positions.
    :type sphere_max_y: float
    :param sphere_min_z: Minimum z-coordinate for sphere light positions.
    :type sphere_min_z: float
    :param sphere_max_z: Maximum z-coordinate for sphere light positions.
    :type sphere_max_z: float
    :param sphere_min_intensity: Minimum intensity for sphere lights.
    :type sphere_min_intensity: float
    :param sphere_max_intensity: Maximum intensity for sphere lights.
    :type sphere_max_intensity: float
    :param distant_light_min_rot_x: Minimum rotation angle for distant lights in the x-axis.
    :type distant_light_min_rot_x: float
    :param distant_light_max_rot_x: Maximum rotation angle for distant lights in the x-axis.
    :type distant_light_max_rot_x: float
    :param distant_light_min_rot_y: Minimum rotation angle for distant lights in the y-axis.
    :type distant_light_min_rot_y: float
    :param distant_light_max_rot_y: Maximum rotation angle for distant lights in the y-axis.
    :type distant_light_max_rot_y: float
    :param distant_light_min_rot_z: Minimum rotation angle for distant lights in the z-axis.
    :type distant_light_min_rot_z: float
    :param distant_light_max_rot_z: Maximum rotation angle for distant lights in the z-axis.
    :type distant_light_max_rot_z: float
    :param camera_pos_min_x: Minimum x-coordinate for camera positions.
    :type camera_pos_min_x: float
    :param camera_pos_max_x: Maximum x-coordinate for camera positions.
    :type camera_pos_max_x: float
    :param camera_pos_min_y: Minimum y-coordinate for camera positions.
    :type camera_pos_min_y: float
    :param camera_pos_max_y: Maximum y-coordinate for camera positions.
    :type camera_pos_max_y: float
    :param camera_pos_min_z: Minimum z-coordinate for camera positions.
    :type camera_pos_min_z: float
    :param camera_pos_max_z: Maximum z-coordinate for camera positions.
    :type camera_pos_max_z: float
    :param camera_rot_min_x: Minimum x-angle for camera rotation.
    :type camera_rot_min_x: float
    :param camera_rot_max_x: Maximum x-angle for camera rotation.
    :type camera_rot_max_x: float
    :param camera_rot_min_y: Minimum y-angle for camera rotation.
    :type camera_rot_min_y: float
    :param camera_rot_max_y: Maximum y-angle for camera rotation.
    :type camera_rot_max_y: float
    :param camera_rot_min_z: Minimum z-angle for camera rotation.
    :type camera_rot_min_z: float
    :param camera_rot_max_z: Maximum z-angle for camera rotation.
    :type camera_rot_max_z: float
    :param distant_light_min_intensity: Minimum intensity for distant lights.
    :type distant_light_min_intensity: float
    :param distant_light_max_intensity: Maximum intensity for distant lights.
    :type distant_light_max_intensity: float
    :return: List of dictionaries, each representing a scene's configuration.
    :rtype: list
    """

    # Generate single object scenes
    scene_configs = []
    for usd_model in usd_models:
        for _ in range(num_scenes_per_object):
            scene_config = {}
            scene_config["objects"] = [generate_object_config(usd_model, object_init_min_x, object_init_max_x,
                         object_init_min_y, object_init_max_y,
                         object_init_min_z, object_init_max_z) for _ in range(num_objects_per_scene)]
            scene_config["per_frame_config"] = generate_per_frame_config(num_random_materials,
                         num_frames_per_scene, num_objects_per_scene,
                         num_sphere_lights,
                         sphere_min_x, sphere_max_x,
                         sphere_min_y, sphere_max_y,
                         sphere_min_z, sphere_max_z,
                         sphere_min_intensity, sphere_max_intensity,
                         distant_light_min_rot_x, distant_light_max_rot_x,
                         distant_light_min_rot_y, distant_light_max_rot_y,
                         distant_light_min_rot_z, distant_light_max_rot_z,
                         camera_pos_min_x, camera_pos_max_x,
                         camera_pos_min_y, camera_pos_max_y,
                         camera_pos_min_z, camera_pos_max_z,
                         camera_rot_min_x, camera_rot_max_x,
                         camera_rot_min_y, camera_rot_max_y,
                         camera_rot_min_z, camera_rot_max_z,
                         distant_light_min_intensity, distant_light_max_intensity)

            scene_configs.append(scene_config)

    # Generate multiple object scenes
    for _ in range(len(usd_models)):
        for _ in range(num_scenes_per_object):
            scene_config = {"objects": []}
            for _ in range(num_objects_per_scene):
                usd_model = random.choice(usd_models)
                scene_config["objects"].append(generate_object_config(usd_model, object_init_min_x, object_init_max_x,
                         object_init_min_y, object_init_max_y,
                         object_init_min_z, object_init_max_z))

            scene_config["per_frame_config"] = generate_per_frame_config(num_random_materials,
                         num_frames_per_scene, num_objects_per_scene,
                         num_sphere_lights,
                         sphere_min_x, sphere_max_x,
                         sphere_min_y, sphere_max_y,
                         sphere_min_z, sphere_max_z,
                         sphere_min_intensity, sphere_max_intensity,
                         distant_light_min_rot_x, distant_light_max_rot_x,
                         distant_light_min_rot_y, distant_light_max_rot_y,
                         distant_light_min_rot_z, distant_light_max_rot_z,
                         camera_pos_min_x, camera_pos_max_x,
                         camera_pos_min_y, camera_pos_max_y,
                         camera_pos_min_z, camera_pos_max_z,
                         camera_rot_min_x, camera_rot_max_x,
                         camera_rot_min_y, camera_rot_max_y,
                         camera_rot_min_z, camera_rot_max_z,
                         distant_light_min_intensity, distant_light_max_intensity)

            scene_configs.append(scene_config)

    return scene_configs


def generate_train_val_splits(num_scenes: int, num_frames_per_scene: int, val_share: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the scene indices into training and validation sets based on the specified validation share.

    :param num_scenes: Total number of scenes.
    :type num_scenes: int
    :param num_frames_per_scene: Number of frames in each scene.
    :type num_frames_per_scene: int
    :param val_share: Fraction of the dataset to be used as the validation set.
    :type val_share: float
    :return: Two numpy arrays containing the indices for the training and validation sets, respectively.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    scene_indices = np.arange(num_scenes * num_frames_per_scene)
    return train_test_split(scene_indices, test_size=val_share)


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
    parser.add_argument("--out_dir", default="configs", help="Output directory for the generated configurations")
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
    parser.add_argument("--num_objects_per_cluttered_scene", default=100, type=int,
                        help="Specifies the number of objects in the cluttered scene")
    parser.add_argument("--num_scenes_per_object", default=5, type=int,
                        help="Specifies the number of scenes to generate for each object in the USD directory. Additionally n * num_scenes_per_object scenes will be generated with all objects in the scene. (n is num_frames_per_object times objects in the USD directory)")
    parser.add_argument("--num_cluttered_scenes_per_object", default=5, type=int,
                        help="Specifies the number of cluttered scenes to generate for each object in the USD directory. Additionally n * num_scenes_per_object scenes will be generated with all objects in the scene. (n is num_frames_per_object times objects in the USD directory)")
    parser.add_argument("--num_sphere_lights", default=0, type=int,
                        help="Specifies the number of sphere lights with random light color in the scene")
    parser.add_argument("--object_init_min_x", default=-2.0, type=float,  # TODO
                        help="The minimum initial x coordinate of the object in the scene")
    parser.add_argument("--object_init_max_x", default=-1.5, type=float,  # TODO
                        help="The maximum initial x coordinate of the object in the scene")
    parser.add_argument("--object_init_min_y", default=-0.4, type=float,  # TODO
                        help="The minimum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_max_y", default=0.4, type=float,   # TODO
                        help="The maximum initial y coordinate of the object in the scene")
    parser.add_argument("--object_init_min_z", default=3, type=float,     # TODO
                        help="The minimum initial z coordinate of the object in the scene")
    parser.add_argument("--object_init_max_z", default=6, type=float,     # TODO
                        help="The maximum initial z coordinate of the object in the scene")
    parser.add_argument("--cluttered_scene_object_init_min_x", default=-2.0, type=float,  # TODO
                        help="The minimum initial x coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_max_x", default=-1.5, type=float,  # TODO
                        help="The maximum initial x coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_min_y", default=-0.4, type=float,  # TODO
                        help="The minimum initial y coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_max_y", default=0.4, type=float,  # TODO
                        help="The maximum initial y coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_min_z", default=3, type=float,  # TODO
                        help="The minimum initial z coordinate of the object in the cluttered scene")
    parser.add_argument("--cluttered_scene_object_init_max_z", default=6, type=float,  # TODO
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
    parser.add_argument("--camera_pos_x", default=-0.5, type=float,  #TODO
                        help="The x coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_y", default=-0.3, type=float,  #TODO
                        help="The y coordinate of the camera in the scene")
    parser.add_argument("--camera_pos_z", default=3, type=float,     #TODO
                        help="The z coordinate of the camera in the scene")
    parser.add_argument("--camera_rot_x", default=-180, type=float,  #TODO
                        help="The rotation of the camera around the x axis in degrees")
    parser.add_argument("--camera_rot_y", default=-120, type=float,  #TODO
                        help="The rotation of the camera around the y axis in degrees")
    parser.add_argument("--camera_rot_z", default=-180, type=float,  #TODO
                        help="The rotation of the camera around the z axis in degrees")  #TODO
    parser.add_argument("--distant_light_rot_x", default=-90, type=float,
                        help="The rotation around x of the direct light in the scene")   #TODO
    parser.add_argument("--distant_light_rot_y", default=-90, type=float,
                        help="The rotation around y of the direct light in the scene")   #TODO
    parser.add_argument("--distant_light_rot_z", default=-180, type=float,
                        help="The z coordinate of the direct light in the scene")  #TODO
    parser.add_argument("--distant_light_intensity", default=200, type=float,
                        help="The minimum light intensity of the direct light")
    parser.add_argument("--metallic_scene_distant_light_intensity", default=200, type=float,  #TODO
                        help="The minimum light intensity of the direct light in scenes with metallic material")
    parser.add_argument("--conveyor_belt_speed", default=default_conveyor_belt_speed, type=float,
                        help="The speed of the conveyor belt in the simulation")
    parser.add_argument("--min_x_pos_for_record_start", default=default_min_x_pos_for_record_start, type=float,
                        help="Defines the minimum x coordinate at least one object needs to have passed to start writing the dataset")
    parser.add_argument("--render_frequency", default=default_render_frequency, type=int,
                        help="The render frequency in Hz")
    parser.add_argument("--physics_frequency", default=360, type=int,
                        help="The frequency at which the physics are calculated")

    # Parse args
    args = parser.parse_args(argv)

    # Assign variables
    usd_dir = args.usd_dir
    out_dir = args.out_dir
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
    num_objects_per_cluttered_scene = args.num_objects_per_cluttered_scene
    num_scenes_per_object = args.num_scenes_per_object
    num_cluttered_scenes_per_object = args.num_cluttered_scenes_per_object
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
    metallic_scene_distant_light_intensity = args.metallic_scene_distant_light_intensity
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

    # Check for existing config at out_path
    if os.path.exists(out_dir):
        print(f"Output directory at '{out_dir}' already exists. Exiting...")
        exit(-1)

    # Test usd_dir
    if not os.path.isdir(usd_dir):
        print("USD directory does not exist. Exiting...")
        exit(-1)

    usd_models = io.get_usd_models(usd_dir)

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
                         distant_light_intensity, distant_light_intensity),
        "conveyor_belt_speed": conveyor_belt_speed,
        "min_x_pos_for_record_start": min_x_pos_for_record_start,
        "num_frames_per_scene": num_frames_per_scene,
        "render_frequency": render_frequency,
        "physics_frequency": physics_frequency,
        "generation_script_args": vars(args)
    }

    data_generation_config["train_frames"] = None
    data_generation_config["val_frames"] = None

    # Write config
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + "/test.conf", "w") as f:
        json.dump(data_generation_config, f, indent=4)


if __name__ == '__main__':
    main(sys.argv[1:])
