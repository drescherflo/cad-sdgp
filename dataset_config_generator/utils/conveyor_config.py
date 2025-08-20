import random

import numpy as np
from sklearn.model_selection import train_test_split

from .scene_randomization import generate_random_vec_3, generate_random_rgb_color, generate_materials_conf, generate_sphere_light_confs_for_one_frame, usd_model_to_semantic_class_label


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
                "semantic_class_label": usd_model_to_semantic_class_label(usd_model)
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
    scene_configs = generate_single_object_scenes(usd_models, num_random_materials,
                         num_frames_per_scene, num_objects_per_scene, num_scenes_per_object,
                         num_sphere_lights,
                         object_init_min_x, object_init_max_x,
                         object_init_min_y, object_init_max_y,
                         object_init_min_z, object_init_max_z,
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

    # Generate multiple object scenes
    scene_configs.extend(generate_multiple_object_scenes(usd_models, num_random_materials,
                         num_frames_per_scene, num_objects_per_scene, num_scenes_per_object,
                         num_sphere_lights,
                         object_init_min_x, object_init_max_x,
                         object_init_min_y, object_init_max_y,
                         object_init_min_z, object_init_max_z,
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
                         distant_light_min_intensity, distant_light_max_intensity))

    return scene_configs


def generate_multiple_object_scenes(usd_models: list[str], num_random_materials: int,
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
                         distant_light_min_intensity: float, distant_light_max_intensity: float):
    scene_configs = []
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
                                                                         distant_light_min_rot_x,
                                                                         distant_light_max_rot_x,
                                                                         distant_light_min_rot_y,
                                                                         distant_light_max_rot_y,
                                                                         distant_light_min_rot_z,
                                                                         distant_light_max_rot_z,
                                                                         camera_pos_min_x, camera_pos_max_x,
                                                                         camera_pos_min_y, camera_pos_max_y,
                                                                         camera_pos_min_z, camera_pos_max_z,
                                                                         camera_rot_min_x, camera_rot_max_x,
                                                                         camera_rot_min_y, camera_rot_max_y,
                                                                         camera_rot_min_z, camera_rot_max_z,
                                                                         distant_light_min_intensity,
                                                                         distant_light_max_intensity)

            scene_configs.append(scene_config)

    return scene_configs


def generate_single_object_scenes(usd_models: list[str], num_random_materials: int,
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
                         distant_light_min_intensity: float, distant_light_max_intensity: float):
    scene_configs = []
    for usd_model in usd_models:
        for _ in range(num_scenes_per_object):
            scene_config = {}
            scene_config["objects"] = [generate_object_config(usd_model, object_init_min_x, object_init_max_x,
                                                              object_init_min_y, object_init_max_y,
                                                              object_init_min_z, object_init_max_z) for _ in
                                       range(num_objects_per_scene)]
            scene_config["per_frame_config"] = generate_per_frame_config(num_random_materials,
                                                                         num_frames_per_scene, num_objects_per_scene,
                                                                         num_sphere_lights,
                                                                         sphere_min_x, sphere_max_x,
                                                                         sphere_min_y, sphere_max_y,
                                                                         sphere_min_z, sphere_max_z,
                                                                         sphere_min_intensity, sphere_max_intensity,
                                                                         distant_light_min_rot_x,
                                                                         distant_light_max_rot_x,
                                                                         distant_light_min_rot_y,
                                                                         distant_light_max_rot_y,
                                                                         distant_light_min_rot_z,
                                                                         distant_light_max_rot_z,
                                                                         camera_pos_min_x, camera_pos_max_x,
                                                                         camera_pos_min_y, camera_pos_max_y,
                                                                         camera_pos_min_z, camera_pos_max_z,
                                                                         camera_rot_min_x, camera_rot_max_x,
                                                                         camera_rot_min_y, camera_rot_max_y,
                                                                         camera_rot_min_z, camera_rot_max_z,
                                                                         distant_light_min_intensity,
                                                                         distant_light_max_intensity)

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
def generate_config(
                    usd_models: list[str],
                    frame_height: float, frame_width: float,
                    sub_frames_per_frame: float,
                    num_random_materials: int,
                    probability_of_glass_material: float,
                    num_frames_per_scene: int,
                    num_objects_per_scene: int,
                    num_scenes_per_object: int,
                    num_sphere_lights: int,
                    sphere_min_intensity: float, sphere_max_intensity: float,
                    sphere_min_x: float, sphere_max_x: float,
                    sphere_min_y: float, sphere_max_y: float,
                    sphere_min_z: float, sphere_max_z: float,
                    object_init_min_y: float, object_init_max_x: float,
                    object_init_min_x: float, object_init_max_y: float,
                    object_init_min_z: float, object_init_max_z: float,
                    val_dataset_share: float,
                    camera_pos_min_x: float, camera_pos_max_x: float,
                    camera_pos_min_y: float, camera_pos_max_y: float,
                    camera_pos_min_z: float, camera_pos_max_z: float,
                    camera_rot_min_x: float, camera_rot_max_x: float,
                    camera_rot_min_y: float, camera_rot_max_y: float,
                    camera_rot_min_z: float, camera_rot_max_z: float,
                    distant_light_min_intensity: float, distant_light_max_intensity: float,
                    distant_light_min_rot_x: float, distant_light_max_rot_x: float,
                    distant_light_min_rot_y: float, distant_light_max_rot_y: float,
                    distant_light_min_rot_z: float, distant_light_max_rot_z: float,
                    conveyor_belt_speed: float,
                    min_x_pos_for_record_start: float,
                    render_frequency: float,
                    physics_frequency: float,
                    args
                    ):
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
                                       distant_light_min_rot_x, distant_light_max_rot_x,
                                       distant_light_min_rot_y, distant_light_max_rot_y,
                                       distant_light_min_rot_z, distant_light_max_rot_z,
                                       camera_pos_min_x, camera_pos_max_x,
                                       camera_pos_min_y, camera_pos_max_y,
                                       camera_pos_min_z, camera_pos_max_z,
                                       camera_rot_min_x, camera_rot_max_x,
                                       camera_rot_min_y, camera_rot_max_y,
                                       camera_rot_min_z, camera_rot_max_z,
                                       distant_light_min_intensity, distant_light_max_intensity),
        "conveyor_belt_speed": conveyor_belt_speed,
        "min_x_pos_for_record_start": min_x_pos_for_record_start,
        "num_frames_per_scene": num_frames_per_scene,
        "render_frequency": render_frequency,
        "physics_frequency": physics_frequency,
        "generation_script_args": vars(args)
    }
    (train_frames, val_frames) = generate_train_val_splits(len(data_generation_config["scenes"]), num_frames_per_scene,
                                                           val_dataset_share)
    data_generation_config["train_frames"] = train_frames.tolist()
    data_generation_config["val_frames"] = val_frames.tolist()
    return data_generation_config