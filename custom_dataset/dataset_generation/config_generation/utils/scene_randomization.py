import random


def usd_model_to_semantic_class_label(usd_model: str) -> str:
    return usd_model.removesuffix("_obj.usd").lower().replace(" ", "_")

def generate_random_vec_3(min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float):
    """
    Generates a random 3D vector with each component within specified minimum and maximum values.

    :param min_x: Minimum value for the x-component.
    :type min_x: float
    :param max_x: Maximum value for the x-component.
    :type max_x: float
    :param min_y: Minimum value for the y-component.
    :type min_y: float
    :param max_y: Maximum value for the y-component.
    :type max_y: float
    :param min_z: Minimum value for the z-component.
    :type min_z: float
    :param max_z: Maximum value for the z-component.
    :type max_z: float
    :return: A list representing a random 3D vector [x, y, z].
    :rtype: list[float]
    """

    return [random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.uniform(min_z, max_z)]


def generate_random_rgb_color() -> list[float]:
    """
    Generates a random RGB color.

    Each color component is a float between 0 and 1, representing the intensity of red, green, and blue.

    :return: A list containing three floats, each representing the red, green, and blue color components.
    :rtype: list[float]
    """

    return generate_random_vec_3(0, 1, 0, 1, 0, 1)


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
                                              max_y: float, min_z: float, max_z: float, sphere_min_intensity: float, sphere_max_intensity: float) -> list[dict]:
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
    :param sphere_min_intensity: Minimum light intensity for a light sphere.
    :type sphere_min_intensity: float
    :param sphere_max_intensity: Maximum light intensity for a light sphere.
    :type sphere_max_intensity: float
    :return: A list containing the configurations of sphere lights.
    :rtype: list[dict]
    """

    return [
        {
            "position": generate_random_vec_3(min_x, max_x, min_y, max_y, min_z, max_z),
            "color": generate_random_rgb_color(),
            "intensity": random.uniform(sphere_min_intensity, sphere_max_intensity)
        }
        for _ in range(num_sphere_lights)]
