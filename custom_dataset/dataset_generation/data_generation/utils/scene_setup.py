import numpy as np

import omni.replicator.core as rep
from omni.isaac.core.materials import OmniPBR, OmniGlass
from omni.replicator.core.scripts.utils import ReplicatorItem


def generate_materials(materials_config) -> list[OmniGlass | OmniPBR]:
    """
    Generates a list of OmniGlass or OmniPBR objects based on material configurations.

    :param materials_config: A list of dicts with keys 'is_glass', 'material_idx', 'color',
    and optionally 'surface_roughness' for OmniPBR.
    :type materials_config: list[dict]
    :return: A list of instantiated OmniGlass or OmniPBR objects.
    :rtype: list[OmniGlass | OmniPBR]
    """

    materials = []
    for material_config in materials_config:
        if material_config["is_glass"]:
            materials.append(OmniGlass(f"/obj_materials/material_{material_config['material_idx']}", color=np.array(material_config["color"])))
        else:
            material = OmniPBR(f"/obj_materials/material_{material_config['material_idx']}", color=np.array(material_config["color"]))
            material.set_reflection_roughness(material_config["surface_roughness"])
            materials.append(material)

    return materials


def randomize_sphere_light(sphere_lights: list[ReplicatorItem], sphere_light_idx: int,
                           sphere_light_configs: list[list[dict]]):
    sphere_light = sphere_lights[sphere_light_idx]
    sphere_light_config = [sphere_light_per_frame_config[sphere_light_idx] for sphere_light_per_frame_config in
                           sphere_light_configs]
    sphere_light_positions = [config["position"] for config in sphere_light_config]
    sphere_light_colors = [config["color"] for config in sphere_light_config]
    sphere_light_intensities = [config["intensity"] if "intensity" in config else 1000 for config in
                                sphere_light_config]  # if expression required for compatibility with older configs. 1000 is default value according to https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.create.light (08.02.2024)
    with sphere_light:
        rep.modify.attribute("color", rep.distribution.sequence(sphere_light_colors))
        rep.modify.attribute("intensity", rep.distribution.sequence(sphere_light_intensities))
        rep.modify.pose(position=rep.distribution.sequence(sphere_light_positions))
    return sphere_light
