import numpy as np
from omni.isaac.core.materials import OmniPBR, OmniGlass


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
