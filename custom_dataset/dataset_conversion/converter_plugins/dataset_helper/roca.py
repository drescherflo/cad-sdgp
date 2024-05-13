import os


def obj_path_to_semantic_label(obj_path: str) -> str:
    """
    Converts the file name of an OBJ file to a semantic label.

    :param obj_path: The path to the OBJ file.
    :type obj_path: str
    :return: A semantic label derived from the file name.
    :rtype: str
    """

    return os.path.basename(obj_path).split('.')[0].removesuffix(".obj").lower().replace(" ", "_")
