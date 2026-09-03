import glob
import os


def get_scene_nrs(rep_data_path: str) -> list[str]:
    """
    Retrieves scene numbers from a replicator dataset.

    :param rep_data_path: Path to the replicator dataset containing camera_params_*.json files.
    :type rep_data_path: str
    :return: List of scene numbers.
    :rtype: list[str]
    """

    # Find all JSON files matching the pattern camera_params_*.json
    json_files = glob.glob(os.path.join(rep_data_path, "camera_params_*.json"))

    # Create and return list of scene numbers
    return [os.path.basename(json_file).split('_')[2].split('.')[0] for json_file in json_files]


def camera_params_to_intrinsics(camera_params: dict) -> tuple[float, float, float, float]:
    """
    Computes pixel-space intrinsics from a Replicator camera_params_*.json dict.

    Note: cameraAperture[1] (the vertical aperture reported by Replicator) is a static USD attribute
    default and is not what the renderer actually uses for the vertical field of view - that is
    derived from cameraAperture[0] (horizontal) and the resolution's aspect ratio instead. 
    vertical_aperture_offset is set relative to that same derived aperture, so c_y must be 
    computed with it too, not with cameraAperture[1].

    :param camera_params: Parsed contents of a camera_params_*.json file.
    :type camera_params: dict
    :return: The (f_x, f_y, c_x, c_y) pixel-space intrinsics.
    :rtype: tuple[float, float, float, float]
    """

    focal_length = camera_params["cameraFocalLength"]
    width_px, height_px = camera_params["renderProductResolution"]
    aperture_width = camera_params["cameraAperture"][0]
    vertical_aperture = aperture_width * height_px / width_px
    offset_x, offset_y = camera_params.get("cameraApertureOffset", [0.0, 0.0])  # .get(...) required for compatibility with old datasets

    f_x = f_y = (focal_length * width_px) / aperture_width
    c_x = width_px / 2 + offset_x * width_px / aperture_width
    c_y = height_px / 2 + offset_y * height_px / vertical_aperture
    return f_x, f_y, c_x, c_y
