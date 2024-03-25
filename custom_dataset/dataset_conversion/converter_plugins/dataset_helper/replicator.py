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
