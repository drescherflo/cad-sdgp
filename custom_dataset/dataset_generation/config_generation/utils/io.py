import os


def get_usd_models(usd_dir: str) -> list[str]:
    """
    List all files with the '.usd' extension in the specified directory.

    :param usd_dir: The path to the directory to search in.
    :type usd_dir: str
    :return: A list of file names with the '.usd' extension found in the specified directory.
    :rtype: list
    """

    return [file for file in os.listdir(usd_dir) if file.endswith('.usd')]
