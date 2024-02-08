def __parse_writer_init_args(writer_init_args: list[str]) -> dict:
    """
    Parses initialization arguments for a writer.

    Converts argument strings into a dictionary format, interpreting values as booleans if they match 'true'.
    E.g. ['rgb=true', 'depth=true'] is converted to {'rgb': True, 'depth': True}.

    :param writer_init_args: A list of string arguments.
    :type writer_init_args: list[str]
    :return: A dictionary mapping argument names to their parsed boolean values.
    :rtype: dict
    """

    args_dict = {}
    for arg in writer_init_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args_dict[key] = value.lower() == 'true'
    return args_dict


def parse_writer_args(writer_args: list[list[str]]) -> list[dict]:
    """
    Parses arguments for multiple writers.

    Converts a list of argument lists into a dictionary format, suitable for initializing multiple writers.

    :param writer_args: A list of lists, each containing arguments for a specific writer.
    :type writer_args: list[list[str]]
    :return: A dictionary containing configurations for each writer.
    :rtype: list[dict]
    """

    writers = []
    for writer_arg in writer_args:
        writer_conf = {"name": writer_arg[0], "args": __parse_writer_init_args(writer_arg[1:])}
        writers.append(writer_conf)

    return writers
