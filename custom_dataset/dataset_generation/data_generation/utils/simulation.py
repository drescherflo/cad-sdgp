import sys

from omni.isaac.kit import SimulationApp


def quit_on_error(msg: str, simulation_app: SimulationApp) -> None:
    """
    Terminates the program execution in case of an error and prints an error message.
    This function should be called when an unrecoverable error is encountered.

    :param msg: The message to be printed
    :type msg: str
    :param simulation_app: The current running simulation app.
    :type simulation_app: SimulationApp
    """

    print(msg, file=sys.stderr)
    simulation_app.close()
    exit(-1)
