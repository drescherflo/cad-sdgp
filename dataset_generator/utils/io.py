import os

from omni.isaac.kit import SimulationApp

from .simulation import quit_on_error


def quit_if_out_dir_not_empty(out_dir: str, simulation_app: SimulationApp) -> None:
    """
    Terminates the simulation if the output directory is not empty

    :param out_dir: Path to the output directory.
    :type out_dir: str
    :param simulation_app: The current running simulation app.
    :type simulation_app: SimulationApp
    """

    if len(os.listdir(out_dir)) != 0:
        print("")
        quit_on_error("Output directory is not empty. Exiting...", simulation_app)
