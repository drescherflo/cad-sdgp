from omni.isaac.kit import SimulationApp


def quit_on_error(simulation_app: SimulationApp) -> None:
    """
    Terminates the program execution in case of an error.
    This function should be called when an unrecoverable error is encountered.

    :param simulation_app: The current running simulation app.
    :type simulation_app: SimulationApp
    """

    simulation_app.close()
    exit(-1)
