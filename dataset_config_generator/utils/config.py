import inspect
from typing import Any


def check_range_plausibility(min_val: float, max_val: float) -> None:
    """
    Checks the plausibility of a given range by ensuring the minimum value is not greater than the maximum value.
    Raises a ValueError if the condition is not met, with a message indicating which variable violates the condition.

    This function uses a nested function to retrieve the variable name of the arguments for a more descriptive error message.
    The technique to get the variable names is based on inspecting the call stack and should be used with caution as it
    relies on the specific structure of the call stack and might not work in all Python environments or future versions.

    :param min_val: The minimum value of the range.
    :type min_val: float
    :param max_val: The maximum value of the range.
    :type max_val: float
    :return: None
    :raises ValueError: If min_val is greater than max_val, with a message specifying the variables involved.
    """

    if min_val > max_val:
        def get_var_name(var: Any) -> str:
            # Code from https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string (02.02.2024)
            callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

        raise ValueError(f"{get_var_name(min_val)} must be less or equal than {get_var_name(max_val)}")
