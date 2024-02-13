import os
import importlib
from typing import Type

from .converter_interface import ConverterInterface


def load_converter_plugins() -> list[Type[ConverterInterface]]:
    """
    Loads all converter plugins in the converter_plugins package.

    :return: A list of types derived from the ConverterInterface class.
    """

    plugin_package_name = "converter_plugins"
    plugin_dir = os.path.dirname(os.path.abspath(__file__))  # Path of current package (converter_plugins)

    converter_plugins = []
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            module = importlib.import_module('.' + module_name, package=plugin_package_name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, ConverterInterface) and attribute is not ConverterInterface:
                    converter_plugins.append(attribute)
    return converter_plugins
