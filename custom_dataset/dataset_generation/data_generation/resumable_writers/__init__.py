import os
import importlib
import sys

import omni.replicator.core as rep

from .writer_interface import ResumableWriterInterface


def load_resumable_writer_plugins() -> None:
    """
    Loads resumable writer plugins and registers them in the replicator writer registry.
    """

    plugin_package_name = "resumable_writers"
    plugin_dir = os.path.dirname(os.path.abspath(__file__))  # Path of current package (resumable_writers)

    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            module = importlib.import_module('.' + module_name, package=plugin_package_name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, ResumableWriterInterface) and attribute is not ResumableWriterInterface:
                    rep.WriterRegistry.register(attribute)
