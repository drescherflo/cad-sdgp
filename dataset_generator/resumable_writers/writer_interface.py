from abc import ABC, abstractmethod

from omni.replicator.core import Writer


class ResumableWriterInterface(Writer, ABC):
    """
    Interface defining a resumable writer.
    """

    @abstractmethod
    def __init__(self, output_dir: str, init_frame_nr: int, *args, **kwargs):
        """
        Initializes an instance of the class with the specified output directory and initial frame number.
        This method is abstract and must be implemented by subclasses.

        :param output_dir: The directory where output will be written.
        :type output_dir: str
        :param init_frame_nr: The initial frame number from which to start writing data.
        :type init_frame_nr: int
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        """

        raise NotImplementedError

    @abstractmethod
    def write(self, data: dict):
        """
        Writes the provided data to an output destination.
        This method is abstract and must be implemented by subclasses to define how data is written.

        :param data: The data to be written. Expected to be a dictionary containing the data.
        :type data: dict
        """

        raise NotImplementedError


