from .writer_interface import ResumableWriterInterface
from omni.replicator.core import BasicWriter


class ResumableBasicWriter(ResumableWriterInterface, BasicWriter):
    """
    Wrapper for the BasicWriter class from Replicator with adjustments to be resumable.
    """

    def __init__(self, output_dir: str,
                 init_frame_nr: int,
                 **kwargs):
        """
        Initializes an instance of ResumableBasicWriter with the specified output directory and initial frame number.
        This constructor overrides the abstract method defined in ResumableWriterInterface, setting up the basic writer
        with a specific output directory and initializing frame and sequence identifiers for resumable writing operations.

        :param output_dir: The directory where output files will be written.
        :type output_dir: str
        :param init_frame_nr: The initial frame number from which to start writing data, allowing for resumption of data writing.
        :type init_frame_nr: int
        :param kwargs: Additional keyword arguments passed to the BasicWriter's constructor.
        """

        BasicWriter.__init__(self, output_dir=output_dir, **kwargs)
        self._frame_id = init_frame_nr  # Set frame id to provided current frame number
        self._sequence_id = ""  # prevent resetting _frame_id to 0

    def write(self, data: dict):
        """
        Writes the provided data to the output directory. This method overrides the abstract write method
        defined in ResumableWriterInterface, utilizing the BasicWriter's write functionality to handle the actual data writing.

        :param data: The data to be written to the output directory. Expected to be a dictionary containing the data.
        :type data: dict
        """

        BasicWriter.write(self, data)
