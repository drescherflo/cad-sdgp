from .writer_interface import ResumableWriterInterface
from omni.replicator.core import BasicWriter


class ResumableBasicWriter(ResumableWriterInterface, BasicWriter):

    def __init__(self, output_dir: str,
                 init_frame_nr: int,
                 **kwargs):
        BasicWriter.__init__(self, output_dir=output_dir, **kwargs)
        self._frame_id = init_frame_nr  # Set frame id to provided current frame number
        self._sequence_id = ""  # prevent resetting _frame_id to 0

    def write(self, data: dict):
        BasicWriter.write(self, data)
