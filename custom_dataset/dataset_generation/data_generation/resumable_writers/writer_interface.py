from abc import ABC, abstractmethod

from omni.replicator.core import Writer


class ResumableWriterInterface(Writer, ABC):
    @abstractmethod
    def __init__(self, output_dir: str, init_frame_nr: int, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def write(self, data: dict):
        raise NotImplementedError

