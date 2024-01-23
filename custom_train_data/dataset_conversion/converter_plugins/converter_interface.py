from abc import ABC, abstractmethod


class ConverterInterface(ABC):
    @staticmethod
    @abstractmethod
    def convert(replicator_data_dir: str, obj_files_dir: str, output_dir: str) -> None:
        raise NotImplementedError
