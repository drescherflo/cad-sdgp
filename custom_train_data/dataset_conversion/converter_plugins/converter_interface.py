from abc import ABC, abstractmethod


class ConverterInterface(ABC):
    @staticmethod
    @abstractmethod
    def convert(replicator_data_dir: str, obj_files_dir: str, output_dir: str) -> None:
        """
        Abstract method to be implemented for converting data from a specific format to another.

        :param replicator_data_dir: Directory containing the replicator dataset to be converted.
        :param obj_files_dir: Directory containing obj files for conversion.
        :param output_dir: Directory where the converted data will be saved.
        """
        raise NotImplementedError
