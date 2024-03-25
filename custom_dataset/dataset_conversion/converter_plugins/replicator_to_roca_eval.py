import json
import os
import glob
import shutil

from PIL import Image

from .converter_interface import ConverterInterface
from .dataset_helper import replicator, roca


class ReplicatorToRocaEval(ConverterInterface):
    @staticmethod
    def _generate_class_labels_file(obj_paths: list[str], output_dir: str):
        # Generate semantic class labels
        # Sort is required, because this is how the order in scan2cad_alignment_classes.json is created
        # We need match this file because the index of the class in scan2cad_alignment_classes.json is used as the category_id during training
        semantic_class_labels = sorted([roca.obj_path_to_semantic_label(obj_path) for obj_path in obj_paths])

        # Write class labels to file
        with open(os.path.join(output_dir, "semantic_class_labels.json"), "w") as f:
            json.dump(semantic_class_labels, f, indent=4)

    @staticmethod
    def _convert_intrinsics(rep_data_path: str, output_dir: str, scene_numbers: list[str]):
        # Iterate through all scene_numbers
        for nr in scene_numbers:
            # Read the contents of the JSON file matching the pattern camera_params_*.json
            with open(os.path.join(rep_data_path, f"camera_params_{nr}.json"), 'r') as file:
                camera_params = json.load(file)

            # Extract needed parameters
            camera_focal_length = camera_params['cameraFocalLength']
            render_product_resolution = camera_params['renderProductResolution']
            camera_aperture_width = camera_params['cameraAperture'][0]

            # Calculate f_x, f_y, c_x, c_y
            f_x = f_y = (camera_focal_length * render_product_resolution[0]) / camera_aperture_width
            c_x = render_product_resolution[0] / 2
            c_y = render_product_resolution[1] / 2

            # Write to intrinsics_color.txt
            intrinsics_file_path = os.path.join(output_dir, f"image_{nr}.txt")
            with open(intrinsics_file_path, 'w') as intrinsics_file:
                intrinsics_file.write(f"{f_x} 0 {c_x}\n0 {f_y} {c_y}\n0 0 1\n")

    @staticmethod
    def _convert_images(rep_data_path: str, output_dir: str, scene_numbers: list[str]):
        """
                Copies and converts images from PNG to JPG format and renames them.

                :param rep_data_path: Path to the folder containing rgb_{nr}.png files.
                :param output_dir: Path to the output folder where images will be saved in the format ScanNet25k/tasks/scannet_frames_25k/scene{NR}/color/000000.jpg.
                :param scene_numbers: List of scene numbers.
                :type rep_data_path: str
                :type output_dir: str
                :type scene_numbers: list[str]
                :return: None
                """

        # Iterate through all scene_numbers
        for nr in scene_numbers:
            # Set the new file path
            new_file_path = os.path.join(output_dir, f"image_{nr}.jpg")

            # Open the PNG file and convert it to JPG
            with Image.open(os.path.join(rep_data_path, f"rgb_{nr}.png")) as img:
                img.convert('RGB').save(new_file_path, 'JPEG')

    @staticmethod
    def _copy_world_pose_data(rep_data_path: str, output_dir: str):
        # Load all world pose files
        world_pose_file_paths = glob.glob(os.path.join(rep_data_path, "world_pose_visible_objects_*.json"))
        for world_pose_file_path in world_pose_file_paths:

            # Create new file path and copy
            filename = os.path.basename(world_pose_file_path)
            new_path = os.path.join(output_dir, filename)
            shutil.copy(world_pose_file_path, new_path)

    @staticmethod
    def convert(replicator_data_dir: str, obj_files_dir: str, output_dir: str) -> None:
        """
        Converts data from NVIDIA Replicator to the format of the ROCA evaluator.

        :param replicator_data_dir: Directory containing the NVIDIA Replicator data.
        :param obj_files_dir: Directory containing the OBJ files.
        :param output_dir: Target directory for the converted data.
        :type replicator_data_dir: str
        :type obj_files_dir: str
        :type output_dir: str
        """

        print("Converting the generated evaluation data to the format required by the ROCA evaluator...")

        # Load obj paths
        obj_paths = glob.glob(os.path.join(obj_files_dir, "*.obj"))

        # Load scene numbers
        scene_numbers = replicator.get_scene_nrs(replicator_data_dir)

        # Convert eval data
        print("Generating the semantic class labels file...")
        ReplicatorToRocaEval._generate_class_labels_file(obj_paths, output_dir)
        print("Converting camera intrinsics...")
        ReplicatorToRocaEval._convert_intrinsics(replicator_data_dir, output_dir, scene_numbers)
        print("Converting images...")
        ReplicatorToRocaEval._convert_images(replicator_data_dir, output_dir, scene_numbers)
        print("Copying world pose data...")
        ReplicatorToRocaEval._copy_world_pose_data(replicator_data_dir, output_dir)


