import pickle
import shutil
import os
import glob
import json
import numpy as np
from PIL import Image
import trimesh
from trimesh.sample import sample_surface_even

from .converter_interface import ConverterInterface
from .dataset_helper import replicator, roca


class ReplicatorToRoca(ConverterInterface):
    @staticmethod
    def _get_obj_paths_semantic_labels_and_class_id(obj_dir: str) -> list[dict]:
        """
        Retrieves paths to OBJ files, their semantic labels, and class IDs.

        :param obj_dir: Path to the directory containing the OBJ files.
        :type obj_dir: str
        :return: A list of dictionaries with paths, semantic labels, and class IDs of the OBJ files.
        :rtype: list[dict]
        """

        # Find all .obj files in the obj_path
        obj_paths = glob.glob(os.path.join(obj_dir, "*.obj"))

        # Extract semantic labels / categories from the file name
        semantic_labels = [roca.obj_path_to_semantic_label(obj_path) for obj_path in obj_paths]
        files_and_labels = list(zip(obj_paths, semantic_labels))

        # Sort is required, because this is how the order in scan2cad_alignment_classes.json is created
        # The index of the class in scan2cad_alignment_classes.json is then used as the category_id during training
        files_and_labels = sorted(files_and_labels, key=lambda x: x[1])

        # Build dictionary list
        return [{"obj_file": obj_file, "semantic_label": semantic_label, "class_id": idx} for
                idx, (obj_file, semantic_label) in enumerate(files_and_labels)]

    @staticmethod
    def _replicator_intrinsics_to_scannet(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
        """
        Generates intrinsics_color.txt files from camera parameters JSON files.

        :param rep_data_path: Path to the folder containing camera_params_*.json files.
        :param roca_data_path: Path to the output folder where ScanNet25k/tasks/scannet_frames_25k/scene{NR}/intrinsics_color.txt files will be saved.
        :param scene_numbers: List of scene numbers.
        :type rep_data_path: str
        :type roca_data_path: str
        :type scene_numbers: list[str]
        :return: None
        """

        # Iterate through all scene_numbers
        for nr in scene_numbers:
            # Read the contents of the JSON file matching the pattern camera_params_*.json
            with open(os.path.join(rep_data_path, f"camera_params_{nr}.json"), 'r') as file:
                camera_params = json.load(file)

            # Calculate f_x, f_y, c_x, c_y
            f_x, f_y, c_x, c_y = replicator.camera_params_to_intrinsics(camera_params)

            # Prepare the output directory
            output_dir = os.path.join(roca_data_path, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}")
            os.makedirs(output_dir, exist_ok=True)

            # Write to intrinsics_color.txt
            intrinsics_file_path = os.path.join(output_dir, "intrinsics_color.txt")
            with open(intrinsics_file_path, 'w') as intrinsics_file:
                intrinsics_file.write(f"{f_x} 0 {c_x} 0\n0 {f_y} {c_y} 0\n0 0 1 0\n0 0 0 1\n")

    @staticmethod
    def _replicator_image_to_scannet(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
        """
        Copies and converts images from PNG to JPG format and renames them.

        :param rep_data_path: Path to the folder containing rgb_{nr}.png files.
        :param roca_data_path: Path to the output folder where images will be saved in the format ScanNet25k/tasks/scannet_frames_25k/scene{NR}/color/000000.jpg.
        :param scene_numbers: List of scene numbers.
        :type rep_data_path: str
        :type roca_data_path: str
        :type scene_numbers: list[str]
        :return: None
        """

        # Iterate through all scene_numbers
        for nr in scene_numbers:
            # Prepare the output directory
            output_dir = os.path.join(roca_data_path, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}/color")
            os.makedirs(output_dir, exist_ok=True)

            # Set the new file path
            new_file_path = os.path.join(output_dir, "000000.jpg")

            # Open the PNG file and convert it to JPG
            with Image.open(os.path.join(rep_data_path, f"rgb_{nr}.png")) as img:
                img.convert('RGB').save(new_file_path, 'JPEG')

    @staticmethod
    def _copy_obj_files(roca_data_path: str, obj_paths_labels_ids: list[dict]) -> None:
        """
        Copies .obj files to the ROCA data path respecting the ShapeNet data structure.

        :param obj_paths_labels_ids: List of dictionaries with obj file paths and their semantic labels and class ids.
        :param roca_data_path: ROCA data path where .obj files will be copied to.
        :type roca_data_path: str
        :type obj_paths_labels_ids: list[dict]
        :return: None
        """

        for obj_file_label_id in obj_paths_labels_ids:
            # Prepare the output directory
            output_dir = os.path.join(roca_data_path, f"ShapeNetCore.v2/{obj_file_label_id['class_id']}/0/models")
            os.makedirs(output_dir, exist_ok=True)

            # Set the new file path
            new_file_path = os.path.join(output_dir, "model_normalized.obj")

            # Copy the .obj file
            shutil.copy(obj_file_label_id["obj_file"], new_file_path)

    @staticmethod
    def _generate_full_annotations_json(rep_data_path: str, roca_data_path: str, scene_numbers: list[str],
                                        obj_paths_labels_ids: list[dict]) -> None:
        """
        Generates a full_annotations.json file from world_pose_visible_objects_[nr].json files.

        :param rep_data_path: Path to the folder containing world_pose_visible_objects_[nr].json files.
        :param roca_data_path: ROCA data path where the Scan2CAD/full_annotations.json will be saved.
        :param scene_numbers: List of scene numbers as strings.
        :param obj_paths_labels_ids: List of dictionaries with obj file paths and their semantic labels and class ids.
        :type rep_data_path: str
        :type roca_data_path: str
        :type scene_numbers: list[str]
        :type obj_paths_labels_ids: list[dict]
        :return: None
        """

        annotations = []
        for nr in scene_numbers:
            # Read the contents of the JSON file
            file_path = os.path.join(rep_data_path, f"world_pose_visible_objects_{nr}.json")
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Process the data
            scene_data = {
                "id_scan": f"scene{nr}",
                "trs": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0]
                }
            }

            aligned_models = []
            for obj in data:
                # Find matching class_id
                semantic_label = obj["semantic_labels"]["class"]
                class_id = None
                for obj_file_label_id in obj_paths_labels_ids:
                    if obj_file_label_id["semantic_label"] == semantic_label:
                        class_id = obj_file_label_id["class_id"]
                        break

                if class_id is None:
                    raise RuntimeError("No matching class id found for semantic label '{}'.".format(semantic_label))

                # Process obj data
                aligned_models.append({
                    "sym": "__SYM_NONE",
                    # Assuming symmetry as none for all models since symmetry is not considered during training (https://github.com/cangumeli/ROCA/blob/main/network/roca/data/cad_manager.py ll. 156 (19.01.2024)
                    "catid_cad": str(class_id),
                    "id_cad": "0",
                    "trs": {
                        "translation": [obj['pose']['position']['x'],
                                        obj['pose']['position']['y'],
                                        obj['pose']['position']['z']],
                        "rotation": [obj['pose']['orientation']['w'],
                                     obj['pose']['orientation']['x'],
                                     obj['pose']['orientation']['y'],
                                     obj['pose']['orientation']['z']],
                        "scale": [1.0, 1.0, 1.0]
                    }
                })

            scene_data["aligned_models"] = aligned_models
            annotations.append(scene_data)

        # Saving the annotations to full_annotations.json
        output_path = os.path.join(roca_data_path, "Scan2CAD", "full_annotations.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as output_file:
            json.dump(annotations, output_file, indent=4)

    @staticmethod
    def _generate_metadata_taxonomy_9(roca_metadata_path: str, obj_paths_labels_ids: list[dict]) -> None:
        """
        Generates a taxonomy JSON file for ROCA metadata.

        :param roca_metadata_path: File path where the taxonomy file will be saved.
        :param obj_paths_labels_ids: List of dictionaries with obj file paths and their semantic labels and class ids.
        :type roca_metadata_path: str
        :type obj_paths_labels_ids: list[dict]
        :return: None
        """

        taxonomy = [{"name": obj_path_label_id["semantic_label"], "shapenet": str(obj_path_label_id["class_id"])} for
                    obj_path_label_id in obj_paths_labels_ids]
        json.dump(taxonomy, open(os.path.join(roca_metadata_path, "scan2cad_taxonomy_9.json"), "w"), indent=4)

    @staticmethod
    def __write_list_to_txt(output_path: str, string_list: list[str]) -> None:
        """
        Writes a list of strings to a text file, each string on a new line.

        :param output_path: The file path where the text file will be saved.
        :param string_list: A list of strings to be written to the file.
        :type output_path: str
        :type string_list: list[str]
        :return: None
        """

        with open(output_path, "w") as f:
            f.write("\n".join(string_list))

    @staticmethod
    def _generate_metadata_label_id_files(roca_metadata_path: str, obj_paths_labels_ids: list[dict]) -> None:
        """
        Generates text files containing label IDs for ROCA metadata.

        :param roca_metadata_path: File path where the label ID files will be saved.
        :param obj_paths_labels_ids: List of dictionaries with obj file paths and their semantic labels and class ids.
        :type roca_metadata_path: str
        :type obj_paths_labels_ids: list[dict]
        :return: None
        """

        # Label ID files are just a list of an index (starting by 1) and the name of the label
        # Order and assigned index are unrelated to the previously generate class id
        # For consistency the index is generated by adding 1 to the class id
        labels_with_id_str_list = [f"{obj_path_label_id['class_id'] + 1}\t{obj_path_label_id['semantic_label']}" for
                                   obj_path_label_id in obj_paths_labels_ids]
        ReplicatorToRoca.__write_list_to_txt(os.path.join(roca_metadata_path, "labelids_all.txt"), labels_with_id_str_list)
        ReplicatorToRoca.__write_list_to_txt(os.path.join(roca_metadata_path, "labelids.txt"), labels_with_id_str_list)

    @staticmethod
    def _generate_metadata_train_val_files(replicator_dir: str, roca_metadata_path: str) -> None:
        """
        Generates train and validation split files for ROCA metadata.

        :param replicator_dir: Path containing train_val_scenes.json.
        :param roca_metadata_path: File path where the train and validation files will be saved.
        :type replicator_dir: str
        :type roca_metadata_path: str
        :return: None
        """

        # Read train_val_scenes
        with open(os.path.join(replicator_dir, "train_val_scenes.json"), "r") as f:
            train_val_scenes = json.load(f)

        train_scenes = [f"scene{scene_number:04d}" for scene_number in train_val_scenes["train_scenes"]]
        val_scenes = [f"scene{scene_number:04d}" for scene_number in train_val_scenes["val_scenes"]]

        # Write scenes
        ReplicatorToRoca.__write_list_to_txt(os.path.join(roca_metadata_path, "scannetv2_train.txt"), train_scenes)
        ReplicatorToRoca.__write_list_to_txt(os.path.join(roca_metadata_path, "scannetv2_val.txt"), val_scenes)

        # Write val images
        val_images = [scene_name + " 0" for scene_name in
                      val_scenes]  # Since there is only one picture per scene, use this one
        ReplicatorToRoca.__write_list_to_txt(os.path.join(roca_metadata_path, "val_images.txt"), val_images)

    @staticmethod
    def _replicator_cam_pose_to_scannet(replicator_dir: str, roca_dataset_dir: str, scene_numbers: list[str]) -> None:
        """
        Reads camera parameters and calculates the world to ROS camera view transformation matrix.

        :param replicator_dir: Path containing camera_params_[nr].json files.
        :param roca_dataset_dir: Path where the inverted camera view transform will be saved.
        :param scene_numbers: List of scene numbers as strings.
        :type replicator_dir: str
        :type roca_dataset_dir: str
        :type scene_numbers: list[str]
        :return: None
        """

        for nr in scene_numbers:
            json_file_path = os.path.join(replicator_dir, f"camera_params_{nr}.json")
            # Read the JSON file
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Build T^R_W = T^R_I * T^I_W
            # Extract the world to isaac camera view transform matrix  (T^I_W)
            isaac_camera_view_to_world = np.array(data["cameraViewTransform"]).reshape([4, 4]).transpose()

            # Create isaac camera view to ros camera view transformation (T^R_I)
            ros_camera_view_to_isaac_camera_view = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ])

            # Calculate ros camera view to world camera view transformation (T^R_W)
            ros_camera_view_to_world = ros_camera_view_to_isaac_camera_view @ isaac_camera_view_to_world

            # According to ROCA code in render.py, T^W_R needs to be saved
            world_to_ros_camera_view = np.linalg.inv(ros_camera_view_to_world)

            # Prepare the output directory
            output_dir = os.path.join(roca_dataset_dir, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}/pose")
            os.makedirs(output_dir, exist_ok=True)

            output_file_path = os.path.join(output_dir, "000000.txt")
            with open(output_file_path, 'w') as output_file:
                for row in world_to_ros_camera_view:
                    output_file.write(" ".join([f"{val}" for val in row]) + "\n")

    @staticmethod
    def _generate_point_files(roca_dataset_dir: str, obj_paths_labels_ids: list[dict]):
        point_datasets = []
        for obj_path_label_id in obj_paths_labels_ids:
            # Load the model
            obj_file = obj_path_label_id["obj_file"]
            mesh = trimesh.load(obj_file)

            # Check if every edge is included in two faces
            if not mesh.is_watertight:
                print(
                    f"Warning: The mesh in '{obj_file}' does not have all edges included in two faces. This may lead to inaccurate point sampling.")

            # Sample 1024 points from the mesh surface (like in https://github.com/cangumeli/ROCA/blob/main/network/assets/points_val.pkl (18.01.2024))
            points = np.asarray(sample_surface_even(mesh, 1024)[0]).astype(np.float32)

            # Create dataset for current mesh
            point_datasets.append({
                "points": points,
                "catid_cad": str(obj_path_label_id["class_id"]),
                "id_cad": "0",
                "category_id": obj_path_label_id["class_id"]
            })

        # Write point files
        out_dir = os.path.join(roca_dataset_dir, "Dataset")
        os.makedirs(out_dir, exist_ok=True)
        for split in ["train", "val"]:
            with open(os.path.join(out_dir, f"points_{split}.pkl"), "wb") as f:
                pickle.dump(point_datasets, f)

    @staticmethod
    def convert(replicator_data_dir: str, obj_files_dir: str, output_dir: str) -> None:
        """
        Converts data from NVIDIA Replicator to the ROCA format.

        :param replicator_data_dir: Directory containing the NVIDIA Replicator data.
        :param obj_files_dir: Directory containing the OBJ files.
        :param output_dir: Target directory for the converted data.
        :type replicator_data_dir: str
        :type obj_files_dir: str
        :type output_dir: str
        """

        print("Converting the generated training data from NVIDIA Replicator to ROCA format...")

        # Create roca_metadata_dir if necessary
        roca_metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(roca_metadata_dir, exist_ok=True)

        # Convert training data
        scene_numbers = replicator.get_scene_nrs(replicator_data_dir)
        obj_paths_labels_ids = ReplicatorToRoca._get_obj_paths_semantic_labels_and_class_id(obj_files_dir)
        print("Converting camera intrinsics...")
        ReplicatorToRoca._replicator_intrinsics_to_scannet(replicator_data_dir, output_dir, scene_numbers)
        print("Converting images...")
        ReplicatorToRoca._replicator_image_to_scannet(replicator_data_dir, output_dir, scene_numbers)
        print("Converting camera to world transformations...")
        ReplicatorToRoca._replicator_cam_pose_to_scannet(replicator_data_dir, output_dir, scene_numbers)
        print("Copying CAD models...")
        ReplicatorToRoca._copy_obj_files(output_dir, obj_paths_labels_ids)
        print("Generating full_annotations.json...")
        ReplicatorToRoca._generate_full_annotations_json(replicator_data_dir, output_dir, scene_numbers, obj_paths_labels_ids)
        print("Generating ROCA metadata files...")
        ReplicatorToRoca._generate_metadata_taxonomy_9(roca_metadata_dir, obj_paths_labels_ids)
        ReplicatorToRoca._generate_metadata_label_id_files(roca_metadata_dir, obj_paths_labels_ids)
        ReplicatorToRoca._generate_metadata_train_val_files(replicator_data_dir, roca_metadata_dir)
        print("Generating point files by sampling from CAD models...")
        ReplicatorToRoca._generate_point_files(output_dir, obj_paths_labels_ids)

        print("Done!")
