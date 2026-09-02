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
    # OBJ→USD conversion (obj_to_usd.py) applies a Y-up→Z-up rotation via
    # convert_stage_up_z=True.  This 90° rotation around X maps (x,y,z)→(x,-z,y).
    # We must apply the same rotation to OBJ meshes before computing scales and
    # point clouds so that they match the USD prim frame used by Isaac Sim's
    # get_world_pose() quaternions.
    #_OBJ_TO_USD_ROTATION = np.array([
    #    [1,  0,  0, 0],
    #    [0,  0, -1, 0],
    #    [0,  1,  0, 0],
    #    [0,  0,  0, 1],
    #], dtype=float)

    @staticmethod
    def _compute_obj_scale(obj_path: str) -> list[float]:
        """
        Compute per-axis bounding box extents of an OBJ model in the USD frame.

        The ROCA transform is ``p = R @ diag(s) @ q + t``.
        Since our CAD models are at real-world scale (not normalized), the scale
        equals the bounding box extent along each axis, so that
        ``q_normalized = q_real / s`` puts NOC coordinates in ~[-0.5, 0.5].

        :param obj_path: Path to the OBJ file.
        :return: [sx, sy, sz] bounding box extents in meters (USD frame).
        """
        mesh = trimesh.load(obj_path, force='mesh')
        #mesh.apply_transform(ReplicatorToRoca._OBJ_TO_USD_ROTATION)
        extents = mesh.bounding_box.extents.tolist()  # [x_extent, y_extent, z_extent]
        return extents

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

            # Sort objects by their object index from prim_path to match
            # the alignment_id ordering used in instance segmentation masks
            sorted_data = sorted(data, key=lambda obj: int(obj["prim_path"].rsplit("object_", 1)[-1]))

            aligned_models = []
            for obj in sorted_data:
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

            # Note: despite the variable name, isaac_camera_view_to_world is actually the
            # view matrix (world-to-Isaac-camera), since cameraViewTransform is stored column-major.
            # T_{w2ros} = T_{i2ros} @ T_{w2i} = flip @ view_matrix
            world_to_ros_camera_view = ros_camera_view_to_isaac_camera_view @ isaac_camera_view_to_world

            # Save camera-to-world (ScanNet convention), as expected by ROCA's render.py
            ros_camera_view_to_world = np.linalg.inv(world_to_ros_camera_view)

            output_dir = os.path.join(roca_dataset_dir, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}/pose")
            os.makedirs(output_dir, exist_ok=True)

            output_file_path = os.path.join(output_dir, "000000.txt")
            with open(output_file_path, 'w') as output_file:
                for row in ros_camera_view_to_world:
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

            # Rotate OBJ mesh to USD frame (Y-up → Z-up) before sampling
            #mesh.apply_transform(ReplicatorToRoca._OBJ_TO_USD_ROTATION)

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
    def _replicator_depth_to_roca(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
        """
        Converts depth numpy files from Replicator to 16-bit depth PNGs in ROCA Rendering format.

        :param rep_data_path: Path to the folder containing distance_to_image_plane_*.npy,
               instance_segmentation files, and camera_params_*.json files.
        :param roca_data_path: ROCA data path where Rendering/scene{nr}/depth/ dirs will be created.
        :param scene_numbers: List of scene numbers as zero-padded strings.
        """

        for nr in scene_numbers:
            depth_file = os.path.join(rep_data_path, f"distance_to_image_plane_{nr}.npy")
            depth_meters = np.load(depth_file)

            # Convert meters to millimeters, clip to uint16 range
            depth_mm = np.clip(depth_meters * 1000.0, 0, 65535).astype(np.uint16)

            # Mask using instance segmentation
            seg_file = os.path.join(rep_data_path, f"instance_segmentation_{nr}.png")
            instance_ids = np.array(Image.open(seg_file))
            mapping_file = os.path.join(rep_data_path, f"instance_segmentation_mapping_{nr}.json")
            with open(mapping_file) as f:
                mapping = json.load(f)

            object_ids = set()
            for id_str, label in mapping.items():
                if "/objects/object_" in label:
                    object_ids.add(int(id_str))

            # Instance segmentation has prim paths — use it to mask
            object_mask = np.isin(instance_ids, list(object_ids))
            depth_mm[~object_mask] = 0

            output_dir = os.path.join(roca_data_path, f"Rendering/scene{nr}/depth")
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(depth_mm).save(os.path.join(output_dir, "000000.png"))

    @staticmethod
    def _replicator_instance_seg_to_roca(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
        """
        Converts instance segmentation data from Replicator to integer-valued PNGs in ROCA Rendering format.

        :param rep_data_path: Path to the folder containing instance_segmentation files and mapping JSONs.
        :param roca_data_path: ROCA data path where Rendering/scene{nr}/instance/ dirs will be created.
        :param scene_numbers: List of scene numbers as zero-padded strings.
        """

        for nr in scene_numbers:
            # Load 16-bit grayscale instance segmentation (pixel value = instance ID)
            seg_file = os.path.join(rep_data_path, f"instance_segmentation_{nr}.png")
            instance_ids = np.array(Image.open(seg_file))

            mapping_file = os.path.join(rep_data_path, f"instance_segmentation_mapping_{nr}.json")
            with open(mapping_file) as f:
                mapping = json.load(f)

            # Build mapping from replicator instance IDs to original object indices
            inst_id_to_obj_idx = {}
            for id_str, label in mapping.items():
                instance_id = int(id_str)
                if "/objects/object_" in label:
                    obj_idx = int(label.rsplit("object_", 1)[-1])
                    inst_id_to_obj_idx[instance_id] = obj_idx

            # Remap to sequential 1-indexed alignment IDs sorted by object index,
            # matching the order in full_annotations.json (visible objects sorted by index)
            sorted_obj_indices = sorted(inst_id_to_obj_idx.values())
            obj_idx_to_alignment_id = {idx: rank + 1 for rank, idx in enumerate(sorted_obj_indices)}

            id_to_mask_value = {
                inst_id: obj_idx_to_alignment_id[obj_idx]
                for inst_id, obj_idx in inst_id_to_obj_idx.items()
            }

            # Map instance IDs to ROCA mask values (unmapped IDs stay 0 = background)
            output_mask = np.zeros(instance_ids.shape, dtype=np.uint16)
            for inst_id, mask_val in id_to_mask_value.items():
                output_mask[instance_ids == inst_id] = mask_val

            out_dir = os.path.join(roca_data_path, f"Rendering/scene{nr}/instance")
            os.makedirs(out_dir, exist_ok=True)
            Image.fromarray(output_mask).save(os.path.join(out_dir, "000000.png"))

    @staticmethod
    def _world_pose_to_camera_frame(t_world: list, q_world: list, T_w2c: np.ndarray) -> tuple[list, list]:
        """
        Transforms an object's translation and quaternion from world frame to camera frame.

        :param t_world: Translation [x, y, z] in world coordinates.
        :param q_world: Quaternion [w, x, y, z] in world coordinates.
        :param T_w2c: 4x4 world-to-camera transformation matrix.
        :return: (t_cam, q_cam) — translation and quaternion in camera frame.
        """
        R_w2c = T_w2c[:3, :3]

        # Transform translation: t_cam = R_w2c * t_world + t_offset
        t_cam = (T_w2c @ np.array([*t_world, 1.0]))[:3]

        # Transform rotation: R_cam = R_w2c * R_world
        w, x, y, z = q_world
        R_world = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
        R_cam = R_w2c @ R_world

        # Convert rotation matrix back to quaternion [w, x, y, z]
        trace = np.trace(R_cam)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R_cam[2, 1] - R_cam[1, 2]) * s
            qy = (R_cam[0, 2] - R_cam[2, 0]) * s
            qz = (R_cam[1, 0] - R_cam[0, 1]) * s
        elif R_cam[0, 0] > R_cam[1, 1] and R_cam[0, 0] > R_cam[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_cam[0, 0] - R_cam[1, 1] - R_cam[2, 2])
            qw = (R_cam[2, 1] - R_cam[1, 2]) / s
            qx = 0.25 * s
            qy = (R_cam[0, 1] + R_cam[1, 0]) / s
            qz = (R_cam[0, 2] + R_cam[2, 0]) / s
        elif R_cam[1, 1] > R_cam[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_cam[1, 1] - R_cam[0, 0] - R_cam[2, 2])
            qw = (R_cam[0, 2] - R_cam[2, 0]) / s
            qx = (R_cam[0, 1] + R_cam[1, 0]) / s
            qy = 0.25 * s
            qz = (R_cam[1, 2] + R_cam[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R_cam[2, 2] - R_cam[0, 0] - R_cam[1, 1])
            qw = (R_cam[1, 0] - R_cam[0, 1]) / s
            qx = (R_cam[0, 2] + R_cam[2, 0]) / s
            qy = (R_cam[1, 2] + R_cam[2, 1]) / s
            qz = 0.25 * s

        q_cam = [float(qw), float(qx), float(qy), float(qz)]
        return t_cam.tolist(), q_cam

    @staticmethod
    def _generate_scan2cad_instances_json(output_dir: str) -> None:
        """
        Generates scan2cad_instances_train.json and scan2cad_instances_val.json in COCO format
        from full_annotations.json, instance segmentation masks, and intrinsics files.

        Object poses from full_annotations.json (world frame) are transformed to camera frame
        using the per-scene world-to-camera matrix, since the training pipeline's L_noc loss
        expects camera-frame translations and rotations.

        These files are consumed by the ROCA training pipeline's RocaDataset class.
        """

        # Load required data
        with open(os.path.join(output_dir, "Scan2CAD", "full_annotations.json")) as f:
            full_annotations = json.load(f)

        with open(os.path.join(output_dir, "metadata", "scan2cad_taxonomy_9.json")) as f:
            taxonomy = json.load(f)

        train_scenes = set(open(os.path.join(output_dir, "metadata", "scannetv2_train.txt")).read().strip().split("\n"))
        val_scenes = set(open(os.path.join(output_dir, "metadata", "scannetv2_val.txt")).read().strip().split("\n"))

        categories = [{"id": i, "name": t["name"], "shapenet": t["shapenet"]} for i, t in enumerate(taxonomy)]

        def build_instances(scene_set):
            annotations = []
            ann_id = 0

            for scene_ann in full_annotations:
                scene_id = scene_ann["id_scan"]
                if scene_id not in scene_set:
                    continue

                instance_path = os.path.join(output_dir, "Rendering", scene_id, "instance", "000000.png")
                if not os.path.exists(instance_path):
                    continue

                instance_mask = np.array(Image.open(instance_path))

                # Read 3x3 intrinsics from the 4x4 file
                intrinsics_path = os.path.join(
                    output_dir, "ScanNet25k", "tasks", "scannet_frames_25k", scene_id, "intrinsics_color.txt"
                )
                intrinsics_4x4 = np.loadtxt(intrinsics_path)
                intrinsics_3x3 = intrinsics_4x4[:3, :3].tolist()

                # Load camera-to-world pose (ScanNet convention) and invert to get world-to-camera
                pose_path = os.path.join(
                    output_dir, "ScanNet25k", "tasks", "scannet_frames_25k", scene_id, "pose", "000000.txt"
                )
                T_w2c = np.linalg.inv(np.loadtxt(pose_path))

                unique_ids = set(np.unique(instance_mask)) - {0}

                for model_idx, model in enumerate(scene_ann["aligned_models"]):
                    alignment_id = model_idx + 1
                    if alignment_id not in unique_ids:
                        continue

                    ys, xs = np.where(instance_mask == alignment_id)
                    if len(xs) == 0:
                        continue

                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    w, h = x2 - x1, y2 - y1

                    # Skip degenerate boxes
                    if w <= 0 or h <= 0:
                        continue

                    # Transform pose from world frame to camera frame
                    t_cam, q_cam = ReplicatorToRoca._world_pose_to_camera_frame(
                        model["trs"]["translation"], model["trs"]["rotation"], T_w2c
                    )

                    annotations.append({
                        "id": ann_id,
                        "scene_id": scene_id,
                        "category_id": int(model["catid_cad"]),
                        "alignment_id": alignment_id,
                        "bbox": [x1, y1, w, h],
                        "t": t_cam,
                        "q": q_cam,
                        "intrinsics": intrinsics_3x3,
                    })
                    ann_id += 1

            return {"categories": categories, "images": [], "annotations": annotations}

        dataset_dir = os.path.join(output_dir, "Dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        for split, scene_set in [("train", train_scenes), ("val", val_scenes)]:
            data = build_instances(scene_set)
            with open(os.path.join(dataset_dir, f"scan2cad_instances_{split}.json"), "w") as f:
                json.dump(data, f)
            print(f"  {split}: {len(data['annotations'])} annotations")

    @staticmethod
    def _create_images_symlink(output_dir: str) -> None:
        """Creates an 'Images' symlink pointing to 'ScanNet25k' for training compatibility."""
        link_path = os.path.join(output_dir, "Images")
        if not os.path.exists(link_path):
            os.symlink("ScanNet25k", link_path)

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
        print("Converting depth images...")
        ReplicatorToRoca._replicator_depth_to_roca(replicator_data_dir, output_dir, scene_numbers)
        print("Converting instance segmentation masks...")
        ReplicatorToRoca._replicator_instance_seg_to_roca(replicator_data_dir, output_dir, scene_numbers)
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
        print("Generating scan2cad_instances JSON files...")
        ReplicatorToRoca._generate_scan2cad_instances_json(output_dir)
        print("Creating Images symlink...")
        ReplicatorToRoca._create_images_symlink(output_dir)

        print("Done!")
