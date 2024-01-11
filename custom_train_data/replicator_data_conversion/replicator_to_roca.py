import shutil
import sys
import os
import glob
import json
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def get_scene_nrs(rep_data_path: str) -> list[str]:
    """
    Get scene numbers from replicator dataset

    Args:
    - rep_data_path : Path to the replicator dataset containing camera_params_*.json files

    Returns:
    list of scene numbers (list[str])
    """

    # Find all JSON files matching the pattern camera_params_*.json
    json_files = glob.glob(os.path.join(rep_data_path, "camera_params_*.json"))

    # Create and return list of scene numbers
    return [os.path.basename(json_file).split('_')[2].split('.')[0] for json_file in json_files]


def replicator_intrinsics_to_scannet(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
    """
    Generate intrinsics_color.txt files from camera parameters JSON files.

    Args:
    - rep_data_path (str): Path to the folder containing camera_params_*.json files.
    - roca_data_path (str): Path to the output folder where ScanNet25k/tasks/scannet_frames_25k/scene{NR}/intrinsics_color.txt files will be saved.
    - scene_numbers (list[str]): List of scene numbers

    Returns:
    None
    """

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

        # Prepare the output directory
        output_dir = os.path.join(roca_data_path, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}")
        os.makedirs(output_dir, exist_ok=True)

        # Write to intrinsics_color.txt
        intrinsics_file_path = os.path.join(output_dir, "intrinsics_color.txt")
        with open(intrinsics_file_path, 'w') as intrinsics_file:
            intrinsics_file.write(f"{f_x} 0 {c_x} 0\n0 {f_y} {c_y} 0\n0 0 1 0\n0 0 0 1\n")


def replicator_image_to_scannet(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
    """
    Copy and convert images from PNG to JPG format and rename them.

    Args:
    - rep_data_path (str): Path to the folder containing rgb_{nr}.png files.
    - roca_data_path (str): Path to the output folder where images will be saved in the format ScanNet25k/tasks/scannet_frames_25k/scene{NR}/color/000000.jpg.
    - scene_numbers (list[str]): List of scene numbers

    Returns:
    None
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


def copy_obj_files(roca_data_path: str, obj_path: str) -> None:
    """
    Copy .obj files to the roca data path respecting the ShapeNet data structure.

    Args:
    - rep_data_path (str): Not used in this function, but included for consistency.
    - roca_data_path (str): ROCA data path where .obj files will be copied to.
    - obj_path (str): Path to the folder containing [ModelName].obj files.

    Returns:
    None
    """

    # Find all .obj files in the obj_path
    obj_files = glob.glob(os.path.join(obj_path, "*.obj"))

    for obj_file in obj_files:
        # Extract the model name from the file name
        model_name = os.path.basename(obj_file).split('.')[0]

        # Prepare the output directory
        output_dir = os.path.join(roca_data_path, f"ShapeNetCore.v2/{model_name}/0/models")
        os.makedirs(output_dir, exist_ok=True)

        # Set the new file path
        new_file_path = os.path.join(output_dir, "model_normalized.obj")

        # Copy the .obj file
        shutil.copy(obj_file, new_file_path)


def generate_full_annotations_json(rep_data_path: str, roca_data_path: str, scene_numbers: list[str]) -> None:
    """
    Generate a full_annotations.json file from world_pose_visible_objects_[nr].json files.

    Args:
    - rep_data_path (str): Path to the folder containing world_pose_visible_objects_[nr].json files.
    - roca_data_path (str): ROCA data path where the Scan2CAD/full_annotations.json will be saved.
    - scene_numbers (list[str]): List of scene numbers as strings.

    Returns:
    None
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
            # Process obj data
            aligned_models.append({
                "sym": "__SYM_NONE",  # Assuming symmetry as none for all models
                "catid_cad": obj["obj_path"].split("/")[-1].split(".")[0],
                "id_cad": "0",
                "trs": {
                    "translation": [obj['pose']['position']['x'],
                                    obj['pose']['position']['y'],
                                    obj['pose']['position']['z']],
                    "rotation": [obj['pose']['orientation']['w'],
                                 obj['pose']['orientation']['x'],
                                 obj['pose']['orientation']['y'],
                                 obj['pose']['orientation']['z']],
                    "scale": [0.001, 0.001, 0.001]
                }
            })

        scene_data["aligned_models"] = aligned_models
        annotations.append(scene_data)

    # Saving the annotations to full_annotations.json
    output_path = os.path.join(roca_data_path, "Scan2CAD", "full_annotations.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as output_file:
        json.dump(annotations, output_file, indent=4)


def generate_labels_from_objs(obj_path: str) -> list[str]:
    """
    Generate labels from all obj files in obj_path

    Args:
    - obj_path (str): path to obj files

    Returns:
    List of labels (list[str])
    """
    # Find all .obj files in the obj_path
    obj_files = glob.glob(os.path.join(obj_path, "*.obj"))
    # Remove file type from filename
    return [os.path.basename(obj_path).split(".")[0] for obj_path in obj_files]


def generate_metadata_taxonomy_9(roca_metadata_path: str, labels: list[str]) -> None:
    """
    Generates a taxonomy JSON file for ROCA metadata.

    This function creates a taxonomy file named 'scan2cad_taxonomy_9.json' in the specified directory.
    The taxonomy file contains a list of dictionaries, each representing a label from the provided labels list,
    with each label being mapped to its corresponding ShapeNet label.

    Args:
    - roca_metadata_path (str): The file path where the taxonomy file will be saved.
    - labels (list[str]): A list of labels to include in the taxonomy file.

    Returns:
    None
    """

    taxonomy = [{"name": label, "shapenet": label} for label in labels]
    json.dump(taxonomy, open(os.path.join(roca_metadata_path, "scan2cad_taxonomy_9.json"), "w"), indent=4)


def write_list_to_txt(output_path: str, string_list: list[str]) -> None:
    """
    Writes a list of strings to a text file, with each string on a new line.

    Args:
    - output_path (str): The file path where the text file will be saved.
    - string_list (list[str]): A list of strings to be written to the file.

    Returns:
    None
    """

    with open(output_path, "w") as f:
        f.write("\n".join(string_list))


def generate_metadata_label_id_files(roca_metadata_path: str, labels: list[str]) -> None:
    """
    Generates text files containing label IDs for ROCA metadata.

    This function creates two files: 'labelids_all.txt' and 'labelids.txt' in the specified directory.
    Both files contain the same labels, each on a new line, as provided in the labels list prepended by an ID.

    Args:
    - roca_metadata_path (str): The file path where the label ID files will be saved.
    - labels (list[str]): A list of labels to include in the label ID files.

    Returns:
    None
    """
    labels_with_id = [f"{i + 1} {labels[i]}" for i in range(len(labels))]
    write_list_to_txt(os.path.join(roca_metadata_path, "labelids_all.txt"), labels_with_id)
    write_list_to_txt(os.path.join(roca_metadata_path, "labelids.txt"), labels_with_id)


def generate_metadata_train_val_files(roca_metadata_path: str, scene_numbers: list[str]) -> None:
    """
    Generates train and validation split files for ROCA metadata.

    This function creates three files: 'scannetv2_train.txt', 'scannetv2_val.txt', and 'val_images.txt'
    in the specified directory. The train and validation splits are made from the provided scene numbers,
    with 20% of the scenes being used for validation. The 'val_images.txt' file contains one image per scene
    in the validation split.

    Args:
    - roca_metadata_path (str): The file path where the train and validation files will be saved.
    - scene_numbers (list[str]): A list of scene numbers to be split into train and validation sets.

    Returns:
    None
    """

    # Create train and val split with 20% validation
    scene_names = [f"scene{scene_number}" for scene_number in scene_numbers]
    train_scenes, val_scenes = train_test_split(scene_names, test_size=0.2)

    # Write scenes
    write_list_to_txt(os.path.join(roca_metadata_path, "scannetv2_train.txt"), train_scenes)
    write_list_to_txt(os.path.join(roca_metadata_path, "scannetv2_val.txt"), val_scenes)

    # Write val images
    val_images = [scene_name + " 0" for scene_name in val_scenes]  # Since there is only one picture per scene, use this one
    write_list_to_txt(os.path.join(roca_metadata_path, "val_images.txt"), val_images)


def replicator_cam_pose_to_scannet(replicator_dir: str, roca_dataset_dir: str, scene_numbers) -> None:
    """
    Reads camera parameters from JSON files, calculates the world to ros camera view transformation matrix,
    and saves it in a text file.

    Args:
    - rep_data_path (str): Path to the folder containing camera_params_[nr].json files.
    - roca_data_path (str): Path where the inverted camera view transform will be saved.
    - scene_numbers (list[str]): List of scene numbers as strings.

    Returns:
    None
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
        test = ros_camera_view_to_world @ np.array([-1, -0.5, 0, 1])  # sollte [-1, 0.5, 5] sein

        # According to ROCA code in render.py, T^W_R needs to be saved
        world_to_ros_camera_view = np.linalg.inv(ros_camera_view_to_world)

        # Prepare the output directory
        output_dir = os.path.join(roca_dataset_dir, f"ScanNet25k/tasks/scannet_frames_25k/scene{nr}/pose")
        os.makedirs(output_dir, exist_ok=True)

        output_file_path = os.path.join(output_dir, "000000.txt")
        with open(output_file_path, 'w') as output_file:
            for row in world_to_ros_camera_view:
                output_file.write(" ".join([f"{val}" for val in row]) + "\n")


def main(args: list[str]) -> None:
    # Create argument parser
    parser = argparse.ArgumentParser(description="Converts the generated training data from NVIDIA Replicator to the format required by ROCA")
    parser.add_argument("--rep_dir", help="Input directory containing the files in the Replicator format", required=True)
    parser.add_argument("--roca_dataset_dir", help="Directory that ROCA will use for training data generation", required=True)
    parser.add_argument("--roca_metadata_dir", help="Metadata directory in the ROCA GitHub-Repository", required=True)
    parser.add_argument("--obj_dir", help="Directory with all CAD Models in OBJ format", required=True)

    args = parser.parse_args(args)

    # Test if rep_dir and obj_dir exist
    if not os.path.isdir(args.rep_dir):
        print(f"The NVIDIA replicator directory {args.rep_dir} does not exist. Existing...")
        exit(-1)
    if not os.path.isdir(args.obj_dir):
        print(f"The OBJ model directory {args.rep_dir} does not exist. Existing...")
        exit(-1)

    # Warn user about new test and val splits
    print("Converting the generated training data from NVIDIA Replicator to ROCA format...")
    print("Warning! This will create a new randomized training and validation data split!")
    print("You may want to backup and restore 'val_images.txt', 'scannetv2_val.txt, 'scannetv2_train.txt' in your ROCA metadata directory.")
    input("Press any key to continue...")

    # Create roca_dataset_dir and roca_metadata_dir if necessary
    os.makedirs(args.roca_dataset_dir, exist_ok=True)
    os.makedirs(args.roca_metadata_dir, exist_ok=True)

    # Convert training data
    replicator_dir = args.rep_dir
    obj_dir = args.obj_dir
    roca_dataset_dir = args.roca_dataset_dir
    roca_metadata_dir = args.roca_metadata_dir

    scene_numbers = get_scene_nrs(replicator_dir)
    print("Converting camera intrinsics...")
    replicator_intrinsics_to_scannet(replicator_dir, roca_dataset_dir, scene_numbers)
    print("Converting images...")
    replicator_image_to_scannet(replicator_dir, roca_dataset_dir, scene_numbers)
    print("Converting camera to world transformations...")
    replicator_cam_pose_to_scannet(replicator_dir, roca_dataset_dir, scene_numbers)
    print("Copying CAD models...")
    copy_obj_files(roca_dataset_dir, obj_dir)
    print("Generating full_annotations.json...")
    generate_full_annotations_json(replicator_dir, roca_dataset_dir, scene_numbers)
    print("Generating ROCA metadata files...")
    labels = generate_labels_from_objs(obj_dir)
    generate_metadata_taxonomy_9(roca_metadata_dir, labels)
    generate_metadata_label_id_files(roca_metadata_dir, labels)
    generate_metadata_train_val_files(roca_metadata_dir, scene_numbers)

    print("Done!")


if __name__ == '__main__':
    main(sys.argv[1:])
