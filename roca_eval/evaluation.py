import numpy as np
import quaternion
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import scipy.stats as stats


from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import os
import glob
import json
import math


# Zuerst für cluttered
# Pro Netz jede Szene
    # Fehler in Translation
    # Fehler in Rotation
    # Fehler in Objekttyp

# Durchschnitt plus STD über alle Szenen pro Netz
# Hoffentlich ergibt sich dann ein Favorit bis max 3 Favs

# Durchschnitt Fehler pro Szene
# Fehler über Frame-Verlauf

# Tabelle mit Mapping, welches Objekt wurde korrekt erkannt, welches nicht

# Anzahl fehlender Objekte mit Verdeckungsgrad

# Dann nochmal für uncluttered


def calc_per_axis_position_difference(ground_truth_object, found_object):
    return [ground_truth_object["pose"]["position"]["x"] - found_object["translation"][0],
            ground_truth_object["pose"]["position"]["y"] - found_object["translation"][1],
            ground_truth_object["pose"]["position"]["z"] - found_object["translation"][2]]


def calc_distance(ground_truth_object, found_object):
    return np.linalg.norm(calc_per_axis_position_difference(ground_truth_object, found_object))


def calc_per_axis_scale_difference(found_object):
    return [1 - found_object["scale"][0],
            1 - found_object["scale"][1],
            1 - found_object["scale"][2]]


def calc_scale_distance(found_object):
    return np.linalg.norm(calc_per_axis_scale_difference(found_object))


def calc_rotation_distance(ground_truth_object, found_object):
    # Use metric 4 from Huynh - Metrics for 3D Rotations because it has no restrictions on angles and returns values in range [0,1]
    gt_quat = np.quaternion(ground_truth_object["pose"]["orientation"]["w"],
                            ground_truth_object["pose"]["orientation"]["x"],
                            ground_truth_object["pose"]["orientation"]["y"],
                            ground_truth_object["pose"]["orientation"]["z"])
    found_quat = np.quaternion(found_object["rotation"][0],
                               found_object["rotation"][1],
                               found_object["rotation"][2],
                               found_object["rotation"][3])

    dot_product = np.abs(found_quat.w * gt_quat.w + found_quat.x * gt_quat.x + found_quat.y * gt_quat.y + found_quat.z * gt_quat.z)
    return 1 - dot_product


def ceil_to_pos(x, pos):
    if math.isnan(x):
        return 1
    ceil_pot = 10.0 ** pos
    return math.ceil(x / ceil_pot) * ceil_pot


def get_world_to_camera_transform(image_number, dataset_path):
    json_file_path = os.path.join(dataset_path, f"camera_params_{image_number}.json")
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
    return world_to_ros_camera_view


def main(eval_dataset_path: str, output_dir: str):
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Create dict for num_objects to inference time
    num_found_objects_to_inference_time = {}
    all_dataset_results = {}

    for dataset_type_name in ["cluttered", "uncluttered"]:
        os.makedirs(os.path.join(output_dir, dataset_type_name), exist_ok=True)
        dataset_type_dataset_paths = sorted(glob.glob(os.path.join(eval_dataset_path, "converted", dataset_type_name, "*")))
        dataset_type_dataset_paths = [dataset_type_dataset_path for dataset_type_dataset_path in dataset_type_dataset_paths if not dataset_type_dataset_path.endswith("glass") and not dataset_type_dataset_path.endswith("default") and not "4" in os.path.basename(dataset_type_dataset_path) and not "5" in os.path.basename(dataset_type_dataset_path)]# and "ma_small_object_2_conveyor" in os.path.basename(dataset_type_dataset_path)]
        per_dataset_results = {}
        for dataset_type_dataset_path in dataset_type_dataset_paths:
            dataset_name = os.path.basename(dataset_type_dataset_path)
            per_dataset_results[dataset_name] = {}
            neural_net_paths = sorted(glob.glob(os.path.join(dataset_type_dataset_path, "ReplicatorToRocaEval", "eval_raw_data", "*")))
            for neural_net_path in neural_net_paths:
                print("Processing", neural_net_path)
                neural_net_name = os.path.basename(neural_net_path)
                per_dataset_results[dataset_name][neural_net_name] = {}
                per_dataset_results[dataset_name][neural_net_name]["per_frame_results"] = []

                # Load results for dataset and neural net
                neural_net_json = os.path.join(neural_net_path, "eval_raw_data.json")
                with open(neural_net_json, "r") as f:
                    neural_net_results = json.load(f)

                # Create list to store data for pandas df for dataset
                dataset_neural_net_df_data = []

                # Calc per frame stats
                for frame_result in neural_net_results["per_frame_results"]:
                    # Load ground truth data
                    image_name: str = frame_result["image"]
                    image_number = image_name.removesuffix(".jpg").removeprefix("image_")
                    ground_truth_visible_objects_data_path = os.path.join(dataset_type_dataset_path, "ReplicatorToRocaEval", f"world_pose_visible_objects_{image_number}.json")
                    with open(ground_truth_visible_objects_data_path, "r") as f:
                        ground_truth_visible_objects = json.load(f)

                    # Get number of objects
                    ground_truth_visible_objects_count = len(ground_truth_visible_objects)
                    found_object_count = len(frame_result["instances"])
                    if found_object_count > ground_truth_visible_objects_count:
                        print(f" Warning! Num found objects greater than ground truth ({found_object_count} > {ground_truth_visible_objects_count})")
                        print(f"Results from: '{neural_net_name}' - frame: {image_number}")

                    # Calc and store inference time
                    inference_time = frame_result["inference_end_time"] - frame_result["inference_start_time"]
                    if found_object_count in num_found_objects_to_inference_time.keys():
                        num_found_objects_to_inference_time[found_object_count].append(inference_time)
                    else:
                        num_found_objects_to_inference_time[found_object_count] = [inference_time]

                    # transform found objects form camera to world frame
                    world_to_camera_transform = get_world_to_camera_transform(image_number, os.path.join(dataset_type_dataset_path, "ReplicatorToRocaEval"))
                    for found_object in frame_result["instances"]:
                        # transform object origin
                        detected_object_origin_camera = np.array([found_object["translation"][0], found_object["translation"][1], found_object["translation"][2], 1])
                        detected_object_origin_world = world_to_camera_transform @ detected_object_origin_camera
                        detected_object_origin_camera /= detected_object_origin_world[3]
                        found_object["translation"] = detected_object_origin_world[:3]

                        # transform object rotation
                        world_to_camera_rotation = world_to_camera_transform[:3, :3]
                        camera_to_obj_quat = np.quaternion(found_object["rotation"][0],
                                                   found_object["rotation"][1],
                                                   found_object["rotation"][2],
                                                   found_object["rotation"][3])
                        camera_to_obj_rotation = quaternion.as_rotation_matrix(camera_to_obj_quat)
                        world_to_obj_rotation = world_to_camera_rotation @ camera_to_obj_rotation
                        world_to_obj_quat = quaternion.from_rotation_matrix(world_to_obj_rotation)
                        found_object["rotation"] = [world_to_obj_quat.w, world_to_obj_quat.x, world_to_obj_quat.y, world_to_obj_quat.z]

                    # Find closest objects with ground_truth_objects as reference
                    # TODO threshold?
                    found_to_gt_objects = []
                    for found_obj_idx, found_obj in enumerate(frame_result["instances"]):
                        # Find matching object correlation between found_obj and gt_object
                        min_distance = float("inf")
                        min_distance_gt_obj = None
                        min_distance_gt_obj_idx = -1
                        for gt_obj_idx, gt_obj in enumerate(ground_truth_visible_objects):
                            distance = calc_distance(gt_obj, found_obj)
                            if distance < min_distance:
                                min_distance = distance
                                min_distance_gt_obj = gt_obj
                                min_distance_gt_obj_idx = gt_obj_idx

                        # Remove found_obj from list to speed up performance for further search and prevent double assignments
                        del ground_truth_visible_objects[min_distance_gt_obj_idx]

                        # Store assignment
                        found_to_gt_objects.append((found_obj, min_distance_gt_obj))

                        if len(ground_truth_visible_objects) == 0:
                            break  # no objects to find correspondence, because more objects were detected than in image

                    # Create per object statistics
                    # Distance
                    position_difference_per_axis = [calc_per_axis_position_difference(gt_obj, found_obj) for found_obj, gt_obj in found_to_gt_objects]
                    distance_errors = [calc_distance(gt_obj, found_obj) for found_obj, gt_obj in found_to_gt_objects]
                    distance_error_mean = np.mean(distance_errors) if len(found_to_gt_objects) > 0 else None
                    distance_error_std = np.std(distance_errors) if len(found_to_gt_objects) > 0 else None
                    distance_error_min = np.min(distance_errors) if len(found_to_gt_objects) > 0 else None
                    distance_error_max = np.max(distance_errors) if len(found_to_gt_objects) > 0 else None

                    # Rotation error
                    rotation_errors = [calc_rotation_distance(gt_obj, found_obj) for found_obj, gt_obj in found_to_gt_objects]
                    rotation_error_mean = np.mean(rotation_errors) if len(found_to_gt_objects) > 0 else None
                    rotation_error_std = np.std(rotation_errors) if len(found_to_gt_objects) > 0 else None
                    rotation_error_min = np.min(rotation_errors) if len(found_to_gt_objects) > 0 else None
                    rotation_error_max = np.max(rotation_errors) if len(found_to_gt_objects) > 0 else None

                    # Scale error
                    scale_difference_per_axis = [calc_per_axis_scale_difference(found_obj) for found_obj, _ in found_to_gt_objects]
                    scale_errors = [calc_scale_distance(found_obj) for found_obj, _ in found_to_gt_objects]
                    scale_error_mean = np.mean(scale_errors) if len(found_to_gt_objects) > 0 else None
                    scale_error_std = np.std(scale_errors) if len(found_to_gt_objects) > 0 else None
                    scale_error_min = np.min(scale_errors) if len(found_to_gt_objects) > 0 else None
                    scale_error_max = np.max(scale_errors) if len(found_to_gt_objects) > 0 else None

                    # Occlusion ratio of undetected objects
                    occlusion_ratio_non_detected_objects = [gt_obj["occlusion_ratio"] for gt_obj in ground_truth_visible_objects]
                    undetected_occlusion_mean = np.mean(occlusion_ratio_non_detected_objects) if len(occlusion_ratio_non_detected_objects) > 0 else None
                    undetected_occlusion_std = np.std(occlusion_ratio_non_detected_objects) if len(occlusion_ratio_non_detected_objects) > 0 else None
                    undetected_occlusion_min = np.min(occlusion_ratio_non_detected_objects) if len(occlusion_ratio_non_detected_objects) > 0 else None
                    undetected_occlusion_max = np.max(occlusion_ratio_non_detected_objects) if len(occlusion_ratio_non_detected_objects) > 0 else None

                    # Prepare data for object classification / object type
                    label_to_predicted_label = {}
                    for found_obj, gt_obj in found_to_gt_objects:
                        correct_label = gt_obj["semantic_labels"]["class"]
                        predicated_label = found_obj["semantic_label"]
                        if correct_label in label_to_predicted_label.keys():
                            label_to_predicted_label[correct_label].append(predicated_label)
                        else:
                            label_to_predicted_label[correct_label] = [predicated_label]

                    # Calc correct and incorrect classifications
                    # Initialize variables to count correct and incorrect classifications
                    correct_classifications = 0
                    incorrect_classifications = 0
                    # Iterate through the dictionary to count correct and incorrect classifications
                    for key, predictions in label_to_predicted_label.items():
                        for prediction in predictions:
                            if prediction == key:
                                correct_classifications += 1
                            else:
                                incorrect_classifications += 1


                    # Store per frame results
                    per_dataset_results[dataset_name][neural_net_name]["per_frame_results"].append({
                        "ground_truth_object_count": ground_truth_visible_objects,
                        "found_object_count": found_object_count,
                        "inference_time": inference_time,
                        "position_difference_per_axis": position_difference_per_axis,
                        "distance_errors": distance_errors,
                        "rotation_errors": rotation_errors,
                        "scale_difference_per_axis": scale_difference_per_axis,
                        "scale_errors": scale_errors,
                        "occlusion_of_non_detected_objects": occlusion_ratio_non_detected_objects,
                        "label_to_predicted_label": label_to_predicted_label,
                        "correct_classifications": correct_classifications,
                        "incorrect_classifications": incorrect_classifications
                    })

                    dataset_neural_net_df_data.append([ground_truth_visible_objects_count, found_object_count, inference_time,
                                                    distance_error_mean, distance_error_std, distance_error_min, distance_error_max,
                                                    rotation_error_mean, rotation_error_std, rotation_error_min, rotation_error_max,
                                                    scale_error_mean, scale_error_std, scale_error_min, scale_error_max,
                                                    undetected_occlusion_mean, undetected_occlusion_std, undetected_occlusion_min, undetected_occlusion_max,
                                                    correct_classifications, incorrect_classifications])

                # Store dataset data for this neural net model in csv
                column_names = ["Object Count", "Predicted Object Count", "Inference Time (s)",
                                                   "Mean Distance Error (m)", "STD Distance Error (m)", "Min Distance Error (m)", "Max Distance Error (m)",
                                                   "Mean Rotation Error", "STD Rotation Error", "Min Rotation Error", "Max Rotation Error",
                                                   "Mean Scale Error", "STD Scale Error", "Min Scale Error", "Max Scale Error",
                                                   "Undetected Mean Occlusion Ratio", "Undetected STD Occlusion Ratio", "Undetected Min Occlusion Ratio", "Undetected Max Occlusion Ratio",
                                                   "Correct Classification Count", "Incorrect Classification Count"]
                dataset_neural_net_df = pd.DataFrame(dataset_neural_net_df_data, columns=column_names)
                out_dir = os.path.join(output_dir, dataset_type_name, dataset_name, neural_net_name)
                os.makedirs(out_dir, exist_ok=True)
                csv_path = os.path.join(out_dir, f"{dataset_name}-{neural_net_name}.csv")
                dataset_neural_net_df.to_csv(csv_path, index=False, sep=";")

                # Save dataframe in results
                per_dataset_results[dataset_name][neural_net_name]["dataframe"] = dataset_neural_net_df

                # Create per neural net plots
                # Inference Time
                plt.figure(dpi=300)
                plt.title(f"Inferenz-Zeit pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                plt.plot(range(len(neural_net_results["per_frame_results"])), dataset_neural_net_df["Inference Time (s)"])
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Inferenz-Zeit [s]")
                plt.grid(axis="y")
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "inference_time.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()


                # Found Objects per Frame
                plt.figure(dpi=300)
                plt.title(f"Anzahl detektierte Objekte pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                plt.plot(range(len(neural_net_results["per_frame_results"])), dataset_neural_net_df["Object Count"],
                         label="Anzahl sichtbarer Objekte")
                plt.plot(range(len(neural_net_results["per_frame_results"])), dataset_neural_net_df["Predicted Object Count"],
                         label="Anzahl detektierter Objekte")
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Anzahl Objekte")
                plt.legend()
                plt.ylim([0, ceil_to_pos(np.max(dataset_neural_net_df["Object Count"]), 1)])
                plt.grid(axis="y")
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "detected_objects.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()


                # Distance error
                distance_errors = dataset_neural_net_df["Mean Distance Error (m)"]
                distance_std = dataset_neural_net_df["STD Distance Error (m)"]
                frames = range(len(neural_net_results["per_frame_results"]))
                plt.figure(dpi=300)
                plt.title(f"Distanzfehler pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                plt.plot(frames, distance_errors, label="Mittlerer Distanzfehler")
                plt.fill_between(frames, distance_errors - distance_std, distance_errors + distance_std, alpha=0.2)
                plt.plot(frames, dataset_neural_net_df["Max Distance Error (m)"], label="Maximaler Distanzfehler")
                plt.plot(frames, dataset_neural_net_df["Min Distance Error (m)"], label="Minimaler Distanzfehler")
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Distanzfehler [m]")
                plt.legend()
                #plt.ylim([0, ceil_to_pos(np.max(distance_errors), -1)])
                plt.grid(axis="y")
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "distance_error.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()

                del distance_errors, distance_std


                # Rotation error
                rotation_errors = dataset_neural_net_df["Mean Rotation Error"]
                rotation_std = dataset_neural_net_df["STD Rotation Error"]
                frames = range(len(neural_net_results["per_frame_results"]))
                plt.figure(dpi=300)
                plt.title(f"Rotationsfehler pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                plt.plot(frames, rotation_errors, label="Mittlerer Rotationsfehler")
                plt.fill_between(frames, rotation_errors - rotation_std, rotation_errors + rotation_std, alpha=0.2)
                plt.plot(frames, dataset_neural_net_df["Max Rotation Error"], label="Maximaler Rotationsfehler")
                plt.plot(frames, dataset_neural_net_df["Min Rotation Error"], label="Minimaler Rotationsfehler")
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Rotationsfehler")
                plt.legend()
                # plt.ylim([0, ceil_to_pos(np.max(distance_errors), -1)])
                plt.grid(axis="y")
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "rotation_error.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()

                del rotation_errors, rotation_std


                # Scale error
                scale_errors = dataset_neural_net_df["Mean Scale Error"]
                scale_std = dataset_neural_net_df["STD Scale Error"]
                frames = range(len(neural_net_results["per_frame_results"]))
                plt.figure(dpi=300)
                plt.title(f"Skalierungsfehler pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                plt.plot(frames, scale_errors, label="Mittlerer Skalierungsfehler")
                plt.fill_between(frames, scale_errors - scale_std, scale_errors + scale_std, alpha=0.2)
                plt.plot(frames, dataset_neural_net_df["Max Scale Error"], label="Maximaler Skalierungsfehler")
                plt.plot(frames, dataset_neural_net_df["Min Scale Error"], label="Minimaler Skalierungsfehler")
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Skalierungsfehler")
                plt.legend()
                # plt.ylim([0, ceil_to_pos(np.max(distance_errors), -1)])
                plt.grid(axis="y")
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "scale_error.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()

                del scale_errors, scale_std


                # Occlusion of undetected objects
                occlusion_ratio = dataset_neural_net_df["Undetected Mean Occlusion Ratio"]
                occlusion_std = dataset_neural_net_df["Undetected STD Occlusion Ratio"]
                frames = range(len(neural_net_results["per_frame_results"]))
                plt.figure(dpi=300)
                plt.title(f"Verdeckungsgrad unentdeckter Objekte pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                plt.plot(frames, occlusion_ratio, label="Mittlerer Verdeckungsgrad")
                plt.fill_between(frames, occlusion_ratio - occlusion_std, occlusion_ratio + occlusion_std, alpha=0.2)
                plt.plot(frames, dataset_neural_net_df["Undetected Max Occlusion Ratio"], label="Maximaler Verdeckungsgrad")
                plt.plot(frames, dataset_neural_net_df["Undetected Min Occlusion Ratio"], label="Minimaler Verdeckungsgrad")
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Verdeckungsgrad")
                plt.legend()
                plt.ylim([-0.05, 1.2])
                plt.grid(axis="y")
                plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "occlusion-undetected-objects.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()

                del occlusion_ratio, occlusion_std


                # Correct and incorrect classifications count
                correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
                incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
                frames = range(len(neural_net_results["per_frame_results"]))
                plt.figure(dpi=300)
                plt.title(f"Korrekte und inkorrekte Klassifikationen pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                plt.plot(frames, correct_classifications_per_frame, label="Korrekte Klassifikationen")
                plt.plot(frames, incorrect_classifications_per_frame, label="Inkorrekte Klassifikationen")
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Anzahl Klassifikationen")
                plt.legend()
                plt.ylim([0, ceil_to_pos(np.max(correct_classifications_per_frame), 1)])
                plt.grid(axis="y")
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "classification_counts.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()


                # Correct and incorrect classifications ratio
                plt.figure(dpi=300)
                plt.title(f"Anteil korrekte und inkorrekte Klassifikationen pro Frame\n\n"
                          f"Datensatz: {dataset_name}\n"
                          f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
                correct_classification_ratio = correct_classifications_per_frame / (correct_classifications_per_frame + incorrect_classifications_per_frame)
                incorrect_classification_ratio = 1 - correct_classification_ratio
                plt.fill_between(frames, 0, correct_classification_ratio, label="Korrekte Klassifikationen")
                #plt.fill_between(frames, correct_classification_ratio, 1, label="Inkorrekte Klassifikationen")
                plt.fill_between(frames, 0, incorrect_classification_ratio, label="Inkorrekte Klassifikationen")
                plt.xlabel("Frame-Nummer")
                plt.ylabel("Anteil Klassifikationen")
                plt.legend()
                plt.ylim([-0.05, 1.2])
                plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
                plt.grid(axis="y")
                plt.tight_layout()

                plot_path = os.path.join(out_dir, "classification_ratio.pdf")
                plt.savefig(plot_path)
                #plt.show()
                plt.close()

                del correct_classifications_per_frame, incorrect_classifications_per_frame

        # Create plots
        # Per dataset / per object type plots
        print("Creating per dataset plots...")
        for dataset_name, dataset_result in per_dataset_results.items():
            print("Dataset:", dataset_name)
            out_dir = os.path.join(output_dir, dataset_type_name, dataset_name)
            # Inference Time
            plt.figure(dpi=300)
            plt.title(f"Inferenz-Zeit pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                plt.plot(range(len(neural_net_results["per_frame_results"])), dataset_neural_net_df["Inference Time (s)"], label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Inferenz-Zeit [s]")
            plt.legend()
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "inference_time.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            # Found Objects per Frame
            plt.figure(dpi=300)
            plt.title(f"Anzahl detektierte Objekte pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            num_objects_per_frame = next(iter(dataset_result.values()))["dataframe"]["Object Count"]
            plt.plot(range(len(neural_net_results["per_frame_results"])), num_objects_per_frame,
                     label="Anzahl sichtbarer Objekte")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                plt.plot(range(len(neural_net_results["per_frame_results"])), dataset_neural_net_df["Predicted Object Count"],
                     label=f"Detektierte Objekte ({neural_net_name})")
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Anzahl Objekte")
            plt.legend()
            plt.ylim([0, ceil_to_pos(np.max(num_objects_per_frame), 1)])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "detected_objects.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            # Distance error
            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Maximaler Distanzfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            all_max_distance_errors = []
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                max_distance_errors = dataset_neural_net_df["Max Distance Error (m)"]
                all_max_distance_errors.extend(max_distance_errors)
                plt.plot(frames, max_distance_errors, label=neural_net_name.removesuffix("_Augmentation"))
            distance_plots_y_lim_upper = ceil_to_pos(np.max(all_max_distance_errors), 0)
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Distanzfehler [m]")
            plt.legend()
            plt.ylim([0, distance_plots_y_lim_upper])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "distance_error_max.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Minimaler Distanzfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                min_distance_errors = dataset_neural_net_df["Min Distance Error (m)"]
                plt.plot(frames, min_distance_errors, label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Distanzfehler [m]")
            plt.legend()
            plt.ylim([0, distance_plots_y_lim_upper])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "distance_error_min.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Mittlerer Distanzfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                distance_errors = dataset_neural_net_df["Mean Distance Error (m)"]
                distance_std = dataset_neural_net_df["STD Distance Error (m)"]
                plt.plot(frames, distance_errors, label=f"{neural_net_name}")
                plt.fill_between(frames, distance_errors - distance_std, distance_errors + distance_std, alpha=0.2)
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Distanzfehler [m]")
            plt.legend()
            plt.ylim([0, distance_plots_y_lim_upper])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "distance_error_mean.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            # TODO: wenn STD sich nicht sinnvoll darstellen lässt, tabelle erzeugen mit Zeile von STD für jedes Netz und diese abspeichern

            del distance_errors, distance_std, all_max_distance_errors, distance_plots_y_lim_upper

            # Rotation error
            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Mittlerer Rotationsfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                rotation_errors = dataset_neural_net_df["Mean Rotation Error"]
                rotation_std = dataset_neural_net_df["STD Rotation Error"]
                plt.plot(frames, rotation_errors, label=f"{neural_net_name}")
                plt.fill_between(frames, rotation_errors - rotation_std, rotation_errors + rotation_std, alpha=0.2)
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Rotationsfehler")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "rotation_error_mean.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Minimaler Rotationsfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                plt.plot(frames, dataset_neural_net_df["Min Rotation Error"], label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Rotationsfehler")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "rotation_error_min.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Maximaler Rotationsfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                plt.plot(frames, dataset_neural_net_df["Max Rotation Error"], label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Rotationsfehler")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "rotation_error_max.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            del rotation_errors, rotation_std

            # Scale error
            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Maximaler Skalierungsfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            all_max_scale_errors = []
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                max_scale_errors = dataset_neural_net_df["Max Scale Error"]
                all_max_scale_errors.extend(max_scale_errors)
                plt.plot(frames, max_scale_errors, label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Skalierungsfehler")
            plt.legend()
            scale_plots_y_lim_upper = ceil_to_pos(np.max(all_max_scale_errors), -3)
            plt.ylim([0, scale_plots_y_lim_upper])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "scale_error_max.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Minimaler Skalierungsfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                plt.plot(frames, dataset_neural_net_df["Min Scale Error"], label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Skalierungsfehler")
            plt.legend()
            plt.ylim([0, scale_plots_y_lim_upper])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "scale_error_min.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Mittlerer Skalierungsfehler pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                scale_errors = dataset_neural_net_df["Mean Scale Error"]
                scale_std = dataset_neural_net_df["STD Scale Error"]
                plt.plot(frames, scale_errors, label=f"{neural_net_name}")
                plt.fill_between(frames, scale_errors - scale_std, scale_errors + scale_std, alpha=0.2)
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Skalierungsfehler")
            plt.legend()
            plt.ylim([0, scale_plots_y_lim_upper])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "scale_error_mean.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            del scale_errors, scale_std, all_max_scale_errors, scale_plots_y_lim_upper

            # # Correct and incorrect classifications count (disabled because each net may detect a different number of objects. therefore these absolute data are not comparable)
            # frames = range(len(neural_net_results["per_frame_results"]))
            # plt.figure(dpi=300)
            # plt.title(f"Korrekte und inkorrekte Klassifikationen pro Frame\n\n"
            #           f"Datensatz: {dataset_name}")
            # all_correct_classifications = []
            # for neural_net_name, neural_net_result in dataset_result.items():
            #     dataset_neural_net_df = neural_net_result["dataframe"]
            #     correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
            #     incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
            #     plt.plot(frames, correct_classifications_per_frame, label=f"Korrekte Klassifikationen ({neural_net_name})")
            #     plt.plot(frames, incorrect_classifications_per_frame, label=f"Inkorrekte Klassifikationen ({neural_net_name})")
            #     all_correct_classifications.extend(correct_classifications_per_frame)
            # plt.xlabel("Frame-Nummer")
            # plt.ylabel("Anzahl Klassifikationen")
            # plt.legend()
            # plt.ylim([0, ceil_to_pos(np.max(all_correct_classifications), 1)])
            # plt.grid(axis="y")
            # plt.tight_layout()
            #
            # plot_path = os.path.join(out_dir, "classification_counts.pdf")
            # plt.savefig(plot_path)
            # #plt.show()
            # plt.close()

            # Correct and incorrect classifications ratio
            plt.figure(dpi=300)
            plt.title(f"Anteil korrekter Klassifikationen pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
                incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
                correct_classification_ratio = correct_classifications_per_frame / (
                            correct_classifications_per_frame + incorrect_classifications_per_frame)
                plt.plot(frames, correct_classification_ratio, label=f"{neural_net_name}")
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Anteil Klassifikationen")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "classification_ratio_correct.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            plt.figure(dpi=300)
            plt.title(f"Anteil inkorrekter Klassifikationen pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
                incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
                correct_classification_ratio = correct_classifications_per_frame / (
                        correct_classifications_per_frame + incorrect_classifications_per_frame)
                incorrect_classification_ratio = 1 - correct_classification_ratio
                plt.plot(frames, incorrect_classification_ratio, label=f"{neural_net_name}")
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Anteil Klassifikationen")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "classification_ratio_incorrect.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            del correct_classifications_per_frame, incorrect_classifications_per_frame

            # Occlusion ration of undetected objects
            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Maximaler Verdeckungsgrad unentdeckter Objekte pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                plt.plot(frames, dataset_neural_net_df["Undetected Max Occlusion Ratio"], label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Verdeckungsgrad")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.grid(axis="y")
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "occlusion-undetected-objects_max.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Minimaler Verdeckungsgrad unentdeckter Objekte pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                plt.plot(frames, dataset_neural_net_df["Undetected Min Occlusion Ratio"], label=neural_net_name.removesuffix("_Augmentation"))
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Verdeckungsgrad")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.grid(axis="y")
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "occlusion-undetected-objects_min.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Mittlerer Verdeckungsgrad unentdeckter Objekte pro Frame\n\n"
                      f"Datensatz: {dataset_name}")
            for neural_net_name, neural_net_result in dataset_result.items():
                dataset_neural_net_df = neural_net_result["dataframe"]
                occlusion_ratio = dataset_neural_net_df["Undetected Mean Occlusion Ratio"]
                occlusion_std = dataset_neural_net_df["Undetected STD Occlusion Ratio"]
                plt.plot(frames, occlusion_ratio, label=neural_net_name.removesuffix("_Augmentation"))
                plt.fill_between(frames, occlusion_ratio - occlusion_std, occlusion_ratio + occlusion_std, alpha=0.2)
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Verdeckungsgrad")
            plt.legend()
            plt.ylim([-0.05, 1.2])
            plt.grid(axis="y")
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "occlusion-undetected-objects_mean.pdf")
            plt.savefig(plot_path)
            #plt.show()
            plt.close()

            del occlusion_ratio, occlusion_std

        # Per Material
        print("Creating per material plots")
        materials = list(set([dataset_name.split("_")[-1] for dataset_name in per_dataset_results.keys()])) # get unique material names
        neural_net_names = []
        per_material_results = {}
        for material in materials:
            material_dataset_keys = [dataset_name for dataset_name in per_dataset_results.keys() if dataset_name.endswith(material)]
            # Aquire raw data for all neural nets in all matching datasets
            per_material_results[material] = {}
            for material_dataset_name in material_dataset_keys:
                dataset_results = per_dataset_results[material_dataset_name]
                for neural_net_name, neural_net_result in dataset_results.items():
                    if neural_net_name not in neural_net_names:
                        neural_net_names.append(neural_net_name)
                    if neural_net_name not in per_material_results[material]:
                        per_material_results[material][neural_net_name] = {
                            "found_objects_ratio": [],
                            "distance_errors": [],
                            "rotation_errors": [],
                            "scale_errors": [],
                            "occlusion_ratio_undetected_objects": [],
                            "correct_classifications_ratio": [],
                        }
                    dataset_neural_net_df = neural_net_result["dataframe"]

                    # ratio of found objects
                    per_frame_found_ratio = dataset_neural_net_df["Predicted Object Count"] / dataset_neural_net_df["Object Count"]
                    per_material_results[material][neural_net_name]["found_objects_ratio"].extend(per_frame_found_ratio)

                    # distance error, rotation error, scale error, occlusion of undetected objects
                    distance_errors = []
                    rotation_errors = []
                    scale_errors = []
                    occlusion_ratio_undetected_objects = []

                    for frame_result in neural_net_result["per_frame_results"]:
                        distance_errors.extend(frame_result["distance_errors"])
                        rotation_errors.extend(frame_result["rotation_errors"])
                        scale_errors.extend(frame_result["scale_errors"])
                        occlusion_ratio_undetected_objects.extend(frame_result["occlusion_of_non_detected_objects"])
                    per_material_results[material][neural_net_name]["distance_errors"].extend(distance_errors)
                    per_material_results[material][neural_net_name]["rotation_errors"].extend(rotation_errors)
                    per_material_results[material][neural_net_name]["scale_errors"].extend(scale_errors)
                    per_material_results[material][neural_net_name]["occlusion_ratio_undetected_objects"] = occlusion_ratio_undetected_objects

                    # correct_classifications_ratio
                    correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
                    incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
                    correct_classifications_ratio = correct_classifications_per_frame / (correct_classifications_per_frame + incorrect_classifications_per_frame)
                    per_material_results[material][neural_net_name]["correct_classifications_ratio"] = correct_classifications_ratio

        # Create per material plots
        out_dir = os.path.join(output_dir, dataset_type_name)
        materials = sorted(materials)
        x_labels = materials
        x = np.arange(len(x_labels))  # label locations
        num_bars_per_group = len(neural_net_names)
        bar_width = 0.9 / num_bars_per_group
        best_bar_width = 0.9 / 4

        # Create plot data
        neural_net_found_object_means_grouped_by_neural_net_name = {}
        neural_net_found_object_std_grouped_by_neural_net_name = {}
        neural_net_distance_error_means_grouped_by_neural_net_name = {}
        neural_net_distance_error_std_grouped_by_neural_net_name = {}
        neural_net_rotation_error_means_grouped_by_neural_net_name = {}
        neural_net_rotation_error_std_grouped_by_neural_net_name = {}
        neural_net_scale_error_means_grouped_by_neural_net_name = {}
        neural_net_scale_error_std_grouped_by_neural_net_name = {}
        neural_net_occlusion_ratio_means_grouped_by_neural_net_name = {}
        neural_net_occlusion_ratio_std_grouped_by_neural_net_name = {}
        neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name = {}
        neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name = {}
        for material in materials:
            for neural_net_name in neural_net_names:
                if neural_net_name not in neural_net_found_object_means_grouped_by_neural_net_name:
                    neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[neural_net_name] = []

                # Found objects ratio
                neural_net_detected_object_share = per_material_results[material][neural_net_name]["found_objects_ratio"]
                neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name].append(np.mean(neural_net_detected_object_share))
                neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name].append(np.std(neural_net_detected_object_share))

                # Distance error
                neural_net_distance_errors = per_material_results[material][neural_net_name]["distance_errors"]
                neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_distance_errors))
                neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_distance_errors))

                # Rotation error
                neural_net_rotation_errors = per_material_results[material][neural_net_name]["rotation_errors"]
                neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_rotation_errors))
                neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_rotation_errors))

                # Scale error
                neural_net_scale_errors = per_material_results[material][neural_net_name]["scale_errors"]
                neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_scale_errors))
                neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_scale_errors))

                # occlusion of undetected objects
                neural_net_occlusion_ratios = per_material_results[material][neural_net_name]["occlusion_ratio_undetected_objects"]
                neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_occlusion_ratios))
                neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_occlusion_ratios))

                # correct classification ratio
                neural_net_correct_classifications_ratios = per_material_results[material][neural_net_name]["correct_classifications_ratio"]
                neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_correct_classifications_ratios))
                neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_correct_classifications_ratios))

        # Create plots
        ## Ratio of found objects
        fig, ax = plt.subplots() # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name], bar_width, label=neural_net_name.removesuffix("_Augmentation"), yerr=neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil gefundener Objekte")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil gefundener Objekte nach Objektmaterial")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "detected-objects-per-material.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Ratio of found objects (best)
        best_neural_net_names = [neural_net_name for neural_net_name in neural_net_names if neural_net_name.endswith("Conveyor") or neural_net_name.endswith("Conveyor_Augmentation")]
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width, neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil gefundener Objekte")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil gefundener Objekte nach Objektmaterial")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "detected-objects-per-material-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Distance error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Distanzfehler [m]")
        #ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Distanzfehler nach Objektmaterial")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        #ax.set_ylim([0, 1.2])
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "distance-errors-per-material.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Distance error (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Distanzfehler [m]")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Distanzfehler nach Objektmaterial")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "distance-errors-per-material-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Rotation error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Rotationsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Rotationsfehler nach Objektmaterial")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "rotation-errors-per-material.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Rotation error (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Rotationsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Rotationsfehler nach Objektmaterial")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "rotation-errors-per-material-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Scale error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax, round_decimals=4)
        ax.set_ylabel("Skalierungsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Skalierungsfehler nach Objektmaterial")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "scale-errors-per-material.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Scale error (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width, neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax, round_decimals=4)
        ax.set_ylabel("Skalierungsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Skalierungsfehler nach Objektmaterial")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "scale-errors-per-material-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Occlusion of undetected objects
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Verdeckungsgrad")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Verdeckungsgrad unentdeckter Objekte nach Objektmaterial")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "occlusion-undetected-per-material.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Occlusion of undetected objects (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Verdeckungsgrad")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Verdeckungsgrad unentdeckter Objekte nach Objektmaterial")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "occlusion-undetected-per-material-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Correct classifications ratio
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil korrekter Klassifizierungen")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil korrekter Klassifizierungen nach Objektmaterial")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "correct-classifications-per-material.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Correct classifications ratio
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[
                               neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil korrekter Klassifizierungen")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil korrekter Klassifizierungen nach Objektmaterial")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "correct-classifications-per-material-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        # Per Object Type
        print("Creating per object type plots")
        object_types = list(
            set(["_".join(dataset_name.split("_")[:-2]) for dataset_name in per_dataset_results.keys()]))  # get unique object type names
        neural_net_names = []
        per_object_type_results = {}
        for object_type in object_types:
            object_type_dataset_keys = [dataset_name for dataset_name in per_dataset_results.keys() if
                                     dataset_name.startswith(object_type)]
            # Aquire raw data for all neural nets in all matching datasets
            per_object_type_results[object_type] = {}
            for object_dataset_name in object_type_dataset_keys:
                dataset_results = per_dataset_results[object_dataset_name]
                for neural_net_name, neural_net_result in dataset_results.items():
                    if neural_net_name not in neural_net_names:
                        neural_net_names.append(neural_net_name)
                    if neural_net_name not in per_object_type_results[object_type]:
                        per_object_type_results[object_type][neural_net_name] = {
                            "found_objects_ratio": [],
                            "distance_errors": [],
                            "rotation_errors": [],
                            "scale_errors": [],
                            "occlusion_ratio_undetected_objects": [],
                            "correct_classifications_ratio": [],
                        }
                    dataset_neural_net_df = neural_net_result["dataframe"]

                    # ratio of found objects
                    per_frame_found_ratio = dataset_neural_net_df["Predicted Object Count"] / dataset_neural_net_df[
                        "Object Count"]
                    per_object_type_results[object_type][neural_net_name]["found_objects_ratio"].extend(per_frame_found_ratio)

                    # distance error, rotation error, scale error, occlusion of undetected objects
                    distance_errors = []
                    rotation_errors = []
                    scale_errors = []
                    occlusion_ratio_undetected_objects = []

                    for frame_result in neural_net_result["per_frame_results"]:
                        distance_errors.extend(frame_result["distance_errors"])
                        rotation_errors.extend(frame_result["rotation_errors"])
                        scale_errors.extend(frame_result["scale_errors"])
                        occlusion_ratio_undetected_objects.extend(frame_result["occlusion_of_non_detected_objects"])
                    per_object_type_results[object_type][neural_net_name]["distance_errors"].extend(distance_errors)
                    per_object_type_results[object_type][neural_net_name]["rotation_errors"].extend(rotation_errors)
                    per_object_type_results[object_type][neural_net_name]["scale_errors"].extend(scale_errors)
                    per_object_type_results[object_type][neural_net_name][
                        "occlusion_ratio_undetected_objects"] = occlusion_ratio_undetected_objects

                    # correct_classifications_ratio
                    correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
                    incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
                    correct_classifications_ratio = correct_classifications_per_frame / (
                                correct_classifications_per_frame + incorrect_classifications_per_frame)
                    per_object_type_results[object_type][neural_net_name][
                        "correct_classifications_ratio"] = correct_classifications_ratio

        # Create per object type plots
        out_dir = os.path.join(output_dir, dataset_type_name)
        object_types = sorted(object_types)
        x_labels = object_types
        x = np.arange(len(x_labels))  # label locations
        num_bars_per_group = len(neural_net_names)
        bar_width = 0.9 / num_bars_per_group
        best_bar_width = 0.9 / 4

        # Create plot data
        neural_net_found_object_means_grouped_by_neural_net_name = {}
        neural_net_found_object_std_grouped_by_neural_net_name = {}
        neural_net_distance_error_means_grouped_by_neural_net_name = {}
        neural_net_distance_error_std_grouped_by_neural_net_name = {}
        neural_net_rotation_error_means_grouped_by_neural_net_name = {}
        neural_net_rotation_error_std_grouped_by_neural_net_name = {}
        neural_net_scale_error_means_grouped_by_neural_net_name = {}
        neural_net_scale_error_std_grouped_by_neural_net_name = {}
        neural_net_occlusion_ratio_means_grouped_by_neural_net_name = {}
        neural_net_occlusion_ratio_std_grouped_by_neural_net_name = {}
        neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name = {}
        neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name = {}
        for object_type in object_types:
            for neural_net_name in neural_net_names:
                if neural_net_name not in neural_net_found_object_means_grouped_by_neural_net_name:
                    neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name] = []
                    neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[neural_net_name] = []

                # Found objects ratio
                neural_net_detected_object_share = per_object_type_results[object_type][neural_net_name]["found_objects_ratio"]
                neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_detected_object_share))
                neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_detected_object_share))

                # Distance error
                neural_net_distance_errors = per_object_type_results[object_type][neural_net_name]["distance_errors"]
                neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_distance_errors))
                neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_distance_errors))

                # Rotation error
                neural_net_rotation_errors = per_object_type_results[object_type][neural_net_name]["rotation_errors"]
                neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_rotation_errors))
                neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_rotation_errors))

                # Scale error
                neural_net_scale_errors = per_object_type_results[object_type][neural_net_name]["scale_errors"]
                neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_scale_errors))
                neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_scale_errors))

                # occlusion of undetected objects
                neural_net_occlusion_ratios = per_object_type_results[object_type][neural_net_name][
                    "occlusion_ratio_undetected_objects"]
                neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_occlusion_ratios))
                neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_occlusion_ratios))

                # correct classification ratio
                neural_net_correct_classifications_ratios = per_object_type_results[object_type][neural_net_name][
                    "correct_classifications_ratio"]
                neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name].append(
                    np.mean(neural_net_correct_classifications_ratios))
                neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[neural_net_name].append(
                    np.std(neural_net_correct_classifications_ratios))

        # Create plots
        print("Creating per dataset type plots")
        ## Ratio of found objects
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil gefundener Objekte")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil gefundener Objekte nach Objekttyp")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "detected-objects-per-object-type.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Ratio of found objects (best)
        best_neural_net_names = [neural_net_name for neural_net_name in neural_net_names if neural_net_name.endswith("Conveyor") or neural_net_name.endswith("Conveyor_Augmentation")]
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width, neural_net_found_object_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_found_object_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil gefundener Objekte")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil gefundener Objekte nach Objekttyp")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "detected-objects-per-object-type-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Distance error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Distanzfehler [m]")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Distanzfehler nach Objekttyp")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "distance-errors-per-object-type.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Distance error (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_distance_error_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_distance_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Distanzfehler [m]")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Distanzfehler nach Objekttyp")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "distance-errors-per-object-type-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Rotation error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Rotationsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Rotationsfehler nach Objekttyp")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "rotation-errors-per-object-type.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Rotation error (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_rotation_error_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_rotation_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax)
        ax.set_ylabel("Rotationsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Rotationsfehler nach Objekttyp")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "rotation-errors-per-object-type-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Scale error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax, round_decimals=4)
        ax.set_ylabel("Skalierungsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Skalierungsfehler nach Objekttyp")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "scale-errors-per-object-type.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Scale error (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width, neural_net_scale_error_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_scale_error_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel(rects, ax, round_decimals=4)
        ax.set_ylabel("Skalierungsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Skalierungsfehler nach Objekttyp")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "scale-errors-per-object-type-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Occlusion of undetected objects
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width, neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Verdeckungsgrad")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Verdeckungsgrad unentdeckter Objekte nach Objekttyp")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "occlusion-undetected-per-object-type.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Occlusion of undetected objects (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_occlusion_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_occlusion_ratio_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Verdeckungsgrad")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Verdeckungsgrad unentdeckter Objekte nach Objekttyp")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "occlusion-undetected-per-object-type-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        ## Correct classifications ratio
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * bar_width,
                           neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil korrekter Klassifizierungen")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil korrekter Klassifizierungen nach Objekttyp")
        x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "correct-classifications-per-object-type.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()

        ## Correct classifications ratio (best)
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        for i, neural_net_name in enumerate(best_neural_net_names):  # add data for each neural net to plot
            rects = ax.bar(x + i * best_bar_width,
                           neural_net_correct_classifications_ratio_means_grouped_by_neural_net_name[neural_net_name],
                           best_bar_width, label=neural_net_name.removesuffix("_Augmentation"),
                           yerr=neural_net_correct_classifications_ratio_std_grouped_by_neural_net_name[
                               neural_net_name],
                           error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
            # ax.bar_label(rects, padding=3)
            autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil korrekter Klassifizierungen")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil korrekter Klassifizierungen nach Objekttyp")
        x_ticks = x + best_bar_width * (4 - 1) / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        ax.legend()
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "correct-classifications-per-object-type-best.pdf")
        plt.savefig(plot_path)
        # plt.show()
        plt.close()

        # Whole dataset type (cluttered / uncluttered)
        neural_net_names = []
        # Aquire raw data for all neural nets in all datasets
        dataset_type_results = {}
        for dataset_name in per_dataset_results.keys():
            dataset_results = per_dataset_results[dataset_name]
            for neural_net_name, neural_net_result in dataset_results.items():
                if neural_net_name not in neural_net_names:
                    neural_net_names.append(neural_net_name)
                if neural_net_name not in dataset_type_results:
                    dataset_type_results[neural_net_name] = {
                        "found_objects_ratio": [],
                        "distance_errors": [],
                        "rotation_errors": [],
                        "scale_errors": [],
                        "occlusion_ratio_undetected_objects": [],
                        "correct_classifications_ratio": [],
                    }
                if neural_net_name not in all_dataset_results:
                    all_dataset_results[neural_net_name] = {
                        "found_objects_ratio": [],
                        "distance_errors": [],
                        "rotation_errors": [],
                        "scale_errors": [],
                        "occlusion_ratio_undetected_objects": [],
                        "correct_classifications_ratio": [],
                    }
                dataset_neural_net_df = neural_net_result["dataframe"]

                # ratio of found objects
                per_frame_found_ratio = dataset_neural_net_df["Predicted Object Count"] / dataset_neural_net_df[
                    "Object Count"]
                dataset_type_results[neural_net_name]["found_objects_ratio"].extend(
                    per_frame_found_ratio)
                all_dataset_results[neural_net_name]["found_objects_ratio"].extend(
                    per_frame_found_ratio)

                # distance error, rotation error, scale error, occlusion of undetected objects
                distance_errors = []
                rotation_errors = []
                scale_errors = []
                occlusion_ratio_undetected_objects = []

                for frame_result in neural_net_result["per_frame_results"]:
                    distance_errors.extend(frame_result["distance_errors"])
                    rotation_errors.extend(frame_result["rotation_errors"])
                    scale_errors.extend(frame_result["scale_errors"])
                    occlusion_ratio_undetected_objects.extend(frame_result["occlusion_of_non_detected_objects"])
                dataset_type_results[neural_net_name]["distance_errors"].extend(distance_errors)
                dataset_type_results[neural_net_name]["rotation_errors"].extend(rotation_errors)
                dataset_type_results[neural_net_name]["scale_errors"].extend(scale_errors)
                dataset_type_results[neural_net_name][
                    "occlusion_ratio_undetected_objects"] = occlusion_ratio_undetected_objects
                all_dataset_results[neural_net_name]["distance_errors"].extend(distance_errors)
                all_dataset_results[neural_net_name]["rotation_errors"].extend(rotation_errors)
                all_dataset_results[neural_net_name]["scale_errors"].extend(scale_errors)
                all_dataset_results[neural_net_name][
                    "occlusion_ratio_undetected_objects"] = occlusion_ratio_undetected_objects

                # correct_classifications_ratio
                correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
                incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
                correct_classifications_ratio = correct_classifications_per_frame / (
                        correct_classifications_per_frame + incorrect_classifications_per_frame)
                dataset_type_results[neural_net_name][
                    "correct_classifications_ratio"] = correct_classifications_ratio
                all_dataset_results[neural_net_name][
                    "correct_classifications_ratio"] = correct_classifications_ratio

        # Create per dataset type plots
        out_dir = os.path.join(output_dir, dataset_type_name)
        neural_net_names = sorted(neural_net_names)
        #x = np.arange(len(x_labels))  # label locations
        #num_bars_per_group = len(neural_net_names)
        #bar_width = 0.9 / num_bars_per_group

        # Create plot data
        neural_net_found_object_means = {}
        neural_net_found_object_std = {}
        neural_net_distance_error_means = {}
        neural_net_distance_error_std = {}
        neural_net_rotation_error_means = {}
        neural_net_rotation_error_std = {}
        neural_net_scale_error_means = {}
        neural_net_scale_error_std = {}
        neural_net_occlusion_ratio_means = {}
        neural_net_occlusion_ratio_std = {}
        neural_net_correct_classifications_ratio_means = {}
        neural_net_correct_classifications_ratio_std = {}
        for neural_net_name in neural_net_names:
            if neural_net_name not in neural_net_found_object_means:
                neural_net_found_object_means[neural_net_name] = []
                neural_net_found_object_std[neural_net_name] = []
                neural_net_distance_error_means[neural_net_name] = []
                neural_net_distance_error_std[neural_net_name] = []
                neural_net_rotation_error_means[neural_net_name] = []
                neural_net_rotation_error_std[neural_net_name] = []
                neural_net_scale_error_means[neural_net_name] = []
                neural_net_scale_error_std[neural_net_name] = []
                neural_net_occlusion_ratio_means[neural_net_name] = []
                neural_net_occlusion_ratio_std[neural_net_name] = []
                neural_net_correct_classifications_ratio_means[neural_net_name] = []
                neural_net_correct_classifications_ratio_std[neural_net_name] = []

            # Found objects ratio
            neural_net_detected_object_share = dataset_type_results[neural_net_name][
                "found_objects_ratio"]
            neural_net_found_object_means[neural_net_name].append(
                np.mean(neural_net_detected_object_share))
            neural_net_found_object_std[neural_net_name].append(
                np.std(neural_net_detected_object_share))

            # Distance error
            neural_net_distance_errors = dataset_type_results[neural_net_name]["distance_errors"]
            neural_net_distance_error_means[neural_net_name].append(
                np.mean(neural_net_distance_errors))
            neural_net_distance_error_std[neural_net_name].append(
                np.std(neural_net_distance_errors))

            # Rotation error
            neural_net_rotation_errors = dataset_type_results[neural_net_name]["rotation_errors"]
            neural_net_rotation_error_means[neural_net_name].append(
                np.mean(neural_net_rotation_errors))
            neural_net_rotation_error_std[neural_net_name].append(
                np.std(neural_net_rotation_errors))

            # Scale error
            neural_net_scale_errors = dataset_type_results[neural_net_name]["scale_errors"]
            neural_net_scale_error_means[neural_net_name].append(
                np.mean(neural_net_scale_errors))
            neural_net_scale_error_std[neural_net_name].append(
                np.std(neural_net_scale_errors))

            # occlusion of undetected objects
            neural_net_occlusion_ratios = dataset_type_results[neural_net_name][
                "occlusion_ratio_undetected_objects"]
            neural_net_occlusion_ratio_means[neural_net_name].append(
                np.mean(neural_net_occlusion_ratios))
            neural_net_occlusion_ratio_std[neural_net_name].append(
                np.std(neural_net_occlusion_ratios))

            # correct classification ratio
            neural_net_correct_classifications_ratios = dataset_type_results[neural_net_name][
                "correct_classifications_ratio"]
            neural_net_correct_classifications_ratio_means[neural_net_name].append(
                np.mean(neural_net_correct_classifications_ratios))
            neural_net_correct_classifications_ratio_std[neural_net_name].append(
                np.std(neural_net_correct_classifications_ratios))

        # Create plots
        ## Ratio of found objects
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        results = [neural_net_found_object_means[neural_net_name] for neural_net_name in neural_net_names]
        results = [val for sublist in results for val in sublist]  # flatten results
        stds = [neural_net_found_object_std[neural_net_name] for neural_net_name in neural_net_names]
        stds = [val for sublist in stds for val in sublist]  # flatten stds
        rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds, error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
        autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil gefundener Objekte")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil gefundener Objekte")
        #x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        #ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "detected-objects.pdf")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        ## Distance error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        results = [neural_net_distance_error_means[neural_net_name] for neural_net_name in neural_net_names]
        results = [val for sublist in results for val in sublist]  # flatten results
        stds = [neural_net_distance_error_std[neural_net_name] for neural_net_name in neural_net_names]
        stds = [val for sublist in stds for val in sublist]  # flatten stds
        rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds, error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
        autolabel(rects, ax)
        ax.set_ylabel("Distanzfehler [m]")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Distanzfehler")
        #x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        #ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "distance-errors.pdf")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        ## Rotation error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        results = [neural_net_rotation_error_means[neural_net_name] for neural_net_name in neural_net_names]
        results = [val for sublist in results for val in sublist]  # flatten results
        stds = [neural_net_rotation_error_std[neural_net_name] for neural_net_name in neural_net_names]
        stds = [val for sublist in stds for val in sublist]  # flatten stds
        rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds, error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
        autolabel(rects, ax)
        ax.set_ylabel("Rotationsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Rotationsfehler")
        #x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        #ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "rotation-errors.pdf")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        ## Scale error
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        results = [neural_net_scale_error_means[neural_net_name] for neural_net_name in neural_net_names]
        results = [val for sublist in results for val in sublist]  # flatten results
        stds = [neural_net_scale_error_std[neural_net_name] for neural_net_name in neural_net_names]
        stds = [val for sublist in stds for val in sublist]  # flatten stds
        rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds, error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
        autolabel(rects, ax, round_decimals=4)
        ax.set_ylabel("Skalierungsfehler")
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Skalierungsfehler")
        #x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_ylim([0, 1.2])
        #ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "scale-errors.pdf")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        ## Occlusion of undetected objects
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        results = [neural_net_occlusion_ratio_means[neural_net_name] for neural_net_name in neural_net_names]
        results = [val for sublist in results for val in sublist]  # flatten results
        stds = [neural_net_occlusion_ratio_std[neural_net_name] for neural_net_name in neural_net_names]
        stds = [val for sublist in stds for val in sublist]  # flatten stds
        rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds, error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
        autolabel_percent(rects, ax)
        ax.set_ylabel("Verdeckungsgrad")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Verdeckungsgrad")
        #x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        #ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "occlusion-undetected.pdf")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        ## Correct classifications ratio
        fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
        results = [neural_net_correct_classifications_ratio_means[neural_net_name] for neural_net_name in neural_net_names]
        results = [val for sublist in results for val in sublist]  # flatten results
        stds = [neural_net_correct_classifications_ratio_std[neural_net_name] for neural_net_name in neural_net_names]
        stds = [val for sublist in stds for val in sublist]  # flatten stds
        rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds, error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
        autolabel_percent(rects, ax)
        ax.set_ylabel("Anteil korrekter Klassifizierungen")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_title("Mittlerer Anteil korrekter Klassifizierungen")
        #x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
        #ax.set_xticks(x_ticks)
        #ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', labelrotation=90)
        plt.ylim(bottom=0)
        current_max = plt.ylim()[1]
        if current_max < 1.4:
            plt.ylim(top=1.4)
        #ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(out_dir, "correct-classifications.pdf")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

    # Create all dataset plots
    print("Creating global plots")
    out_dir = os.path.join(output_dir)
    neural_net_names = sorted(neural_net_names)
    x_labels = neural_net_names
    # x = np.arange(len(x_labels))  # label locations
    # num_bars_per_group = len(neural_net_names)
    # bar_width = 0.9 / num_bars_per_group

    # Create plot data
    neural_net_found_object_means = {}
    neural_net_found_object_std = {}
    neural_net_distance_error_means = {}
    neural_net_distance_error_std = {}
    neural_net_rotation_error_means = {}
    neural_net_rotation_error_std = {}
    neural_net_scale_error_means = {}
    neural_net_scale_error_std = {}
    neural_net_occlusion_ratio_means = {}
    neural_net_occlusion_ratio_std = {}
    neural_net_correct_classifications_ratio_means = {}
    neural_net_correct_classifications_ratio_std = {}
    for neural_net_name in neural_net_names:
        if neural_net_name not in neural_net_found_object_means:
            neural_net_found_object_means[neural_net_name] = []
            neural_net_found_object_std[neural_net_name] = []
            neural_net_distance_error_means[neural_net_name] = []
            neural_net_distance_error_std[neural_net_name] = []
            neural_net_rotation_error_means[neural_net_name] = []
            neural_net_rotation_error_std[neural_net_name] = []
            neural_net_scale_error_means[neural_net_name] = []
            neural_net_scale_error_std[neural_net_name] = []
            neural_net_occlusion_ratio_means[neural_net_name] = []
            neural_net_occlusion_ratio_std[neural_net_name] = []
            neural_net_correct_classifications_ratio_means[neural_net_name] = []
            neural_net_correct_classifications_ratio_std[neural_net_name] = []

        # Found objects ratio
        neural_net_detected_object_share = all_dataset_results[neural_net_name][
            "found_objects_ratio"]
        neural_net_found_object_means[neural_net_name].append(
            np.mean(neural_net_detected_object_share))
        neural_net_found_object_std[neural_net_name].append(
            np.std(neural_net_detected_object_share))

        # Distance error
        neural_net_distance_errors = all_dataset_results[neural_net_name]["distance_errors"]
        neural_net_distance_error_means[neural_net_name].append(
            np.mean(neural_net_distance_errors))
        neural_net_distance_error_std[neural_net_name].append(
            np.std(neural_net_distance_errors))

        # Rotation error
        neural_net_rotation_errors = all_dataset_results[neural_net_name]["rotation_errors"]
        neural_net_rotation_error_means[neural_net_name].append(
            np.mean(neural_net_rotation_errors))
        neural_net_rotation_error_std[neural_net_name].append(
            np.std(neural_net_rotation_errors))

        # Scale error
        neural_net_scale_errors = all_dataset_results[neural_net_name]["scale_errors"]
        neural_net_scale_error_means[neural_net_name].append(
            np.mean(neural_net_scale_errors))
        neural_net_scale_error_std[neural_net_name].append(
            np.std(neural_net_scale_errors))

        # occlusion of undetected objects
        neural_net_occlusion_ratios = all_dataset_results[neural_net_name][
            "occlusion_ratio_undetected_objects"]
        neural_net_occlusion_ratio_means[neural_net_name].append(
            np.mean(neural_net_occlusion_ratios))
        neural_net_occlusion_ratio_std[neural_net_name].append(
            np.std(neural_net_occlusion_ratios))

        # correct classification ratio
        neural_net_correct_classifications_ratios = all_dataset_results[neural_net_name][
            "correct_classifications_ratio"]
        neural_net_correct_classifications_ratio_means[neural_net_name].append(
            np.mean(neural_net_correct_classifications_ratios))
        neural_net_correct_classifications_ratio_std[neural_net_name].append(
            np.std(neural_net_correct_classifications_ratios))

    # Create plots
    ## Ratio of found objects
    fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
    results = [neural_net_found_object_means[neural_net_name] for neural_net_name in neural_net_names]
    results = [val for sublist in results for val in sublist]  # flatten results
    stds = [neural_net_found_object_std[neural_net_name] for neural_net_name in neural_net_names]
    stds = [val for sublist in stds for val in sublist]  # flatten stds
    rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds,
                   error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
    autolabel_percent(rects, ax)
    ax.set_ylabel("Anteil gefundener Objekte")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Mittlerer Anteil gefundener Objekte")
    # x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylim(bottom=0)
    current_max = plt.ylim()[1]
    if current_max < 1.4:
        plt.ylim(top=1.4)
    # ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "detected-objects.pdf")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    ## Distance error
    fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
    results = [neural_net_distance_error_means[neural_net_name] for neural_net_name in neural_net_names]
    results = [val for sublist in results for val in sublist]  # flatten results
    stds = [neural_net_distance_error_std[neural_net_name] for neural_net_name in neural_net_names]
    stds = [val for sublist in stds for val in sublist]  # flatten stds
    rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds,
                   error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
    autolabel(rects, ax)
    ax.set_ylabel("Distanzfehler [m]")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Mittlerer Distanzfehler")
    # x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', labelrotation=90)
    # ax.set_ylim([0, 1.2])
    # ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "distance-errors.pdf")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    ## Rotation error
    fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
    results = [neural_net_rotation_error_means[neural_net_name] for neural_net_name in neural_net_names]
    results = [val for sublist in results for val in sublist]  # flatten results
    stds = [neural_net_rotation_error_std[neural_net_name] for neural_net_name in neural_net_names]
    stds = [val for sublist in stds for val in sublist]  # flatten stds
    rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds,
                   error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
    autolabel(rects, ax)
    ax.set_ylabel("Rotationsfehler")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Mittlerer Rotationsfehler")
    # x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylim(bottom=0)
    current_max = plt.ylim()[1]
    if current_max < 1.4:
        plt.ylim(top=1.4)
    # ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "rotation-errors.pdf")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    ## Scale error
    fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
    results = [neural_net_scale_error_means[neural_net_name] for neural_net_name in neural_net_names]
    results = [val for sublist in results for val in sublist]  # flatten results
    stds = [neural_net_scale_error_std[neural_net_name] for neural_net_name in neural_net_names]
    stds = [val for sublist in stds for val in sublist]  # flatten stds
    rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds,
                   error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
    autolabel(rects, ax, round_decimals=4)
    ax.set_ylabel("Skalierungsfehler")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Mittlerer Skalierungsfehler")
    # x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', labelrotation=90)
    # ax.set_ylim([0, 1.2])
    # ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "scale-errors.pdf")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    ## Occlusion of undetected objects
    fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
    results = [neural_net_occlusion_ratio_means[neural_net_name] for neural_net_name in neural_net_names]
    results = [val for sublist in results for val in sublist]  # flatten results
    stds = [neural_net_occlusion_ratio_std[neural_net_name] for neural_net_name in neural_net_names]
    stds = [val for sublist in stds for val in sublist]  # flatten stds
    rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds,
                   error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
    autolabel_percent(rects, ax)
    ax.set_ylabel("Verdeckungsgrad")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Mittlerer Verdeckungsgrad")
    # x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylim(bottom=0)
    current_max = plt.ylim()[1]
    if current_max < 1.4:
        plt.ylim(top=1.4)
    # ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "occlusion-undetected.pdf")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    ## Correct classifications ratio
    fig, ax = plt.subplots()  # figsize=(12.0, 4.8), dpi=300)
    results = [neural_net_correct_classifications_ratio_means[neural_net_name] for neural_net_name in
               neural_net_names]
    results = [val for sublist in results for val in sublist]  # flatten results
    stds = [neural_net_correct_classifications_ratio_std[neural_net_name] for neural_net_name in neural_net_names]
    stds = [val for sublist in stds for val in sublist]  # flatten stds
    rects = ax.bar([neural_net_name.removesuffix("_Augmentation") for neural_net_name in neural_net_names], results, yerr=stds,
                   error_kw=dict(ecolor='lightgray', lw=2, capsize=5, capthick=2))
    autolabel_percent(rects, ax)
    ax.set_ylabel("Anteil korrekter Klassifizierungen")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Mittlerer Anteil korrekter Klassifizierungen")
    # x_ticks = x + bar_width * (num_bars_per_group - 1) / 2
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylim(bottom=0)
    current_max = plt.ylim()[1]
    if current_max < 1.4:
        plt.ylim(top=1.4)
    # ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "correct-classifications.pdf")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    ## Radar plot
    share_plot_axes_names = ["Anteil detektiert", "Anteil korrekt klassifiziert", "Verdeckungsgrad undetektiert"]
    radar_plot = radar_factory(len(share_plot_axes_names), frame="circle")
    data = {}
    for neural_net_name in neural_net_names:
        data[neural_net_name] = []
        data[neural_net_name].append(neural_net_found_object_means[neural_net_name][0])
        data[neural_net_name].append(neural_net_correct_classifications_ratio_means[neural_net_name][0])
        data[neural_net_name].append(neural_net_occlusion_ratio_means[neural_net_name][0])
    fig, ax = plt.subplots(figsize=(5, 5), #nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))
    ax.set_title("Objekterkennung und Klassifizierung")
    for neural_net_name, d in data.items():
        ax.plot(radar_plot, d, label=neural_net_name.removesuffix("_Augmentation"))
        ax.fill(radar_plot, d, alpha=0.25)
    ax.set_varlabels(share_plot_axes_names)

    plt.show()

    # absolute_plot_axes = ["Distanzfehler", "Rotationsfehler", "Skalierungsfehler"]



    # num_found_objects_to_inference_time
    x_num_objects = []
    y_inference_time = []
    for num_found_object, inference_times in num_found_objects_to_inference_time.items():
        x_num_objects.extend(np.ones(len(inference_times)) * num_found_object)
        y_inference_time.extend(inference_times)

    ## remove outlier (TODO: check if setup outlier ist only in first dataset processed by eval_raw_data generation)
    # x_num_objects = x_num_objects[1:]
    # y_inference_time = y_inference_time[1:]

    # calc correlation coefficient and linear regression
    corr_coef = stats.pearsonr(x_num_objects, y_inference_time).correlation
    linregress = stats.linregress(x_num_objects, y_inference_time)
    linregress_x = np.linspace(np.min(x_num_objects), np.max(x_num_objects))
    linregress_y = linregress.slope * linregress_x + linregress.intercept

    ## build plot
    plt.figure(dpi=300)
    plt.title("Anzahl gefundener Objekte in Relation zur Inferenz-Zeit\n\n"
              f"Korrelationskoeffizient: {np.round(corr_coef, 5)}")
    plt.scatter(x_num_objects, y_inference_time)
    plt.plot(linregress_x, linregress_y, "r",
             label=fr"Lineare Regression: $f(x) = {np.round(linregress.slope, 5)} \cdot x + {np.round(linregress.intercept, 5)}$")
    plt.ylabel("Inferenz-Zeit [s]")
    plt.xlabel("Anzahl gefundener Objekte")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "found_objects_to_inference_time.pdf")
    plt.savefig(plot_path)
    # plt.show()
    plt.close()

    del corr_coef, linregress, linregress_x, linregress_y


# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py (23.10.22)
def autolabel(rects, ax, round_decimals=2):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, round_decimals)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    rotation=90, color="red")


def autolabel_percent(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(np.round(height * 100, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    rotation=90, color="red")


# https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html (10.05.2024)
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == '__main__':
    #eval_dataset_path = "/Users/flo/eval_dataset"
    #eval_dataset_path = "/home/flo/eval_dataset"
    eval_dataset_path = "/media/flo/8ACA8610CA85F8A9/eval_dataset"
    output_dir = "output"
    #eval_dataset_path = "C:\\Users\\floriand\\eval_dataset"
    #output_dir = r".\\output"
    main(eval_dataset_path, output_dir)
