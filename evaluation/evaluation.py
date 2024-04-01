import numpy as np
import quaternion
import matplotlib.pyplot as plt

import os
import glob
import json


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

    return 1 - abs(np.dot(gt_quat, found_quat))


def main(eval_dataset_path: str):
    num_found_objects_to_inference_time = {}

    # Cluttered
    cluttered_dataset_paths = glob.glob(os.path.join(eval_dataset_path, "converted", "cluttered", "*"))
    per_dataset_results = {}
    for cluttered_dataset_path in cluttered_dataset_paths:
        dataset_name = os.path.basename(cluttered_dataset_path)
        per_dataset_results[dataset_name] = {}
        neural_net_paths = glob.glob(os.path.join(cluttered_dataset_path, "ReplicatorToRocaEval", "eval_raw_data", "*"))
        for neural_net_path in neural_net_paths:
            neural_net_name = os.path.basename(neural_net_path)
            per_dataset_results[dataset_name][neural_net_name] = []

            # Load results for dataset and neural net
            neural_net_json = os.path.join(neural_net_path, "eval_raw_data.json")
            with open(neural_net_json, "r") as f:
                neural_net_results = json.load(f)

            # Calc per frame stats
            for frame_result in neural_net_results["per_frame_results"]:
                # Load ground truth data
                image_name: str = frame_result["image"]
                image_number = image_name.removesuffix(".jpg").removeprefix("image_")
                ground_truth_visible_objects_data_path = os.path.join(cluttered_dataset_path, "ReplicatorToRocaEval", f"world_pose_visible_objects_{image_number}.json")
                with open(ground_truth_visible_objects_data_path, "r") as f:
                    ground_truth_visible_objects = json.load(f)

                # Get number of objects
                ground_truth_visible_objects_count = len(ground_truth_visible_objects)
                found_object_count = len(frame_result["instances"])
                if found_object_count > ground_truth_visible_objects_count:
                    raise ValueError(f"Num found objects greater than ground truth ({found_object_count} > {ground_truth_visible_objects_count})")

                # Calc and store inference time
                inference_time = frame_result["inference_end_time"] - frame_result["inference_start_time"]
                if found_object_count in num_found_objects_to_inference_time.keys():
                    num_found_objects_to_inference_time[found_object_count].append(inference_time)
                else:
                    num_found_objects_to_inference_time[found_object_count] = [inference_time]

                # TODO transform found objects to world frame

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

                # Create per object statistics
                # Distance
                position_difference_per_axis = [calc_per_axis_position_difference(gt_obj, found_obj) for found_obj, gt_obj in found_to_gt_objects]
                distance_errors = [calc_distance(gt_obj, found_obj) for found_obj, gt_obj in found_to_gt_objects]
                #distance_error_mean = np.mean(distance_errors)
                #distance_error_std = np.std(distance_errors)

                # Rotation error
                rotation_errors = [calc_rotation_distance(gt_obj, found_obj) for found_obj, gt_obj in found_to_gt_objects]
                #distance_error_mean = np.mean(distance_errors)
                #distance_error_std = np.std(distance_errors)

                # Scale error
                scale_difference_per_axis = [calc_per_axis_scale_difference(found_obj) for found_obj, _ in found_to_gt_objects]
                scale_errors = [calc_scale_distance(found_obj) for found_obj, _ in found_to_gt_objects]
                #mean
                #std

                # Occlusion ration of undetected objects
                occlusion_ratio_non_detected_objects = [gt_obj["occlusion_ratio"] for gt_obj in ground_truth_visible_objects]
                # mean
                # std
                # min
                # max

                # Objektklassifizierung / Objekttyp
                label_to_predicted_label = {}
                for found_obj, gt_obj in found_to_gt_objects:
                    correct_label = gt_obj["semantic_labels"]["class"]
                    predicated_label = found_obj["semantic_label"]
                    if correct_label in label_to_predicted_label.keys():
                        label_to_predicted_label[correct_label].append(predicated_label)
                    else:
                        label_to_predicted_label[correct_label] = [predicated_label]

                # Store per frame results
                per_dataset_results[dataset_name][neural_net_name].append({
                    "position_difference_per_axis": position_difference_per_axis,
                    "distance_errors": distance_errors,
                    "rotation_errors": rotation_errors,
                    "scale_difference_per_axis": scale_difference_per_axis,
                    "scale_errors": scale_errors,
                    "occlusion_of_non_detected_objects": occlusion_ratio_non_detected_objects,
                    "label_to_predicted_label": label_to_predicted_label
                })

    # Create plots
    # num_found_objects_to_inference_time
    plt.plot()

    # Vergleich Netze zu verschiedenen Materialien


if __name__ == '__main__':
    eval_dataset_path = "/Users/flo/eval_dataset"
    main(eval_dataset_path)
