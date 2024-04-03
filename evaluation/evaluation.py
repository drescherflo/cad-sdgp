import numpy as np
import quaternion
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import scipy.stats as stats

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
    ceil_pot = 10.0 * pos
    return math.ceil(x / ceil_pot) * ceil_pot



def main(eval_dataset_path: str, output_dir: str):
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Create dict for num_objects to inference time
    num_found_objects_to_inference_time = {}

    # Cluttered
    dataset_type_name = "cluttered"
    os.makedirs(os.path.join(output_dir, dataset_type_name), exist_ok=True)
    cluttered_dataset_paths = glob.glob(os.path.join(eval_dataset_path, "converted", "cluttered", "*"))
    per_dataset_results = {}
    for cluttered_dataset_path in cluttered_dataset_paths:
        dataset_name = os.path.basename(cluttered_dataset_path)
        per_dataset_results[dataset_name] = {}
        neural_net_paths = glob.glob(os.path.join(cluttered_dataset_path, "ReplicatorToRocaEval", "eval_raw_data", "*"))
        for neural_net_path in neural_net_paths:
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
                distance_error_mean = np.mean(distance_errors)
                distance_error_std = np.std(distance_errors)
                distance_error_min = np.min(distance_errors)
                distance_error_max = np.max(distance_errors)

                # Rotation error
                rotation_errors = [calc_rotation_distance(gt_obj, found_obj) for found_obj, gt_obj in found_to_gt_objects]
                rotation_error_mean = np.mean(rotation_errors)
                rotation_error_std = np.std(rotation_errors)
                rotation_error_min = np.min(rotation_errors)
                rotation_error_max = np.max(rotation_errors)

                # Scale error
                scale_difference_per_axis = [calc_per_axis_scale_difference(found_obj) for found_obj, _ in found_to_gt_objects]
                scale_errors = [calc_scale_distance(found_obj) for found_obj, _ in found_to_gt_objects]
                scale_error_mean = np.mean(scale_errors)
                scale_error_std = np.std(scale_errors)
                scale_error_min = np.min(scale_errors)
                scale_error_max = np.max(scale_errors)

                # Occlusion ratio of undetected objects
                occlusion_ratio_non_detected_objects = [gt_obj["occlusion_ratio"] for gt_obj in ground_truth_visible_objects]
                undetected_occlusion_mean = np.mean(occlusion_ratio_non_detected_objects)
                undetected_occlusion_std = np.std(occlusion_ratio_non_detected_objects)
                undetected_occlusion_min = np.min(occlusion_ratio_non_detected_objects)
                undetected_occlusion_max = np.max(occlusion_ratio_non_detected_objects)

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
            plt.show()


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
            plt.show()


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
            plt.show()

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
            plt.show()

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
            plt.show()

            del scale_errors, scale_std


            # Occlusion of undetected objects
            occlusion_ratio = dataset_neural_net_df["Undetected Mean Occlusion Ratio"]
            occlusion_std = dataset_neural_net_df["Undetected STD Occlusion Ratio"]
            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Verdeckungsfaktor unentdeckter Objekte pro Frame\n\n"
                      f"Datensatz: {dataset_name}\n"
                      f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
            plt.plot(frames, occlusion_ratio, label="Mittlerer Verdeckungsfaktor")
            plt.fill_between(frames, occlusion_ratio - occlusion_std, occlusion_ratio + occlusion_std, alpha=0.2)
            plt.plot(frames, dataset_neural_net_df["Undetected Max Occlusion Ratio"], label="Maximaler Verdeckungsfaktor")
            plt.plot(frames, dataset_neural_net_df["Undetected Min Occlusion Ratio"], label="Minimaler Verdeckungsfaktor")
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Verdeckungsfaktor")
            plt.legend()
            # plt.ylim([0, ceil_to_pos(np.max(distance_errors), -1)])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "occlusion-undetected-objects.pdf")
            plt.savefig(plot_path)
            plt.show()

            del occlusion_ratio, occlusion_std


            # Correct and incorrect classifications count
            correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
            incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
            frames = range(len(neural_net_results["per_frame_results"]))
            plt.figure(dpi=300)
            plt.title(f"Korrekte und inkorrekte Klassifikationen pro Frame\n\n"
                      f"Datensatz: {dataset_name}\n"
                      f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
            plt.plot(frames, correct_classifications_per_frame, label="Korrekte Klassifizierungen")
            plt.plot(frames, incorrect_classifications_per_frame, label="Inkorrekte Klassifizierungen")
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Anzahl Klassifizierungen")
            plt.legend()
            plt.ylim([0, ceil_to_pos(np.max(correct_classifications_per_frame), 1)])
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "classification_counts.pdf")
            plt.savefig(plot_path)
            plt.show()


            # Correct and incorrect classifications ratio
            plt.figure(dpi=300)
            plt.title(f"Anteil Korrekte und inkorrekte Klassifikationen pro Frame\n\n"
                      f"Datensatz: {dataset_name}\n"
                      f"Neuronales Netz: {neural_net_name.removesuffix('_Augmentation')}")
            correct_classification_ratio = correct_classifications_per_frame / (correct_classifications_per_frame + incorrect_classifications_per_frame)
            incorrect_classification_ratio = 1 - correct_classification_ratio
            plt.fill_between(frames, 0, correct_classification_ratio, label="Korrekte Klassifizierungen")
            #plt.fill_between(frames, correct_classification_ratio, 1, label="Inkorrekte Klassifizierungen")
            plt.fill_between(frames, 0, incorrect_classification_ratio, label="Inkorrekte Klassifizierungen")
            plt.xlabel("Frame-Nummer")
            plt.ylabel("Anteil Klassifizierungen")
            plt.legend()
            plt.ylim([0, 1.05])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.grid(axis="y")
            plt.tight_layout()

            plot_path = os.path.join(out_dir, "classification_ratio.pdf")
            plt.savefig(plot_path)
            plt.show()

            del correct_classifications_per_frame, incorrect_classifications_per_frame


    # Create plots
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
    plt.title("Anzahl gefundener Objekte zur Inferenz-Zeit\n\n"
              f"Korrelationskoeffizient: {np.round(corr_coef, 5)}")
    plt.scatter(x_num_objects, y_inference_time)
    plt.plot(linregress_x, linregress_y, "r", label=fr"Lineare Regression: $f(x) = {np.round(linregress.slope, 5)} \cdot x + {np.round(linregress.intercept, 5)}$")
    plt.ylabel("Inferenz-Zeit [s]")
    plt.xlabel("Anzahl gefundener Objekte")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "found_objects_to_inference_time.pdf")
    plt.savefig(plot_path)
    plt.show()

    del corr_coef, linregress, linregress_x, linregress_y


    # Per dataset plots
    for dataset_name, dataset_result in per_dataset_results.items():
        out_dir = os.path.join(output_dir, dataset_type_name, dataset_name)
        # Inference Time
        plt.figure(dpi=300)
        plt.title(f"Inferenz-Zeit pro Frame\n\n"
                  f"Datensatz: {dataset_name}")
        for neural_net_name, neural_net_result in dataset_result.items():
            dataset_neural_net_df = neural_net_result["dataframe"]
            plt.plot(range(len(neural_net_results["per_frame_results"])), dataset_neural_net_df["Inference Time (s)"], label=neural_net_name)
        plt.xlabel("Frame-Nummer")
        plt.ylabel("Inferenz-Zeit [s]")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()

        plot_path = os.path.join(out_dir, "inference_time.pdf")
        plt.savefig(plot_path)
        plt.show()

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
        plt.show()

        # Distance error
        frames = range(len(neural_net_results["per_frame_results"]))
        plt.figure(dpi=300)
        plt.title(f"Distanzfehler pro Frame\n\n"
                  f"Datensatz: {dataset_name}")
        for neural_net_name, neural_net_result in dataset_result.items():
            dataset_neural_net_df = neural_net_result["dataframe"]
            distance_errors = dataset_neural_net_df["Mean Distance Error (m)"]
            distance_std = dataset_neural_net_df["STD Distance Error (m)"]
            plt.plot(frames, distance_errors, label=f"Mittlerer Distanzfehler ({neural_net_name})")
            plt.fill_between(frames, distance_errors - distance_std, distance_errors + distance_std, alpha=0.2)
            #plt.plot(frames, dataset_neural_net_df["Max Distance Error (m)"], label="Maximaler Distanzfehler")  TODO test reenable with more neural nets
            #plt.plot(frames, dataset_neural_net_df["Min Distance Error (m)"], label="Minimaler Distanzfehler")  TODO test reenable with more neural nets
        plt.xlabel("Frame-Nummer")
        plt.ylabel("Distanzfehler [m]")
        plt.legend()
        # plt.ylim([0, ceil_to_pos(np.max(distance_errors), -1)])
        plt.grid(axis="y")
        plt.tight_layout()

        plot_path = os.path.join(out_dir, "distance_error.pdf")
        plt.savefig(plot_path)
        plt.show()

        del distance_errors, distance_std

        # Rotation error
        frames = range(len(neural_net_results["per_frame_results"]))
        plt.figure(dpi=300)
        plt.title(f"Rotationsfehler pro Frame\n\n"
                  f"Datensatz: {dataset_name}")
        for neural_net_name, neural_net_result in dataset_result.items():
            dataset_neural_net_df = neural_net_result["dataframe"]
            rotation_errors = dataset_neural_net_df["Mean Rotation Error"]
            rotation_std = dataset_neural_net_df["STD Rotation Error"]
            plt.plot(frames, rotation_errors, label=f"Mittlerer Rotationsfehler ({neural_net_name})")
            plt.fill_between(frames, rotation_errors - rotation_std, rotation_errors + rotation_std, alpha=0.2)
            #plt.plot(frames, dataset_neural_net_df["Max Rotation Error"], label="Maximaler Rotationsfehler") TODO test reenable with more neural nets
            #plt.plot(frames, dataset_neural_net_df["Min Rotation Error"], label="Minimaler Rotationsfehler") TODO test reenable with more neural nets
        plt.xlabel("Frame-Nummer")
        plt.ylabel("Rotationsfehler")
        plt.legend()
        # plt.ylim([0, ceil_to_pos(np.max(distance_errors), -1)])  TODO: 0 to 1
        plt.grid(axis="y")
        plt.tight_layout()

        plot_path = os.path.join(out_dir, "rotation_error.pdf")
        plt.savefig(plot_path)
        plt.show()

        del rotation_errors, rotation_std

        # Scale error
        frames = range(len(neural_net_results["per_frame_results"]))
        plt.figure(dpi=300)
        plt.title(f"Skalierungsfehler pro Frame\n\n"
                  f"Datensatz: {dataset_name}")
        for neural_net_name, neural_net_result in dataset_result.items():
            dataset_neural_net_df = neural_net_result["dataframe"]
            scale_errors = dataset_neural_net_df["Mean Scale Error"]
            scale_std = dataset_neural_net_df["STD Scale Error"]
            plt.plot(frames, scale_errors, label=f"Mittlerer Skalierungsfehler ({neural_net_name})")
            plt.fill_between(frames, scale_errors - scale_std, scale_errors + scale_std, alpha=0.2)
            #plt.plot(frames, dataset_neural_net_df["Max Scale Error"], label="Maximaler Skalierungsfehler")
            #plt.plot(frames, dataset_neural_net_df["Min Scale Error"], label="Minimaler Skalierungsfehler")
        plt.xlabel("Frame-Nummer")
        plt.ylabel("Skalierungsfehler")
        plt.legend()
        # plt.ylim([0, ceil_to_pos(np.max(distance_errors), -1)])
        plt.grid(axis="y")
        plt.tight_layout()

        plot_path = os.path.join(out_dir, "scale_error.pdf")
        plt.savefig(plot_path)
        plt.show()

        del scale_errors, scale_std

        # Correct and incorrect classifications count
        frames = range(len(neural_net_results["per_frame_results"]))
        plt.figure(dpi=300)
        plt.title(f"Korrekte und inkorrekte Klassifikationen pro Frame\n\n"
                  f"Datensatz: {dataset_name}")
        all_correct_classifications = []
        for neural_net_name, neural_net_result in dataset_result.items():
            dataset_neural_net_df = neural_net_result["dataframe"]
            correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
            incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
            plt.plot(frames, correct_classifications_per_frame, label=f"Korrekte Klassifizierungen ({neural_net_name})")
            plt.plot(frames, incorrect_classifications_per_frame, label=f"Inkorrekte Klassifizierungen ({neural_net_name})")
            all_correct_classifications.extend(correct_classifications_per_frame)
        plt.xlabel("Frame-Nummer")
        plt.ylabel("Anzahl Klassifizierungen")
        plt.legend()
        plt.ylim([0, ceil_to_pos(np.max(all_correct_classifications), 1)])
        plt.grid(axis="y")
        plt.tight_layout()

        plot_path = os.path.join(out_dir, "classification_counts.pdf")
        plt.savefig(plot_path)
        plt.show()

        # Correct and incorrect classifications ratio
        plt.figure(dpi=300)
        plt.title(f"Anteil Korrekte und inkorrekte Klassifikationen pro Frame\n\n"
                  f"Datensatz: {dataset_name}")
        for neural_net_name, neural_net_result in dataset_result.items():
            dataset_neural_net_df = neural_net_result["dataframe"]
            correct_classifications_per_frame = dataset_neural_net_df["Correct Classification Count"]
            incorrect_classifications_per_frame = dataset_neural_net_df["Incorrect Classification Count"]
            correct_classification_ratio = correct_classifications_per_frame / (
                        correct_classifications_per_frame + incorrect_classifications_per_frame)
            incorrect_classification_ratio = 1 - correct_classification_ratio
            plt.plot(frames,correct_classification_ratio, label=f"Korrekte Klassifizierungen ({neural_net_name})")
            plt.plot(frames, incorrect_classification_ratio, label=f"Inkorrekte Klassifizierungen ({neural_net_name})")
        plt.xlabel("Frame-Nummer")
        plt.ylabel("Anteil Klassifizierungen")
        plt.legend()
        plt.ylim([0, 1.05])
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.grid(axis="y")
        plt.tight_layout()

        plot_path = os.path.join(out_dir, "classification_ratio.pdf")
        plt.savefig(plot_path)
        plt.show()

        del correct_classifications_per_frame, incorrect_classifications_per_frame

        breakpoint()



    # Vergleich Netze zu verschiedenen Materialien


if __name__ == '__main__':
    eval_dataset_path = "/Users/flo/eval_dataset"
    output_dir = "output"
    #eval_dataset_path = "C:\\Users\\floriand\\eval_dataset"
    #output_dir = r".\\output"
    main(eval_dataset_path, output_dir)
