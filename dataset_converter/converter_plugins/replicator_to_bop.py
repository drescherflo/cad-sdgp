import json
import os

import numpy as np
import trimesh
from PIL import Image

from .converter_interface import ConverterInterface
from .dataset_helper import bop, replicator


class ReplicatorToBop(ConverterInterface):
    """
    Converts an NVIDIA Replicator dataset to the BOP format.
    """

    _OBJECT_PRIM_MARKER = "/objects/object_"

    # bop_toolkit expects .jpg for a *_pbr split, and one depth unit per millimeter is the BOP default
    _RGB_EXT = "jpg"
    _DEPTH_SCALE = 1.0

    # The visibility an object needs to be worth localizing, as in bop_toolkit enumerate_test_targets.py
    _BOP19_MIN_VISIB_FRACT = 0.1

    _DEFAULT_ARGS = {
        "split": "train_pbr",
        "val_split": "val_pbr",
        "dataset_name": "sodah",
    }

    @staticmethod
    def _parse_args(kwargs: dict) -> dict:
        """
        Validates the plugin arguments and fills in the defaults.

        :param kwargs: Arguments passed on the command line as key=value pairs.
        :return: The complete argument dictionary.
        """

        unknown = sorted(set(kwargs) - set(ReplicatorToBop._DEFAULT_ARGS))
        if unknown:
            raise RuntimeError("Unknown argument(s) for ReplicatorToBop: {}. Supported: {}.".format(
                ", ".join(unknown), ", ".join(sorted(ReplicatorToBop._DEFAULT_ARGS))))

        args = dict(ReplicatorToBop._DEFAULT_ARGS)
        args.update(kwargs)

        if args["split"] == args["val_split"]:
            raise RuntimeError("split and val_split must differ, both are '{}'.".format(args["split"]))

        return args

    @staticmethod
    def _load_meshes(obj_paths_labels_ids: list[dict]) -> dict:
        """
        Loads the CAD models used for the amodal silhouettes, in millimeters like models/.

        :param obj_paths_labels_ids: OBJ paths, semantic labels and object IDs of the models.
        :return: Vertices and faces of each model, keyed by semantic label.
        """

        meshes = {}
        for obj_path_label_id in obj_paths_labels_ids:
            mesh = trimesh.load(obj_path_label_id["obj_file"], force="mesh")
            meshes[obj_path_label_id["semantic_label"]] = {
                "vertices": np.asarray(mesh.vertices, dtype=float) * 1000.0,
                "faces": np.asarray(mesh.faces),
            }

        return meshes

    @staticmethod
    def _split_frames(rep_data_path: str, scene_numbers: list[str], args: dict) -> dict:
        """
        Assigns the Replicator frames to the BOP splits, following train_val_scenes.json.

        :param rep_data_path: Path to the Replicator dataset.
        :param scene_numbers: Sorted list of frame numbers as zero-padded strings.
        :param args: The parsed plugin arguments.
        :return: Frame numbers per split name, in ascending order.
        """

        split_file = os.path.join(rep_data_path, "train_val_scenes.json")
        if not args["val_split"] or not os.path.isfile(split_file):
            if args["val_split"]:
                print(f"  Warning: {split_file} not found, putting all frames into '{args['split']}'")
            return {args["split"]: scene_numbers}

        with open(split_file) as file:
            train_val_scenes = json.load(file)

        val_numbers = {int(number) for number in train_val_scenes["val_scenes"]}
        train_frames = [nr for nr in scene_numbers if int(nr) not in val_numbers]
        val_frames = [nr for nr in scene_numbers if int(nr) in val_numbers]

        return {args["split"]: train_frames, args["val_split"]: val_frames}

    @staticmethod
    def _visible_objects(rep_data_path: str, nr: str) -> list[dict]:
        """
        Reads the visible objects of a frame and joins them with their instance segmentation IDs.

        Ordered by prim path index, so that the BOP ground truth IDs are reproducible.

        :param rep_data_path: Path to the Replicator dataset.
        :param nr: Frame number as a zero-padded string.
        :return: One dictionary per visible object with its pose, semantic label and instance ID.
        """

        with open(os.path.join(rep_data_path, f"world_pose_visible_objects_{nr}.json")) as file:
            world_poses = json.load(file)
        with open(os.path.join(rep_data_path, f"instance_segmentation_mapping_{nr}.json")) as file:
            instance_mapping = json.load(file)

        instance_id_by_prim = {
            prim_path: int(instance_id) for instance_id, prim_path in instance_mapping.items()
            if ReplicatorToBop._OBJECT_PRIM_MARKER in prim_path
        }

        objects = []
        for world_pose in world_poses:
            prim_path = world_pose["prim_path"]
            if prim_path not in instance_id_by_prim:
                continue
            objects.append({
                "prim_path": prim_path,
                "semantic_label": world_pose["semantic_labels"]["class"],
                "instance_id": instance_id_by_prim[prim_path],
                "position": world_pose["pose"]["position"],
                "orientation": world_pose["pose"]["orientation"],
            })

        return sorted(objects, key=lambda obj: int(obj["prim_path"].rsplit("object_", 1)[-1]))

    @staticmethod
    def _write_rgb(rep_data_path: str, nr: str, scene_dir: str, im_id: int) -> None:
        """
        Converts the colour image of a frame to the BOP layout.

        :param rep_data_path: Path to the Replicator dataset.
        :param nr: Frame number as a zero-padded string.
        :param scene_dir: Directory of the BOP scene.
        :param im_id: BOP image ID.
        """

        rgb_dir = os.path.join(scene_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

        # The Replicator images are RGBA with a constant alpha channel
        image = Image.open(os.path.join(rep_data_path, f"rgb_{nr}.png")).convert("RGB")
        image.save(os.path.join(rgb_dir, f"{im_id:06d}.{ReplicatorToBop._RGB_EXT}"), quality=95)

    @staticmethod
    def _write_depth(depth_meters: np.ndarray, scene_dir: str, im_id: int) -> None:
        """
        Writes the depth image of a frame as a 16 bit PNG.

        :param depth_meters: Planar depth in meters as read from the Replicator .npy file.
        :param scene_dir: Directory of the BOP scene.
        :param im_id: BOP image ID.
        """

        depth_dir = os.path.join(scene_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

        depth_units = np.clip(depth_meters * 1000.0 / ReplicatorToBop._DEPTH_SCALE, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_units).save(os.path.join(depth_dir, f"{im_id:06d}.png"))

    @staticmethod
    def _write_mask(mask: np.ndarray, scene_dir: str, mask_dir_name: str, im_id: int, gt_id: int) -> None:
        """
        Writes a single object mask.

        :param mask: Boolean mask of the object.
        :param scene_dir: Directory of the BOP scene.
        :param mask_dir_name: Either 'mask' or 'mask_visib'.
        :param im_id: BOP image ID.
        :param gt_id: Index of the object in the scene_gt entry of this image.
        """

        mask_dir = os.path.join(scene_dir, mask_dir_name)
        os.makedirs(mask_dir, exist_ok=True)

        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(mask_dir, f"{im_id:06d}_{gt_id:06d}.png"))

    @staticmethod
    def _convert_frame(rep_data_path: str, nr: str, scene_dir: str, im_id: int, meshes: dict,
                       obj_id_by_label: dict) -> tuple[dict, list, list]:
        """
        Converts a single Replicator frame to the BOP per-image annotations and images.

        :param rep_data_path: Path to the Replicator dataset.
        :param nr: Frame number as a zero-padded string.
        :param scene_dir: Directory of the BOP scene.
        :param im_id: BOP image ID.
        :param meshes: Vertices and faces of the CAD models, keyed by semantic label.
        :param obj_id_by_label: BOP object ID per semantic label.
        :return: (scene_camera entry, scene_gt entries, scene_gt_info entries) for this image.
        """

        with open(os.path.join(rep_data_path, f"camera_params_{nr}.json")) as file:
            camera_params = json.load(file)

        f_x, f_y, c_x, c_y = replicator.camera_params_to_intrinsics(camera_params)
        cam_K = [f_x, 0.0, c_x, 0.0, f_y, c_y, 0.0, 0.0, 1.0]
        world_to_camera = bop.camera_params_to_world_to_camera(camera_params)
        width, height = camera_params["renderProductResolution"]

        depth_meters = np.load(os.path.join(rep_data_path, f"distance_to_image_plane_{nr}.npy"))
        instance_ids = np.array(Image.open(os.path.join(rep_data_path, f"instance_segmentation_{nr}.png")))

        ReplicatorToBop._write_rgb(rep_data_path, nr, scene_dir, im_id)
        ReplicatorToBop._write_depth(depth_meters, scene_dir, im_id)

        scene_gt = []
        scene_gt_info = []
        for gt_id, obj in enumerate(ReplicatorToBop._visible_objects(rep_data_path, nr)):
            semantic_label = obj["semantic_label"]
            if semantic_label not in obj_id_by_label:
                raise RuntimeError(
                    "No matching object id found for semantic label '{}'. Known labels: {}.".format(
                        semantic_label, ", ".join(sorted(obj_id_by_label))))

            cam_R_m2c, cam_t_m2c = bop.world_pose_to_camera_frame(
                obj["position"], obj["orientation"], world_to_camera)

            scene_gt.append({
                "cam_R_m2c": cam_R_m2c.reshape(9).tolist(),
                "cam_t_m2c": cam_t_m2c.tolist(),
                "obj_id": obj_id_by_label[semantic_label],
            })

            # Visible mask, taken straight from the Replicator instance segmentation
            mask_visib = instance_ids == obj["instance_id"]
            ReplicatorToBop._write_mask(mask_visib, scene_dir, "mask_visib", im_id, gt_id)

            # Amodal silhouette, rasterized from the CAD model in the ground truth pose
            mesh = meshes[semantic_label]
            silhouette, silhouette_area, x_origin, y_origin = bop.rasterize_silhouette(
                mesh["vertices"], mesh["faces"], cam_R_m2c, cam_t_m2c,
                np.array(cam_K).reshape(3, 3), (width, height))
            # Folding in the visible surface keeps mask_visib <= mask exact
            mask_amodal = bop.silhouette_in_image(silhouette, x_origin, y_origin, (width, height))
            mask_amodal |= mask_visib
            ReplicatorToBop._write_mask(mask_amodal, scene_dir, "mask", im_id, gt_id)

            scene_gt_info.append(ReplicatorToBop._object_gt_info(
                mask_visib, mask_amodal, silhouette, silhouette_area, x_origin, y_origin, depth_meters))

        return {"cam_K": cam_K, "depth_scale": ReplicatorToBop._DEPTH_SCALE}, scene_gt, scene_gt_info

    @staticmethod
    def _object_gt_info(mask_visib: np.ndarray, mask_amodal: np.ndarray, silhouette: np.ndarray,
                        silhouette_area: float, x_origin: int, y_origin: int,
                        depth_meters: np.ndarray) -> dict:
        """
        Calculates the scene_gt_info entry of a single object, following bop_toolkit calc_gt_info.py.

        px_count_all and bbox_obj come from the unclipped silhouette, so truncated parts still count.

        :param mask_visib: Visible mask of the object, clipped to the image.
        :param mask_amodal: Amodal mask of the object, clipped to the image.
        :param silhouette: Unclipped amodal silhouette of the object.
        :param silhouette_area: Subpixel accurate area of the silhouette in whole pixels.
        :param x_origin: Image x coordinate of the top left pixel of the silhouette.
        :param y_origin: Image y coordinate of the top left pixel of the silhouette.
        :param depth_meters: Depth image of the frame in meters.
        :return: The scene_gt_info entry.
        """

        # Subpixel area, not mask pixel count, so small objects keep a sane visib_fract
        px_count_visib = int(mask_visib.sum())
        px_count_all = max(int(round(silhouette_area)), px_count_visib)
        px_count_valid = int(np.count_nonzero(depth_meters[mask_amodal] > 0))
        visib_fract = px_count_visib / float(px_count_all) if px_count_all > 0 else 0.0

        bbox_obj = [-1, -1, -1, -1]
        bbox_visib = [-1, -1, -1, -1]
        if px_count_visib > 0:
            visib_ys, visib_xs = mask_visib.nonzero()
            bbox_visib = bop.calc_2d_bbox(visib_xs, visib_ys)
            silhouette_ys, silhouette_xs = silhouette.nonzero()
            bbox_obj = bop.union_2d_bbox(
                bop.calc_2d_bbox(silhouette_xs + x_origin, silhouette_ys + y_origin), bbox_visib)

        return {
            "bbox_obj": bbox_obj,
            "bbox_visib": bbox_visib,
            "px_count_all": px_count_all,
            "px_count_valid": px_count_valid,
            "px_count_visib": px_count_visib,
            "visib_fract": visib_fract,
        }

    @staticmethod
    def _convert_split(rep_data_path: str, output_dir: str, split_name: str, frame_numbers: list[str],
                       meshes: dict, obj_id_by_label: dict) -> None:
        """
        Converts all frames of one split into a single BOP scene.

        Images are renumbered from zero, since a split's frame numbers are not contiguous;
        frame_index.json maps back to the original frames.

        :param rep_data_path: Path to the Replicator dataset.
        :param output_dir: Root of the BOP dataset.
        :param split_name: Directory name of the split.
        :param frame_numbers: Frame numbers of this split as zero-padded strings.
        :param meshes: Vertices and faces of the CAD models, keyed by semantic label.
        :param obj_id_by_label: BOP object ID per semantic label.
        """

        scene_dir = os.path.join(output_dir, split_name, "000000")
        os.makedirs(scene_dir, exist_ok=True)

        scene_camera = {}
        scene_gt = {}
        scene_gt_info = {}
        frame_index = {}

        for im_id, nr in enumerate(frame_numbers):
            camera_entry, gt_entries, gt_info_entries = ReplicatorToBop._convert_frame(
                rep_data_path, nr, scene_dir, im_id, meshes, obj_id_by_label)

            # Unpadded integer keys, unlike the padded file names
            scene_camera[str(im_id)] = camera_entry
            scene_gt[str(im_id)] = gt_entries
            scene_gt_info[str(im_id)] = gt_info_entries
            frame_index[str(im_id)] = nr

            if (im_id + 1) % 250 == 0 or im_id + 1 == len(frame_numbers):
                print(f"    {split_name}: {im_id + 1}/{len(frame_numbers)} frames")

        for file_name, content in (("scene_camera.json", scene_camera), ("scene_gt.json", scene_gt),
                                   ("scene_gt_info.json", scene_gt_info), ("frame_index.json", frame_index)):
            with open(os.path.join(scene_dir, file_name), "w") as file:
                json.dump(content, file)

        annotations = sum(len(entries) for entries in scene_gt.values())
        print(f"  {split_name}: {len(frame_numbers)} images, {annotations} annotations")

    @staticmethod
    def _write_camera_json(rep_data_path: str, output_dir: str, nr: str) -> None:
        """
        Writes the dataset level camera.json, from one frame since the intrinsics are constant.

        :param rep_data_path: Path to the Replicator dataset.
        :param output_dir: Root of the BOP dataset.
        :param nr: Frame number the parameters are taken from.
        """

        with open(os.path.join(rep_data_path, f"camera_params_{nr}.json")) as file:
            camera_params = json.load(file)

        f_x, f_y, c_x, c_y = replicator.camera_params_to_intrinsics(camera_params)
        width, height = camera_params["renderProductResolution"]

        camera = {
            "cx": c_x, "cy": c_y, "depth_scale": ReplicatorToBop._DEPTH_SCALE,
            "fx": f_x, "fy": f_y, "height": height, "width": width,
        }

        with open(os.path.join(output_dir, "camera.json"), "w") as file:
            json.dump(camera, file, indent=2)

    @staticmethod
    def _write_dataset_info(output_dir: str, im_size: tuple[int, int], splits: dict,
                            obj_id_by_label: dict, args: dict) -> None:
        """
        Writes dataset_info.json, the dataset level description of the BOP layout.

        Also carries the semantic label of every obj_id, which is the only place the Replicator
        class names survive the conversion.

        :param output_dir: Root of the BOP dataset.
        :param im_size: Image size as (width, height).
        :param splits: Frame numbers per split name.
        :param obj_id_by_label: BOP object ID per semantic label.
        :param args: The parsed plugin arguments.
        """

        dataset_info = {
            "name": args["dataset_name"],
            "description": "Synthetic dataset in BOP format, generated with the SDGP and converted "
                           "by the ReplicatorToBop converter plugin.",
            "im_size": list(im_size),
            "rgb_ext": ReplicatorToBop._RGB_EXT,
            "depth_scale": ReplicatorToBop._DEPTH_SCALE,
            "models_unit": "mm",
            "splits": {split: {"scene_ids": [0], "im_count": len(frames)}
                       for split, frames in splits.items() if frames},
            "objects": {str(obj_id): label for label, obj_id in obj_id_by_label.items()},
        }

        with open(os.path.join(output_dir, "dataset_info.json"), "w") as file:
            json.dump(dataset_info, file, indent=2)

    @staticmethod
    def _write_test_targets(output_dir: str, split_name: str) -> None:
        """
        Writes the two test target files the BOP evaluation scripts iterate over.

        Both list the object instances per image and are built from the same split. bop19 drives the
        6D localization evaluation and covers only objects that are visible enough, bop24 the 6D
        detection evaluation and covers all of them. Both carry inst_count, which eval_calc_errors.py
        asserts on in localization mode.

        :param output_dir: Root of the BOP dataset.
        :param split_name: Directory name of the split the targets are built from.
        """

        scene_dir = os.path.join(output_dir, split_name, "000000")
        with open(os.path.join(scene_dir, "scene_gt.json")) as file:
            scene_gt = json.load(file)
        with open(os.path.join(scene_dir, "scene_gt_info.json")) as file:
            scene_gt_info = json.load(file)

        for file_name, min_visib_fract in (
                ("test_targets_bop19.json", ReplicatorToBop._BOP19_MIN_VISIB_FRACT),
                ("test_targets_bop24.json", 0.0)):
            targets = []
            for im_id, entries in sorted(scene_gt.items(), key=lambda item: int(item[0])):
                instance_counts = {}
                for gt_id, entry in enumerate(entries):
                    if scene_gt_info[im_id][gt_id]["visib_fract"] < min_visib_fract:
                        continue
                    instance_counts[entry["obj_id"]] = instance_counts.get(entry["obj_id"], 0) + 1
                for obj_id, count in sorted(instance_counts.items()):
                    targets.append({"im_id": int(im_id), "inst_count": count,
                                    "obj_id": obj_id, "scene_id": 0})

            with open(os.path.join(output_dir, file_name), "w") as file:
                json.dump(targets, file)

            print(f"  {file_name}: {len(targets)} targets from '{split_name}'")

    @staticmethod
    def convert(replicator_data_dir: str, obj_files_dir: str, output_dir: str, **kwargs) -> None:
        """
        Converts data from NVIDIA Replicator to the BOP format.

        :param replicator_data_dir: Directory containing the NVIDIA Replicator data.
        :param obj_files_dir: Directory containing the OBJ files.
        :param output_dir: Target directory for the converted data.
        :param kwargs: Plugin arguments, see ReplicatorToBop._DEFAULT_ARGS.
        """

        print("Converting the generated training data from NVIDIA Replicator to BOP format...")

        args = ReplicatorToBop._parse_args(kwargs)

        # get_scene_nrs returns glob order, which is unsorted
        scene_numbers = sorted(replicator.get_scene_nrs(replicator_data_dir))
        if not scene_numbers:
            raise RuntimeError("No camera_params_*.json files found in '{}'.".format(replicator_data_dir))

        obj_paths_labels_ids = bop.obj_paths_semantic_labels_and_obj_ids(obj_files_dir)
        if not obj_paths_labels_ids:
            raise RuntimeError("No OBJ files found in '{}'.".format(obj_files_dir))
        obj_id_by_label = {entry["semantic_label"]: entry["obj_id"] for entry in obj_paths_labels_ids}

        print("Converting CAD models...")
        bop.write_models(obj_paths_labels_ids, os.path.join(output_dir, "models"))

        print("Writing camera.json...")
        ReplicatorToBop._write_camera_json(replicator_data_dir, output_dir, scene_numbers[0])

        print("Converting frames...")
        meshes = ReplicatorToBop._load_meshes(obj_paths_labels_ids)
        splits = ReplicatorToBop._split_frames(replicator_data_dir, scene_numbers, args)
        for split_name, frame_numbers in splits.items():
            if not frame_numbers:
                print(f"  {split_name}: no frames, skipped")
                continue
            ReplicatorToBop._convert_split(replicator_data_dir, output_dir, split_name,
                                           frame_numbers, meshes, obj_id_by_label)

        with open(os.path.join(replicator_data_dir, f"camera_params_{scene_numbers[0]}.json")) as file:
            im_size = tuple(json.load(file)["renderProductResolution"])
        ReplicatorToBop._write_dataset_info(output_dir, im_size, splits, obj_id_by_label, args)

        # A split named test is the natural source, otherwise the validation split as the next best
        # held out data, and only failing that the training split
        candidates = [name for name in splits if "test" in name] + [args["val_split"], args["split"]]
        target_split = next((name for name in candidates if splits.get(name)), None)
        if target_split:
            print("Generating test targets...")
            ReplicatorToBop._write_test_targets(output_dir, target_split)

        print("Done!")
