import json
import os

import numpy as np
import trimesh
from PIL import Image

from .converter_interface import ConverterInterface
from .dataset_helper import bop, replicator


class ReplicatorToBop(ConverterInterface):
    """
    Converts an NVIDIA Replicator dataset to the BOP format used by T-LESS, LM-O and YCB-V.
    """

    _OBJECT_PRIM_MARKER = "/objects/object_"

    _DEFAULT_ARGS = {
        "split": "train_pbr",
        "val_split": "val_pbr",
        "rgb_ext": "jpg",
        "depth_scale": 1.0,
        "amodal_masks": True,
        "detect_symmetries": False,
        "sym_tolerance": 0.015,
        "sym_max_fold": 12,
        "symmetries_file": "",
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

        if args["rgb_ext"] not in ("jpg", "png"):
            raise RuntimeError("rgb_ext must be 'jpg' or 'png', got '{}'.".format(args["rgb_ext"]))
        if float(args["depth_scale"]) <= 0.0:
            raise RuntimeError("depth_scale must be positive, got {}.".format(args["depth_scale"]))
        if args["split"] == args["val_split"]:
            raise RuntimeError("split and val_split must differ, both are '{}'.".format(args["split"]))

        args["depth_scale"] = float(args["depth_scale"])
        args["sym_tolerance"] = float(args["sym_tolerance"])
        args["sym_max_fold"] = int(args["sym_max_fold"])

        return args

    @staticmethod
    def _load_symmetries_override(symmetries_file: str) -> dict:
        """
        Loads the optional symmetry override file.

        :param symmetries_file: Path to the JSON file, or an empty string if there is none.
        :return: Symmetries keyed by semantic label or object ID, empty if no file was given.
        """

        if not symmetries_file:
            return {}
        if not os.path.isfile(symmetries_file):
            raise RuntimeError("The symmetries file '{}' does not exist.".format(symmetries_file))

        with open(symmetries_file) as file:
            return json.load(file)

    @staticmethod
    def _load_meshes(obj_paths_labels_ids: list[dict]) -> dict:
        """
        Loads the CAD models used for the amodal silhouettes.

        The vertices are scaled from meters to millimeters and are otherwise left alone, matching
        the models written to models/ and the frame the object poses refer to.

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
        Assigns the Replicator frames to the BOP splits.

        The generators write the train/val assignment as global frame indices to
        train_val_scenes.json. If that file is missing, or if no validation split was requested,
        every frame goes into the training split.

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

        Objects are ordered by their index in the prim path, so that the BOP ground truth ID of an
        object is stable and reproducible.

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
    def _write_rgb(rep_data_path: str, nr: str, scene_dir: str, im_id: int, rgb_ext: str) -> None:
        """
        Converts the colour image of a frame to the BOP layout.

        :param rep_data_path: Path to the Replicator dataset.
        :param nr: Frame number as a zero-padded string.
        :param scene_dir: Directory of the BOP scene.
        :param im_id: BOP image ID.
        :param rgb_ext: Extension of the colour images, 'jpg' or 'png'.
        """

        rgb_dir = os.path.join(scene_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

        # The Replicator images are RGBA with a constant alpha channel
        image = Image.open(os.path.join(rep_data_path, f"rgb_{nr}.png")).convert("RGB")
        image.save(os.path.join(rgb_dir, f"{im_id:06d}.{rgb_ext}"), quality=95)

    @staticmethod
    def _write_depth(depth_meters: np.ndarray, scene_dir: str, im_id: int, depth_scale: float) -> None:
        """
        Writes the depth image of a frame as a 16 bit PNG.

        Unlike the ROCA converter the depth is not masked to the object pixels: BOP expects the full
        scene depth, and both px_count_valid and the visibility test of bop_toolkit depend on it.

        :param depth_meters: Planar depth in meters as read from the Replicator .npy file.
        :param scene_dir: Directory of the BOP scene.
        :param im_id: BOP image ID.
        :param depth_scale: Millimeters per unit of the 16 bit depth image.
        """

        depth_dir = os.path.join(scene_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

        depth_units = np.clip(depth_meters * 1000.0 / depth_scale, 0, 65535).astype(np.uint16)
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
                       obj_id_by_label: dict, args: dict) -> tuple[dict, list, list]:
        """
        Converts a single Replicator frame to the BOP per-image annotations and images.

        :param rep_data_path: Path to the Replicator dataset.
        :param nr: Frame number as a zero-padded string.
        :param scene_dir: Directory of the BOP scene.
        :param im_id: BOP image ID.
        :param meshes: Vertices and faces of the CAD models, keyed by semantic label.
        :param obj_id_by_label: BOP object ID per semantic label.
        :param args: The parsed plugin arguments.
        :return: (scene_camera entry, scene_gt entries, scene_gt_info entries) for this image.
        """

        with open(os.path.join(rep_data_path, f"camera_params_{nr}.json")) as file:
            camera_params = json.load(file)

        cam_K = bop.camera_params_to_cam_K(camera_params)
        world_to_camera = bop.camera_params_to_world_to_camera(camera_params)
        width, height = camera_params["renderProductResolution"]

        depth_meters = np.load(os.path.join(rep_data_path, f"distance_to_image_plane_{nr}.npy"))
        instance_ids = np.array(Image.open(os.path.join(rep_data_path, f"instance_segmentation_{nr}.png")))

        ReplicatorToBop._write_rgb(rep_data_path, nr, scene_dir, im_id, args["rgb_ext"])
        ReplicatorToBop._write_depth(depth_meters, scene_dir, im_id, args["depth_scale"])

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
            # The visible surface is part of the silhouette, so folding it in keeps
            # mask_visib <= mask and bbox_visib <= bbox_obj exact despite subpixel disagreement
            mask_amodal = bop.silhouette_in_image(silhouette, x_origin, y_origin, (width, height))
            mask_amodal |= mask_visib
            if args["amodal_masks"]:
                ReplicatorToBop._write_mask(mask_amodal, scene_dir, "mask", im_id, gt_id)

            scene_gt_info.append(ReplicatorToBop._object_gt_info(
                mask_visib, mask_amodal, silhouette, silhouette_area, x_origin, y_origin, depth_meters))

        return {"cam_K": cam_K, "depth_scale": args["depth_scale"]}, scene_gt, scene_gt_info

    @staticmethod
    def _object_gt_info(mask_visib: np.ndarray, mask_amodal: np.ndarray, silhouette: np.ndarray,
                        silhouette_area: float, x_origin: int, y_origin: int,
                        depth_meters: np.ndarray) -> dict:
        """
        Calculates the scene_gt_info entry of a single object.

        The fields follow bop_toolkit/scripts/calc_gt_info.py: px_count_all counts the whole
        silhouette including the part truncated by the image border, which is why it is taken from
        the unclipped silhouette and why bbox_obj may reach outside the image.

        :param mask_visib: Visible mask of the object, clipped to the image.
        :param mask_amodal: Amodal mask of the object, clipped to the image.
        :param silhouette: Unclipped amodal silhouette of the object.
        :param silhouette_area: Subpixel accurate area of the silhouette in whole pixels.
        :param x_origin: Image x coordinate of the top left pixel of the silhouette.
        :param y_origin: Image y coordinate of the top left pixel of the silhouette.
        :param depth_meters: Depth image of the frame in meters.
        :return: The scene_gt_info entry.
        """

        # The subpixel area, not the mask pixel count, so small objects keep a sane visib_fract
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
                       meshes: dict, obj_id_by_label: dict, args: dict) -> None:
        """
        Converts all frames of one split into a single BOP scene.

        Every source dataset becomes one BOP scene per split. The images are renumbered from zero,
        because the Replicator frame numbers of a split are not contiguous; frame_index.json keeps
        the mapping back to the original frames.

        :param rep_data_path: Path to the Replicator dataset.
        :param output_dir: Root of the BOP dataset.
        :param split_name: Directory name of the split.
        :param frame_numbers: Frame numbers of this split as zero-padded strings.
        :param meshes: Vertices and faces of the CAD models, keyed by semantic label.
        :param obj_id_by_label: BOP object ID per semantic label.
        :param args: The parsed plugin arguments.
        """

        scene_dir = os.path.join(output_dir, split_name, "000000")
        os.makedirs(scene_dir, exist_ok=True)

        scene_camera = {}
        scene_gt = {}
        scene_gt_info = {}
        frame_index = {}

        for im_id, nr in enumerate(frame_numbers):
            camera_entry, gt_entries, gt_info_entries = ReplicatorToBop._convert_frame(
                rep_data_path, nr, scene_dir, im_id, meshes, obj_id_by_label, args)

            # The keys are unpadded integer strings, unlike the six digit padded file names
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
    def _write_camera_json(rep_data_path: str, output_dir: str, nr: str, depth_scale: float) -> None:
        """
        Writes the dataset level camera.json.

        The intrinsics are constant across the dataset, so the parameters of one frame describe all
        of them. bop_toolkit reads this file, ROCA's BopDataset does not.

        :param rep_data_path: Path to the Replicator dataset.
        :param output_dir: Root of the BOP dataset.
        :param nr: Frame number the parameters are taken from.
        :param depth_scale: Millimeters per unit of the 16 bit depth images.
        """

        with open(os.path.join(rep_data_path, f"camera_params_{nr}.json")) as file:
            camera_params = json.load(file)

        cam_K = bop.camera_params_to_cam_K(camera_params)
        width, height = camera_params["renderProductResolution"]

        camera = {
            "cx": cam_K[2], "cy": cam_K[5], "depth_scale": depth_scale,
            "fx": cam_K[0], "fy": cam_K[4], "height": height, "width": width,
        }

        with open(os.path.join(output_dir, "camera.json"), "w") as file:
            json.dump(camera, file, indent=2)

    @staticmethod
    def _write_dataset_info(output_dir: str, im_size: tuple[int, int], splits: dict,
                            obj_id_by_label: dict, args: dict) -> None:
        """
        Writes dataset_info.md, the free-form description a BOP dataset ships with.

        Besides describing the dataset it carries the dataset_params.py entry needed to run the
        bop_toolkit scripts on it, filled in with this dataset's actual values.

        :param output_dir: Root of the BOP dataset.
        :param im_size: Image size as (width, height).
        :param splits: Frame numbers per split name.
        :param obj_id_by_label: BOP object ID per semantic label.
        :param args: The parsed plugin arguments.
        """

        name = args["dataset_name"]
        objects = "\n".join(f"| {obj_id} | `{label}` |" for label, obj_id
                             in sorted(obj_id_by_label.items(), key=lambda item: item[1]))
        split_rows = "\n".join(f"| `{split}` | 000000 | {len(frames)} |"
                                for split, frames in splits.items() if frames)

        # A pbr split type makes bop_toolkit override scene_ids with list(range(50))
        split_hint = ""
        if any(split.endswith("_pbr") for split, frames in splits.items() if frames):
            split_hint = (
                "\nNote that `get_split_params` unconditionally overrides `scene_ids` with "
                "`list(range(50))` for a `pbr` split type. Either convert with a split name that "
                "carries no split type, for example `split=train`, or add this dataset to that "
                "exception as well.\n")

        content = f"""# {name}

Synthetic dataset in BOP format, generated with the SDGP and converted by the `ReplicatorToBop`
converter plugin. See `dataset_converter/README.md` in the SDGP repository for the conversion
options and the known limitations.

## Splits

| Split | Scene | Images |
|---|---|---|
{split_rows}

## Objects

| obj_id | Semantic label |
|---|---|
{objects}

## Images

- Resolution: {im_size[0]}x{im_size[1]}
- Colour: `.{args["rgb_ext"]}`
- Depth: 16 bit PNG, `depth_scale` {args["depth_scale"]}, i.e. one unit is {args["depth_scale"]} mm
- Models: millimeters, origin at the CAD origin rather than the bounding box center

## Using bop_toolkit

`bop_toolkit_lib/dataset_params.py` only knows the datasets hardcoded in `get_split_params`, so add:

```python
elif dataset_name == "{name}":
    p["scene_ids"] = [0]
    p["im_size"] = ({im_size[0]}, {im_size[1]})
```
{split_hint}"""

        with open(os.path.join(output_dir, "dataset_info.md"), "w") as file:
            file.write(content)

    @staticmethod
    def _write_test_targets(output_dir: str, split_name: str) -> None:
        """
        Writes test_targets_bop19.json for a split, listing every object instance of every image.

        This file is what the BOP evaluation scripts iterate over.

        :param output_dir: Root of the BOP dataset.
        :param split_name: Directory name of the split the targets are built from.
        """

        with open(os.path.join(output_dir, split_name, "000000", "scene_gt.json")) as file:
            scene_gt = json.load(file)

        targets = []
        for im_id_str, entries in sorted(scene_gt.items(), key=lambda item: int(item[0])):
            instance_counts = {}
            for entry in entries:
                instance_counts[entry["obj_id"]] = instance_counts.get(entry["obj_id"], 0) + 1
            for obj_id, count in sorted(instance_counts.items()):
                targets.append({"im_id": int(im_id_str), "inst_count": count,
                                "obj_id": obj_id, "scene_id": 0})

        with open(os.path.join(output_dir, "test_targets_bop19.json"), "w") as file:
            json.dump(targets, file)

        print(f"  test_targets_bop19.json: {len(targets)} targets from '{split_name}'")

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

        # get_scene_nrs returns them in glob order, which is not sorted
        scene_numbers = sorted(replicator.get_scene_nrs(replicator_data_dir))
        if not scene_numbers:
            raise RuntimeError("No camera_params_*.json files found in '{}'.".format(replicator_data_dir))

        obj_paths_labels_ids = bop.obj_paths_semantic_labels_and_obj_ids(obj_files_dir)
        if not obj_paths_labels_ids:
            raise RuntimeError("No OBJ files found in '{}'.".format(obj_files_dir))
        obj_id_by_label = {entry["semantic_label"]: entry["obj_id"] for entry in obj_paths_labels_ids}

        print("Converting CAD models...")
        bop.write_models(obj_paths_labels_ids, os.path.join(output_dir, "models"),
                         args["detect_symmetries"], args["sym_tolerance"], args["sym_max_fold"],
                         ReplicatorToBop._load_symmetries_override(args["symmetries_file"]))

        with open(os.path.join(output_dir, "obj_id_map.json"), "w") as file:
            json.dump(obj_id_by_label, file, indent=2)

        print("Writing camera.json...")
        ReplicatorToBop._write_camera_json(replicator_data_dir, output_dir, scene_numbers[0],
                                           args["depth_scale"])

        print("Converting frames...")
        meshes = ReplicatorToBop._load_meshes(obj_paths_labels_ids)
        splits = ReplicatorToBop._split_frames(replicator_data_dir, scene_numbers, args)
        for split_name, frame_numbers in splits.items():
            if not frame_numbers:
                print(f"  {split_name}: no frames, skipped")
                continue
            ReplicatorToBop._convert_split(replicator_data_dir, output_dir, split_name,
                                           frame_numbers, meshes, obj_id_by_label, args)

        with open(os.path.join(replicator_data_dir, f"camera_params_{scene_numbers[0]}.json")) as file:
            im_size = tuple(json.load(file)["renderProductResolution"])
        ReplicatorToBop._write_dataset_info(output_dir, im_size, splits, obj_id_by_label, args)

        test_splits = [name for name in splits if "test" in name and splits[name]]
        if test_splits:
            print("Generating test_targets_bop19.json...")
            ReplicatorToBop._write_test_targets(output_dir, test_splits[0])

        print("Done!")
