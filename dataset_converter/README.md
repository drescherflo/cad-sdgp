# dataset_converter

## dataset_converter.py

The script `dataset_converter.py` converts datasets from the NVIDIA Replicator format to another format. It loads converter plugins from the `converter_plugins` package and applies them to the provided data.

### Arguments

- `--obj_dir`: **required**. Directory containing the source files in OBJ format. No default.
- `--replicator_data_dir`: **required**. Directory containing the source files in the Replicator data format. No default. Can be specified multiple times.
- `--output_dir`: **required**. Destination directory for the converted files. Each plugin receives its own subdirectory. No default.
- `--converter`: **required**. Configures a converter as `Name key=value ...`. No default. Can be specified multiple times, once per converter.

Plugin arguments are given as `key=value` pairs after the converter name, in the same style as the `--writer` flag of the dataset generators. Values that read `true` or `false` become booleans, numeric values become `int` or `float`, everything else stays a string.

### Example command

```bash
python dataset_converter.py \
  --obj_dir "/path/to/obj_directory" \
  --replicator_data_dir "/path/to/replicator_directory1" \
  --replicator_data_dir "/path/to/replicator_directory2" \
  --output_dir "/path/to/output_directory" \
  --converter ReplicatorToRoca \
  --converter ReplicatorToBop split=train_pbr
```

### Important notes

- The script requires that the `obj_dir` and `replicator_data_dir` directories exist and contain the appropriate files.
- The `output_dir` is used to store the converted files.
- Custom conversion routines can be added by creating your own plugin. Any plugin that inherits from the `ConverterInterface` class and resides in the `converter_plugins` package can be used as an argument for the `--converter` flag.

## Existing Converter Plugins

### ReplicatorToROCA

This plugin converts the output of NVIDIA Replicator into the format required by [ROCA](https://anonymous.4open.science/r/ROCA-5B00) for training.

### ReplicatorToBop

This plugin converts the output of NVIDIA Replicator into the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) used by T-LESS, LM-O and YCB-V. The result can be trained on directly, and it is laid out so that the `bop_toolkit` scripts can be pointed at it.

```
<output_dir>/
├── camera.json                          # intrinsics and depth_scale of the dataset
├── dataset_info.json                    # image size, splits and the obj_id -> semantic label map
├── test_targets_bop19.json              # evaluation targets, 6D localization
├── test_targets_bop24.json              # evaluation targets, 6D detection
├── models/
│   ├── obj_000001.ply ...               # CAD models in millimeters
│   └── models_info.json                 # extents and diameter of each model
├── models_eval/                         # a copy of models/, see below
│   ├── obj_000001.ply ...
│   └── models_info.json
├── train_pbr/000000/
│   ├── rgb/000000.jpg ...
│   ├── depth/000000.png                 # uint16, multiply by depth_scale for millimeters
│   ├── mask/000000_000000.png           # amodal masks
│   ├── mask_visib/000000_000000.png     # visible masks
│   ├── scene_camera.json
│   ├── scene_gt.json
│   ├── scene_gt_info.json
│   ├── scene_gt_coco.json               # dieselbe GT im COCO-Format
│   └── frame_index.json                 # BOP image id -> Replicator frame number
├── val_pbr/000000/                      # same structure
└── test_pbr/000000/                     # same structure
```

Each source dataset becomes one self-contained BOP dataset with a single scene per split. The images are renumbered from zero per split, and `frame_index.json` maps back to the original Replicator frames.

#### Splits

The frames are split into training, validation and test by fraction, 60/20/20 by default. The `train_val_scenes.json` the dataset generators write is **ignored**: it only knows a train/val assignment, while BOP expects a test split as well.

Frames are assigned at random, the same way the generators build their own train/val split with `train_test_split`. The shuffle is seeded, so converting a dataset twice yields the same assignment and no frame silently moves from training to test between runs. Set `val_fraction=0` or `test_fraction=0`, or an empty `val_split=`/`test_split=`, to leave a split out.

Note that on the conveyor belt the frames of one object run are consecutive and nearly identical, so a random per-frame split puts neighbouring frames of the same run into different splits. Validation and test numbers from that data are optimistic; the 6-DOF frames are independent and unaffected.

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `split` | `train_pbr` | Directory name of the training split |
| `val_split` | `val_pbr` | Directory name of the validation split. Empty drops the split |
| `test_split` | `test_pbr` | Directory name of the test split. Empty drops the split |
| `val_fraction` | `0.2` | Share of the frames that go into the validation split |
| `test_fraction` | `0.2` | Share of the frames that go into the test split |
| `dataset_name` | `sodah` | Name of the dataset, recorded in `dataset_info.json` |

Colour images are always written as `.jpg` (what `bop_toolkit` expects for a `*_pbr` split), the depth images always use `depth_scale` `1.0`, i.e. one unit per millimeter, and `mask/` is always written.

#### models_eval

`eval_calc_errors.py` hardcodes the `eval` model type, so a `models_eval/` directory has to exist for the BOP evaluation to run at all. The official datasets put uniformly remeshed models there, because the vertices are used as sample points for the pose errors.

Here it is a plain copy of `models/`. MSSD and MSPD take a *maximum* over the model vertices and the extremal points are already present in the CAD mesh, so remeshing changes nothing for the BOP19 and BOP24 scores, and VSD renders the mesh anyway. Only the averaging metrics ADD and ADI would benefit, and producing a genuine remesh needs MeshLab or Blender (`remesh_models_for_eval.py`), which is out of scope here. If you evaluate with ADD or ADI, remesh the models yourself first.

#### COCO annotations

`scene_gt_coco.json` is the same ground truth in COCO format, so the dataset can be fed to a standard 2D detection framework and to the BOP Challenge 2022+ detection and segmentation evaluation (`eval_bop22_coco.py`) without a further conversion. It holds no information that is not already in `scene_gt.json`, `scene_gt_info.json` and the mask directories.

Following `calc_gt_coco.py`, the `segmentation` is the visible mask as an uncompressed run length encoding, the `bbox` is the amodal one clipped to the image, and `ignore` is set for objects below `visib_fract` 0.1 so that heavily occluded instances do not distort the mAP. Objects without a single visible pixel are left out entirely.

The run length encoding is byte identical to `bop_toolkit_lib/pycoco_utils.py`, which is what makes it readable by `pycocotools`.

#### Test targets

`test_targets_bop19.json` and `test_targets_bop24.json` list the object instances per image that the BOP evaluation scripts iterate over. They are written from a split named `test` if there is one, otherwise from the validation split as the next best held out data, and only failing that from the training split. Both files carry the `inst_count` field that `eval_calc_errors.py` asserts on in localization mode.

The two differ in what they include. `bop19` drives the 6D localization evaluation, where the method is told which objects to look for, and lists only objects with `visib_fract >= 0.1`, matching `enumerate_test_targets.py`. `bop24` drives the 6D detection evaluation, where it is not, and lists every ground truth object.

Note that `bop24` is nevertheless not quite complete: Replicator only reports the objects it considers visible at all, so fully occluded objects are missing from `scene_gt.json` and therefore from the targets as well, where BOP would list them with `visib_fract: 0.0`.

#### Object symmetries

BOP records the rotational symmetries of every model in `models_info.json`, and ROCA's `BopDataset` reads them from there to pick symmetry-aware training targets. This plugin does not derive them: `models_info.json` is written without symmetry keys, exactly as in LM-O, and the training falls back to the plain losses. The five CAD models currently shipped in `cad_models/` have no rotational symmetry, so nothing is lost for them. For a symmetric part, add the keys to `models_info.json` by hand after the conversion, following the [BOP format documentation](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).

#### Notes and limitations

- **Models are not re-centered.** BOP models are conventionally centered on their bounding box, but `cam_t_m2c` refers to the CAD origin, so the meshes are only scaled from meters to millimeters. `models_info.json` describes the true extents in that frame. No rotation is applied either: the `convert_stage_up_z` flag of the OBJ to USD conversion only sets the stage up-axis metadata and does not rebase the geometry.
- **Depth is not masked** to the object pixels, unlike in the ROCA converter. BOP expects the full scene depth, and `px_count_valid` as well as the visibility test of `bop_toolkit` depend on it.
- **Fully occluded objects are missing.** Replicator only reports the objects its `bounding_box_2d_tight` annotator considers visible, so objects that are completely hidden do not appear at all, where BOP would list them with `visib_fract: 0.0`. This does not affect training, since those annotations are filtered out anyway.
- **`visib_fract` accounts for truncation at the image border**, which Replicator's own `occlusion_ratio` does not. For occluded objects the two agree closely; for objects running out of the frame this plugin reports the lower, correct value.
- The amodal silhouette is rasterized from the CAD model while the visible mask comes from Replicator's renderer. Because the two rasterizers differ, `px_count_all` is taken from the supersampled subpixel area rather than from a pixel count: a plain binary fill counts every touched boundary pixel in full, which inflates `px_count_all` by roughly a perimeter and pushes `visib_fract` of a fully visible object down to about 0.92. The visible mask is additionally folded into the amodal one, so that `mask_visib <= mask` and `bbox_visib <= bbox_obj` hold exactly.

#### Using bop_toolkit

The plugin writes a complete dataset, so `bop_toolkit` is optional. To run its scripts on the result, the dataset has to be registered, because `bop_toolkit_lib/dataset_params.py` is a hardcoded chain that ends in `raise ValueError("Unknown BOP dataset")`. Place the converted dataset under `bop_toolkit`'s `datasets_path` and add a branch to `get_split_params`:

```python
elif dataset_name == "sodah":
    p["scene_ids"] = [0]
    p["im_size"] = (480, 360)
```

Note that the same function unconditionally overrides `p["scene_ids"]` with `list(range(50))` for a `pbr` split type. For a dataset with a single scene either convert with a split name that carries no split type, for example `split=train`, or add the dataset to that exception as well.

Afterwards `calc_gt_masks.py` and `calc_gt_info.py` can regenerate the masks and `scene_gt_info.json` with the toolkit's own renderer, and `eval_bop19_pose.py` can score pose predictions.
