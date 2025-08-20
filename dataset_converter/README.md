# dataset_converter

## dataset_converter.py

The script `dataset_converter.py` converts datasets from the NVIDIA Replicator format to another format. It loads converter plugins from the `converter_plugins` package and applies them to the provided data.

### Arguments

- `--obj_dir`: **required**. Directory containing the source files in OBJ format. No default.
- `--replicator_data_dir`: **required**. Directory containing the source files in the Replicator data format. No default. Can be specified multiple times.
- `--output_dir`: **required**. Destination directory for the converted files. Each plugin receives its own subdirectory. No default.
- `--converters`: **required**. List of converters to convert the Replicator dataset. No default.

### Example command

```bash
python dataset_converter.py \
  --obj_dir "/path/to/obj_directory" \
  --replicator_data_dir "/path/to/replicator_directory1" \
  --replicator_data_dir "/path/to/replicator_directory2" \
  --output_dir "/path/to/output_directory" \
  --converters converter1 converter2
```

### Important notes

- The script requires that the `obj_dir` and `replicator_data_dir` directories exist and contain the appropriate files.
- The `output_dir` is used to store the converted files.
- Custom conversion routines can be added by creating your own plugin. Any plugin that inherits from the `ConverterInterface` class and resides in the `converter_plugins` package can be used as an argument for the `--converters` flag.

## Existing Converter Plugins

### ReplicatorToROCA

This plugin converts the output of NVIDIA Replicator into the format required by [ROCA](https://github.com/drescherflo/ROCA) for training.

