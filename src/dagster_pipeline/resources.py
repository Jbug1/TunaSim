from importlib.util import spec_from_file_location, module_from_spec

from dagster import ConfigurableResource


def load_config(config_path: str):
    """Load a Python config file as a module (same pattern as existing Scripts)."""
    module_spec = spec_from_file_location("config", config_path)
    config = module_from_spec(module_spec)
    module_spec.loader.exec_module(config)
    return config


# subdirectory names for each asset's outputs
ASSET_DIRS = [
    "cleaned_dataset",
    "retrieved_dataset",
    "mces_database",
    "fold_assignments",
    "training_datasets",
]


class PipelineConfig(ConfigurableResource):
    """Pipeline configuration: base output directory, raw input, and per-asset config paths."""

    base_output_dir: str
    raw_input_path: str  # path to the combined raw pickle (external input)

    cleaning_config_path: str
    fold_inputs_config_path: str
    fold_datasets_config_path: str
