import os

import dagster as dg

from dagster_pipeline.assets import (
    pipeline_structure,
    cleaned_dataset,
    retrieved_dataset,
    mces_database,
    fold_assignments,
    training_datasets,
)
from dagster_pipeline.resources import PipelineConfig

CONFIGS = os.path.join(os.path.dirname(__file__), "configs")

defs = dg.Definitions(
    assets=[pipeline_structure, cleaned_dataset, retrieved_dataset, mces_database, fold_assignments, training_datasets],
    resources={
        "pipeline_config": PipelineConfig(
            cleaning_config_path=f"{CONFIGS}/cleanDatasetConfig.py",
            fold_inputs_config_path=f"{CONFIGS}/foldInputsConfig.py",
            fold_datasets_config_path=f"{CONFIGS}/foldDatasetsConfig.py",
        ),
    },
)
