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
            base_output_dir="/Users/jonahpoczobutt/projects/TunaRes/pipeline_output",
            raw_input_path="/Users/jonahpoczobutt/projects/raw_data/highres_pickles/nist23_metlin_combined.pkl",
            cleaning_config_path=f"{CONFIGS}/cleanDatasetConfig.py",
            fold_inputs_config_path=f"{CONFIGS}/foldInputsConfig.py",
            fold_datasets_config_path=f"{CONFIGS}/foldDatasetsConfig.py",
        ),
    },
)
