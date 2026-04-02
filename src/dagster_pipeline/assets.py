import logging
import os
import pickle
import shutil
from os import makedirs

import dagster as dg
import pandas as pd

from dagster_pipeline.resources import ASSET_DIRS, PipelineConfig, load_config


def _asset_dir(pipeline_config: PipelineConfig, asset_name: str) -> str:
    """Return the subdirectory for a given asset."""
    return os.path.join(pipeline_config.base_output_dir, asset_name)


def _setup_file_logging(pipeline_config: PipelineConfig, asset_name: str) -> logging.FileHandler:
    """Add a file handler to the root logger and the dagster logger so that both
    library code (via root) and asset code (via context.log / dagster) write to
    the asset's log file.  Returns the handler so it can be removed later."""

    out_dir = _asset_dir(pipeline_config, asset_name)
    log_path = os.path.join(out_dir, f"{asset_name}.log")
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)

    dagster_logger = logging.getLogger("dagster")
    dagster_logger.addHandler(handler)

    return handler

def _teardown_file_logging(handler: logging.FileHandler):
    """Remove and close the file handler added by _setup_file_logging."""
    logging.getLogger().removeHandler(handler)
    logging.getLogger("dagster").removeHandler(handler)
    handler.close()

def _archive_config(context, pipeline_config: PipelineConfig, config_path: str, asset_name: str):
    """Copy the config file to the asset's output directory."""
    out_dir = _asset_dir(pipeline_config, asset_name)
    shutil.copy(config_path, out_dir)
    context.log.info(f"Config archived to {out_dir}")

@dg.asset(
    description="Create output directory structure with subdirectories for each pipeline asset.",
)
def pipeline_structure(context: dg.AssetExecutionContext, pipeline_config: PipelineConfig):
    for name in ASSET_DIRS:
        d = os.path.join(pipeline_config.base_output_dir, name)
        makedirs(d, exist_ok=True)
        context.log.info(f"Created {d}")

@dg.asset(
    deps=[pipeline_structure],
    description="Clean raw spectra (deisotoping, noise removal, precursor removal).",
)
def cleaned_dataset(context: dg.AssetExecutionContext, pipeline_config: PipelineConfig):
    from TunaSimNetwork.datasetBuilder import specCleaner

    log_handler = _setup_file_logging(pipeline_config, "cleaned_dataset")
    try:
        cfg = load_config(pipeline_config.cleaning_config_path)
        _archive_config(context, pipeline_config, pipeline_config.cleaning_config_path, "cleaned_dataset")

        raw = pd.read_pickle(pipeline_config.raw_input_path)
        context.log.info(f"Loaded {len(raw)} spectra from {pipeline_config.raw_input_path}")

        cleaner = specCleaner(
            noise_threshold=cfg.noise_threshold,
            precursor_removal_window_mz=cfg.precursor_removal_window_mz,
            deisotoping_gaps=cfg.deisotoping_gaps,
            isotope_mz_tolerance=cfg.isotope_mz_tolerance,
        )

        raw["spectrum"] = cleaner.clean_spectra(raw["spectrum"], raw["precursor"])

        #clear out spectra of zero length after cleaning
        raw = raw.loc[[True if len(i) > 0 else False for i in raw['spectrum']]]

        output_path = os.path.join(_asset_dir(pipeline_config, "cleaned_dataset"), "cleaned.pkl")
        raw.to_pickle(output_path)
        context.log.info(f"Saved cleaned dataset to {output_path}")

    finally:
        _teardown_file_logging(log_handler)

@dg.asset(
    deps=[cleaned_dataset],
    description="Retrieve molecular annotations from PubChem/CTS for all unique InChI key bases.",
)
def retrieved_dataset(context: dg.AssetExecutionContext, pipeline_config: PipelineConfig):
    from TunaSimNetwork.annotationTools import molRetriever

    log_handler = _setup_file_logging(pipeline_config, "retrieved_dataset")
    try:
        cleaned_path = os.path.join(_asset_dir(pipeline_config, "cleaned_dataset"), "cleaned.pkl")
        df = pd.read_pickle(cleaned_path)
        all_keys = list(df["inchikey"].unique())
        context.log.info(f"Retrieving annotations for {len(all_keys)} unique InChIKeys")

        retriever = molRetriever()
        retrieved_df, errored_keys = retriever.get_annotations_for_inchikeys(all_keys)

        if errored_keys:
            context.log.info(f"{len(errored_keys)} keys failed retrieval")

        context.log.info(f"{len(retrieved_df)} keys retrieved")

        output_path = os.path.join(_asset_dir(pipeline_config, "retrieved_dataset"), "retrieved.csv")
        retrieved_df.to_csv(output_path, index=False)
        context.log.info(f"Saved retrieved dataset to {output_path}")

    finally:
        _teardown_file_logging(log_handler)

@dg.asset(
    deps=[retrieved_dataset],
    description="Build pairwise MCES similarity SQLite database from combined dataset.",
)
def mces_database(context: dg.AssetExecutionContext, pipeline_config: PipelineConfig):
    from TunaSimNetwork.annotationTools import simDB
    from TunaSimNetwork.datasetBuilder import foldCreation

    log_handler = _setup_file_logging(pipeline_config, "mces_database")
    try:
        cfg = load_config(pipeline_config.fold_inputs_config_path)
        _archive_config(context, pipeline_config, pipeline_config.fold_inputs_config_path, "mces_database")

        retrieved_path = os.path.join(_asset_dir(pipeline_config, "retrieved_dataset"), "retrieved.csv")
        combined = pd.read_csv(retrieved_path)
        context.log.info(f"Loaded {len(combined)} compounds from combined dataset")

        db_path = os.path.join(_asset_dir(pipeline_config, "mces_database"), "mces.sqlite")
        sim_db = simDB(db_path)
        folder = foldCreation(sim_db=sim_db, n_jobs=cfg.n_jobs)

        inchikey_bases = list(combined["inchikey_base"])
        inchis = list(combined["inchi"])

        context.log.info(
            f"Starting pairwise MCES computation ({len(inchikey_bases)} compounds, {cfg.n_jobs} workers)"
        )
        folder.batch_sim_generation(
            inchikey_bases=inchikey_bases,
            inchis=inchis,
            mzs=inchikey_bases,  # calc_sim_bounds uses this as inchikey_bases for identity check
        )

        # reopen since batch_sim_generation closes the connection
        sim_db = simDB(db_path)
        sim_db.index_tables()
        sim_db.close()

        context.log.info(f"MCES database saved and indexed at {db_path}")

    finally:
        _teardown_file_logging(log_handler)

@dg.asset(
    deps=[mces_database],
    description="Assign compound indices to train/val/test folds using MCES similarity cascading.",
)
def fold_assignments(context: dg.AssetExecutionContext, pipeline_config: PipelineConfig):
    from types import SimpleNamespace

    from TunaSimNetwork.annotationTools import simDB
    from TunaSimNetwork.datasetBuilder import foldCreation

    log_handler = _setup_file_logging(pipeline_config, "fold_assignments")
    try:

        cfg = load_config(pipeline_config.fold_inputs_config_path)
        _archive_config(context, pipeline_config, pipeline_config.fold_inputs_config_path, "fold_assignments")

        db_path = os.path.join(_asset_dir(pipeline_config, "mces_database"), "mces.sqlite")
        sim_db = simDB(db_path)
        folder = foldCreation(sim_db=sim_db, n_jobs=cfg.n_jobs)

        # build fold mappings from config — each source can be pinned to a fold
        cleaned_path = os.path.join(_asset_dir(pipeline_config, "cleaned_dataset"), "cleaned.pkl")
        df = pd.read_pickle(cleaned_path)
        dataset_fold_mappings = []
        for source_name, fold in cfg.fold_mappings.items():
            source_df = df[df["source"] == source_name]
            wrapper = SimpleNamespace(name=source_name, data=source_df)
            dataset_fold_mappings.append((wrapper, fold))

        context.log.info(
            f"Assigning folds with mces_cutoff={cfg.mces_cutoff}, folds={cfg.fold_names}"
        )
        inds_by_fold = folder.assign_inds_to_folds(
            mces_cutoff=cfg.mces_cutoff,
            dataset_fold_mappings=dataset_fold_mappings,
            fold_names=cfg.fold_names,
        )

        identities_by_fold = folder.sim_db.convert_inds_dict_to_identities(inds_by_fold)

        for fold_name, inds in inds_by_fold.items():
            context.log.info(f"  {fold_name}: {len(inds)} identities")

        output_path = os.path.join(_asset_dir(pipeline_config, "fold_assignments"), "fold_assignments_ind.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(inds_by_fold, f)

        output_path = os.path.join(_asset_dir(pipeline_config, "fold_assignments"), "fold_assignments.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(identities_by_fold, f)

        context.log.info(f"Fold assignments saved to {output_path}")

    finally:
        _teardown_file_logging(log_handler)

@dg.asset(
    deps=[fold_assignments],
    description="Build training datasets using trainSetBuilder (query/target spectral matching).",
)
def training_datasets(context: dg.AssetExecutionContext, pipeline_config: PipelineConfig):
    from TunaSimNetwork.datasetBuilder import trainSetBuilder

    log_handler = _setup_file_logging(pipeline_config, "training_datasets")
    try:
        cfg = load_config(pipeline_config.fold_datasets_config_path)
        output_dir = _asset_dir(pipeline_config, "training_datasets")
        _archive_config(context, pipeline_config, pipeline_config.fold_datasets_config_path, "training_datasets")

        # query and target both come from cleaned df
        query_input_path = os.path.join(_asset_dir(pipeline_config, "cleaned_dataset"), "cleaned.pkl")
        target_input_path = os.path.join(_asset_dir(pipeline_config, "cleaned_dataset"), "cleaned.pkl")

        builder = trainSetBuilder(
            query_input_path=query_input_path,
            target_input_path=target_input_path,
            dataset_max_sizes=cfg.dataset_max_sizes,
            fold_identity_mapping = pd.read_pickle(os.path.join(_asset_dir(pipeline_config, "fold_assignments"), "fold_assignments.pkl")),
            identity_column=cfg.identity_column,
            outputs_directory=output_dir,
            ppm_match_window=cfg.ppm_match_window,
            tolerance=cfg.tolerance,
            units_ppm=cfg.units_ppm,
        )

        builder.make_directory_structure()
        builder.break_datasets()

        context.log.info(f"Training datasets created in {output_dir}")

    finally:
        _teardown_file_logging(log_handler)