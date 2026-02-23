from TunaSimNetwork.datasetBuilder import foldCreation, SimDB
from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from sys import argv
import shutil
from os import makedirs, path
import pandas as pd
import sqlite3
import time
from rdkit.Chem import MolFromInchi

def main(config_path):

    module_spec = spec_from_file_location('config', config_path)
    config = module_from_spec(module_spec)
    module_spec.loader.exec_module(config)

    makedirs(config.log_path, exist_ok = True)
    makedirs(config.results_directory, exist_ok = True)
    basicConfig(filename = f'{config.log_path}/cleaning_log.log', 
                level = "INFO", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = getLogger(__name__)

    #copy config file to log folder
    shutil.copy(config_path, f'{config.log_path}/config_fold_benchmark.py')

    logger.info(f'instantiated log and copied config')

    combined_dataset = pd.read_csv(config.combined_path)

    inchi_bases = [i.split('-')[0] for i in combined_dataset['inchikey']]
    mols = [MolFromInchi(i) for i in combined_dataset['inchi']]

    ms_data = pd.read_pickle(config.ms_data_path)

    for i, (rascal_option, timeout, mz_digits, n_jobs, output_dir) in enumerate(config.param_sets):

        sim_db = SimDB(f'{output_dir}/fold_input_data.db')

        folder = foldCreation(rascal_options=rascal_option,
                              mz_digits = mz_digits,
                              mces_timeout = timeout,
                              n_jobs = n_jobs,
                              output_directory = output_dir,
                              sim_db = sim_db)
        
        start = time.time()

        mzs = folder.generate_mz_sets_by_inchi_base(base_retrieval_data = combined_dataset,
                                                    ms_data = ms_data)

        folder.parallel_sim_generation(inchi_bases = inchi_bases,
                                       mols = mols,
                                       mzs = mzs,
                                       max_groups = config.max_groups)
        
        logger.info(f'round {i} completed in {round((time.time() - start) / 60, 2)} minutes')

        with sqlite3.connect(f'{output_dir}/fold_input_data.db') as conn:
            inputs = pd.read_sql('SELECT * FROM similarities', conn)

        mz_matches = len(inputs[inputs['code'] == 3])
        max_bond_pairs = len(inputs[inputs['code'] == 2])

        


