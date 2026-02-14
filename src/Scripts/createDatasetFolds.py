#performs any part of 
    # 1. retrieve structures from pubchem and/or CTS
    # 2. determine folds based on config specified MCES threshold and formula identity

import requests
from joblib import Parallel, delayed
from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from os import makedirs
import shutil
from sys import argv

def main(config_path):

    module_spec = spec_from_file_location('config', config_path)
    config = module_from_spec(module_spec)
    module_spec.loader.exec_module(config)

    makedirs(config.log_path, exist_ok = True)
    makedirs(config.results_directory, exist_ok = True)
    basicConfig(filename = f'{config.log_path}/create_folds_log.log', 
                level = "INFO", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = getLogger(__name__)

    #copy config file to log folder
    shutil.copy(config_path, f'{config.log_path}/create_folds_config.py')

    logger.info(f'instantiated log and copied config')


    #instantiate

    
