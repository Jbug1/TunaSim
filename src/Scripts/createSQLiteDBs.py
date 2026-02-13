#main script to create sqlite DBs of molecular composition and m/z

from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from os import makedirs
import shutil
from sys import argv
from mzDB import mzDB


def main(config_path):

    module_spec = spec_from_file_location('config', config_path)
    config = module_from_spec(module_spec)
    module_spec.loader.exec_module(config)

    makedirs(config.log_path, exist_ok = True)
    makedirs(config.results_directory, exist_ok = True)
    basicConfig(filename = f'{config.log_path}/db_create_log.log', 
                level = "INFO", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = getLogger(__name__)

    #copy config file to log folder
    shutil.copy(config_path, f'{config.log_path}/config_clean.py')

    logger.info(f'instantiated log and copied config')

    #instantiate db creation object
    mzDB_ = mzDB(mz_map = config.mz_map,
                 db_path = config.db_path)
    
    mzDB_.create_mzDB()


if __name__ == '__main__':

    main(argv[1])
