from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from TunaSimNetwork.datasetBuilder import specCleaner
from sys import argv
import shutil
from os import makedirs, path
from pickle import load

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
    shutil.copy(config_path, f'{config.log_path}/config_clean.py')

    logger.info(f'instantiated log and copied config')

    #load raw data
    raw = load(config.raw_path)

    #instantiate cleaner
    cleaner = specCleaner(noise_threshold = config.noise_threshold,
                          precursor_removal = config.precursor_removal_window_mz,
                          deisotoping_gaps = config.deisotoping_gaps,
                          isotope_mz_tolerance = config.isotope_mz_tolerance)
    
    logger.info('instantiated cleaner')
    
    #clean specs according to config specifications
    cleaned = cleaner.clean_spectra(raw)

    logger.info('cleaned specs')

    cleaned.to_pickle(f'{config.results_directory}/{config.cleaned_file_name}')

    logger.info('successfully wrote cleaned specs')

if __name__ == '__main__':

    main(argv[1])



