#main script to run network training
from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from datasetBuilder import trainSetBuilder
from networks import IdentityMatchNetwork
from sys import argv
import shutil
from os import makedirs
from pickle import dump

def main(config_path):

    module_spec = spec_from_file_location('config', config_path)
    config = module_from_spec(module_spec)
    module_spec.loader.exec_module(config)

    makedirs(config.log_path, exist_ok = True)
    basicConfig(filename = f'{config.log_path}/training_log.log', 
                level = "INFO", format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = getLogger(__name__)

    #copy config file to log folder
    shutil.copy(config_path, f'{config.log_path}/config.py')

    logger.info(f'instantiated log and copied config')

    if config.build_datasets:

        trainSetBuilder_ = trainSetBuilder(query_input_path = config.query_input_path,
                                        target_input_path = config.target_input_path,
                                        dataset_max_sizes = config.dataset_max_sizes,
                                        dataset_names = config.dataset_names,
                                        identity_column = config.identity_column,
                                        outputs_directory = config.match_directory,
                                        ppm_match_window = config.ppm_match_window,
                                        ms2_da = config.ms2_da,
                                        ms2_ppm = config.ms2_ppm
                                        )
        
        trainSetBuilder_.make_directory_structure()
        
        trainSetBuilder_.break_datasets()

        logger.info('finished dataset creation')

    #fit network
    network = IdentityMatchNetwork(train_path = f'{config.match_directory}/matched/train.pkl',
                                   val_1_path = f'{config.match_directory}/matched/val_1.pkl',
                                   val_2_path = f'{config.match_directory}/matched/val_2.pkl',
                                   test_path = f'{config.match_directory}/matched/test.pkl',
                                   tunaSim_trainers = config.tunaSim_trainers,
                                   intermediate_outputs_path = config.intermediate_outputs_path,
                                   tunaSim_aggregation_candidates = config.tunaSim_aggregation_candidates,
                                   query_adjustment_candidates = config.query_adjustment_candidates,
                                   residual_downsampling_percentile = config.residual_downsampling_percentile,
                                   model_selection_method = config.model_selection_method)
    
    try:
        network.fit()

    except Exception as e:
        logger.error(f'error during network fitting: {e}')
        raise

    finally:

        with open(f'{config.intermediate_outputs_path}/pickled_objects/network.pkl', 'wb') as handle:

            dump(network, handle)

if __name__ == '__main__':

    main(argv[1])