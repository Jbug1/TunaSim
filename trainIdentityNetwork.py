#main script to run network training
from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from datasetBuilder import trainSetBuilder
from networks import IdentityMatchNetwork
import layers
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
                                        outputs_directory = config.results_directory,
                                        ppm_match_window = config.ppm_match_window,
                                        ms2_da = config.ms2_da,
                                        ms2_ppm = config.ms2_ppm
                                        )
        
        trainSetBuilder_.make_directory_structure()
        
        trainSetBuilder_.break_datasets()

        logger.info('finished dataset creation')

    #create layer objects
    tunasim_layer = layers.tunaSimLayer(trainers = config.tunaSim_trainers,
                                     residual_downsample_percentile = config.residual_downsample_percentile,
                                     inference_jobs = config.inference_jobs,
                                     inference_chunk_size = config.inference_chunk_size)
    
    ensemble_layer = layers.ensembleLayer(candidates = config.ensemble_candidates,
                                          selection_method = config.selection_method)
    
    query_adjustment_layer = layers.groupAdjustmentLayer(candidates = config.query_adjustment_candidates,
                                                         selection_method = config.selection_method,
                                                         groupby_column = 'queryID')

    #create network
    network = IdentityMatchNetwork(train_path = f'{config.results_directory}/matched/train.pkl',
                                   val_1_path = f'{config.results_directory}/matched/val_1.pkl',
                                   val_2_path = f'{config.results_directory}/matched/val_2.pkl',
                                   intermediate_outputs_path = f'{config.results_directory}/intermediate_outputs',
                                   tunaSim_layer = tunasim_layer,
                                   ensemble_layer = ensemble_layer,
                                   query_adjustment_layer = query_adjustment_layer)
    
    try:
        network.fit()

    except Exception as e:
        logger.error(f'error during network fitting: {e}')
        raise

    finally:

        with open(f'{config.results_directory}/network.pkl', 'wb') as handle:

            dump(network, handle)

if __name__ == '__main__':

    main(argv[1])