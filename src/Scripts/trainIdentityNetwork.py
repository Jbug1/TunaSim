#main script to run network training
from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from TunaSimNetwork.networks import IdentityMatchNetwork
from TunaSimNetwork import layers
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
    shutil.copy(config_path, f'{config.log_path}/config_train.py')

    logger.info(f'instantiated log and copied config')

    #create layer objects
    tunasim_layer = layers.tunaSimLayer(trainers = config.tunaSim_trainers,
                                     residual_downsample_percentile = config.residual_downsample_percentile,
                                     inference_jobs = config.inference_jobs,
                                     inference_chunk_size = config.inference_chunk_size)
    
    ensemble_layer = layers.ensembleLayer(candidates = config.ensemble_candidates,
                                          selection_method = config.selection_method,
                                          data_column_str = 'tuna')

    ensemble_layer = layers.ensemble_shell()
    
    query_adjustment_layer = layers.groupAdjustmentLayer(candidates = config.query_adjustment_candidates,
                                                         selection_method = config.selection_method,
                                                         groupby_column = ['queryID'],
                                                         data_column_str = 'top_from_next')
    
    query_adjustment_layer = None

    #create network
    network = IdentityMatchNetwork(train_path = config.train_path,
                                   val_1_path = config.val1_path,
                                   val_2_path = config.val2_path,
                                   intermediate_outputs_path = f'{config.results_directory}/intermediate_outputs',
                                   tunaSim_layer = tunasim_layer,
                                   ensemble_layer = ensemble_layer,
                                   query_adjustment_layer = query_adjustment_layer,
                                   train_match_proportion = config.train_match_proportion,
                                   val_match_proportion = config.val_match_proportion)
    
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