#main script to run network training
import configparser
import pandas as pd
import logging
from datasetBuilder import trainSetBuilder
from networks import IdentityMatchNetwork
from sys import argv
import shutil


def main(config_path):

    config = configparser.ConfigParser()
    config.read(config_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(f'{config.log_path}/train_log.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',  datefmt='%Y-%m-%d %H:%M:%S')
    
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    #copy config file to log folder
    shutil.copy(config_path, f'{config.log_path}/config.py')

    if config.create_datasets:

        trainSetBuilder_ = trainSetBuilder(query_input_path = config.query_input_path,
                                        target_input_path = config.target_input_path,
                                        dataset_max_sizes = config.dataset_max_sizes,
                                        dataset_names = config.dataset_names,
                                        identity_column = config.identity_column,
                                        outputs_directory = config.raw_match_directory
                                        )
        
        trainSetBuilder_.break_datasets()

    logger.info('finished dataset creation')

    #fit network
    network = IdentityMatchNetwork(train_path = f'{config.raw_match_directory}/train.pkl',
                                   val_1_path = f'{config.raw_match_directory}/val_1pkl',
                                   val_2_path = f'{config.raw_match_directory}/val_2.pkl',
                                   test_path = f'{config.raw_match_directory}/test.pkl',
                                   intermediate_outputs_path = config.intermediate_outputs_path,
                                   score_column = config.score_column,
                                   tunasim_groupby_column = config.tunaSim_groupby_column,
                                   reweight_groupby_column = config.reweight_groupby_column,
                                   tunaSim_consolidation_candidates = config.tunaSim_consolidation_candidates,
                                   score_by_group_funcObs = config.scoreByQuery_funcObs,
                                   output_layer_candidates = config.output_layer_candidates,
                                   residual_downsample_percentile = config.residual_downsample_percentile)
    
    network.fit()


if __name__ == '__main__':

    main(argv[2])
    

    



    

        





