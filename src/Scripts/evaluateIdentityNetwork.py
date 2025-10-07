from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from TunaSimNetwork.datasetBuilder import trainSetBuilder
from TunaSimNetwork.oldMetrics import oldMetricEvaluator
from sys import argv
import shutil
from os import makedirs
from pickle import load
from sklearn.metrics import roc_auc_score
from pandas import read_pickle, concat

def main(config_path):

    module_spec = spec_from_file_location('config', config_path)
    config = module_from_spec(module_spec)
    module_spec.loader.exec_module(config)

    makedirs(config.log_path, exist_ok = True)
    makedirs(config.results_directory, exist_ok = True)
    basicConfig(filename = f'{config.log_path}/evaluation_log.log', 
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
                                        outputs_directory = config.matches_input_directory,
                                        ppm_match_window = config.ppm_match_window,
                                        tolerance = config.tolerance,
                                        units_ppm = config.units_ppm
                                        )
        
        trainSetBuilder_.make_directory_structure()
        
        trainSetBuilder_.break_datasets()

        logger.info('finished dataset creation')

    with open(config.network_path, 'rb') as handle:

        network = load(handle)

    logger.info('loaded network')

    network.intermdeiate_outputs_directory = config.results_directory

    match_data = read_pickle(f'{config.matches_input_directory}/{config.dataset_names[0]}.pkl')

    preds = network.predict(match_data, 
                            write_intermediates = config.write_intermediates)

    preds.to_csv(config.output_directory, index = False)

    logger.info('generated network preds')

    network_performance = roc_auc_score(preds['score'], preds['preds'])

    if config.evaluate_old_metrics:

        evaluator = oldMetricEvaluator()

        performance = evaluator.get_evals(match_data)

        logger.info('evaluated original metrics')

        performance = concat((performance, evaluator.get_evals(match_data, reweighted = True)))

        logger.info('evaluated reweighted original metrics')
        
        performance.loc[-1] = ['network', round(network_performance,5)]

        performance.to_csv(f'{config.matches_input_directory}/performance.csv', index = False)

    else:

        with open(f'{config.results_directory}/performance.csv', 'w') as handle:

            handle.write(f'network, {round(network_performance,5)}')


if __name__ == '__main__':

    main(argv[1])


        







    



