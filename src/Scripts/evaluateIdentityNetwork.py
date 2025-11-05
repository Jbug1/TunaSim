from logging import getLogger, basicConfig
from importlib.util import spec_from_file_location, module_from_spec
from TunaSimNetwork.datasetBuilder import trainSetBuilder
from TunaSimNetwork.oldMetrics import oldMetricEvaluator
from sys import argv
import shutil
from os import makedirs
from pickle import load
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

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
    shutil.copy(config_path, f'{config.log_path}/config_eval.py')

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

    network.intermediate_outputs_path = config.results_directory

    match_data = pd.read_pickle(f'{config.matches_input_directory}/matched/{config.dataset_names[0]}.pkl')

    preds = network.predict(match_data, 
                            write_intermediates = config.write_intermediates)

    preds.to_csv(f'{config.results_directory}/network_predicitons.csv', index = False)

    logger.info('generated network preds')

    int_results = list()
    int_names = list()

    int_results.append(round(roc_auc_score(preds['score'], preds['preds']),5))
    int_names.append('full_network')

    if config.network_performance_attribution:

        tunasims = pd.read_csv(f'{config.results_directory}/tunaSim_output.csv')

        for col in tunasims.columns:

            if 'tuna' in col:

                #tunasim layer results are already grouped by max
                try:
                    int_results.append(round(roc_auc_score(tunasims['score'], tunasims[col]),5))

                except Exception as err:
                    logger.info(err)
                    int_results.append(-1)

                int_names.append(col)

                logger.info(f'evaluated: {col}')

        ensembled_tunasims = pd.read_csv(f'{config.results_directory}/ensemble_output.csv')

        int_results.append(round(roc_auc_score(ensembled_tunasims['score'], ensembled_tunasims['preds']),5))
        int_names.append('ensembled')

    network_performance = pd.DataFrame(list(zip(int_names, int_results)), columns = ['name', 'performance'])

    if config.evaluate_old_metrics:

        evaluator = oldMetricEvaluator(groupby_columns = network.tunaSim_layer.trainers[0].groupby_column,
                                       intermediates_path = config.results_directory,
                                       performance_path = config.results_directory)

        performance = evaluator.get_evals(match_data)

        logger.info('evaluated original metrics')

        performance = pd.concat((performance, evaluator.get_evals(match_data, reweighted = True)))

        logger.info('evaluated reweighted original metrics')
        
        network_performance = pd.concat((performance, network_performance))
        
    if config.additional_feature_eval:

        #read tuna outputs and reduce to tunasims and label
        ensemble_input = pd.read_csv(f'{config.train_directory}/tunasims_top_train.csv')
        ensemble_input = ensemble_input[[i for i in ensemble_input.columns if 'tuna' in i or 'score' in i]]

        test_data = pd.read_csv(f'{config.results_directory}/tunaSim_output.csv')
        test_data = test_data[[i for i in test_data.columns if 'tuna' in i or 'score' in i]]

        all_performances = list()
        for i in range(1, len(ensemble_input.columns) - 1):

            sub = ensemble_input.iloc[:, :i]
            sub_test = test_data.iloc[:, :i]

            sub_performances = list()
            for model in config.ensemble_candidates:

                model.fit(sub, ensemble_input.iloc[:,-1])
                sub_performances.append((f'{i}_{str(model)}', round(roc_auc_score(test_data.iloc[:,-1], model.predict_proba(sub_test)[:,1]),5)))

            all_performances = all_performances + sub_performances
            logger.info(f'completed all ensemble candidates for {i}')

        all_performances = pd.DataFrame(all_performances, columns = ['name', 'performance'])
        network_performance = pd.concat((network_performance, all_performances))

    network_performance.to_csv(f'{config.results_directory}/performance.csv', index = False)


if __name__ == '__main__':

    main(argv[1])


        







    



