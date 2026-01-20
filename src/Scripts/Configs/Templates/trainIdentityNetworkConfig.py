#config ffor trainIdentityNetwork.py
from TunaSimNetwork.funcTrainer import tunaSimTrainer, baseShell
from sklearn.ensemble import HistGradientBoostingClassifier as gbc
from TunaSimNetwork.oldMetrics import oldMetricEvaluator
from functools import partial
import numpy as np

#logging
log_path = '/Users/jonahpoczobutt/projects/TunaRes/network_logs_msg_4'
results_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results_msg_4'

#datasetBuilder params
build_datasets = False
dataset_names = ['train', 'val_1', 'val_2']
dataset_max_sizes = [1e7, 1e7, 1e7]
query_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/ms_gym_train_noprec_clean_2.pkl'
target_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/ms_gym_train_noprec_clean_2.pkl'
ppm_match_window = 10
identity_column = 'inchi_base'
tolerance = 0.01
units_ppm = False

#tunasim bounds go here
bounds_mult = {
    'sim_a': (1,1),
    'sim_b': (1e-3, 2),
    'query_intensity_a': (1e-3,2),
    'query_intensity_b': (1e-3,2),
    'target_intensity_a': (1e-3,2),
    'target_intensity_b': (1e-3,2)
    }

bounds_dif = {
    'dif_a':(-1,-1),
    'dif_b': (1e-3, 2),
    'query_intensity_a': (1e-3,2),
    'query_intensity_b': (1e-3,2),
    'target_intensity_a': (1e-3,2),
    'target_intensity_b': (1e-3,2)
    }

init_vals = {
    'mult_a' : 1,
    'mult_b': 1,
    'dif_a': 1,
    'dif_b': 1,
    'target_intensity_a': 0.01,
    'query_intensity_a': 0.01,
    'target_intensity_b': 1,
    'query_intensity_b': 1
    }

n_tunasims_additional = 7
tunasims_n_iter = 5e5
residual_downsample_percentile = 50
tunaSim_balance_column = 'score'
tunaSim_groupby_column = ['queryID', 'inchi_base']
learning_rate = 0.0005
intermediate_outputs_path = f'{results_directory}/intermediate_outputs'
inference_jobs = 4
inference_chunk_size = 1e6
n_inits_per_bound = 5

##############################################################
#network params 
train_match_proportion = 0.5
val_match_proportion = 0.5

bounds_collection = {'bounds_mult' : bounds_mult,
                     'bounds_dif' : bounds_dif}

tunaSim_trainers = list()

# tunaSim_trainers.append(baseShell('tuna_fidelity_base',
#                                   oldMetricEvaluator.fidelity_similarity,
#                                   balance_column = tunaSim_balance_column,
#                                     groupby_column = tunaSim_groupby_column))

# for i in range(n_tunasims_additional):
    
#     tunaSim_trainers.append(tunaSimTrainer(f'tuna_{i+1}',
#                                 init_vals = init_vals,
#                                 n_inits_per_bound = n_inits_per_bound,
#                                 bounds_collection = bounds_collection,
#                                 max_iter = tunasims_n_iter,
#                                 learning_rate = learning_rate,
#                                 balance_column = tunaSim_balance_column,
#                                 groupby_column = tunaSim_groupby_column))

import pickle

with open ('/Users/jonahpoczobutt/projects/TunaRes/network_results_msg_2/network.pkl', 'rb') as handle:

    network_old = pickle.load(handle)

tunaSim_trainers = network_old.tunaSim_layer.trainers

########################################################################
#similarity aggreagtion layer params
selection_method = 'top'
learning_rates = [0.05, 0.1, 0.25, 0.5]
max_leaf_nodes = [20, 31, 40, 80]
max_iter = [100, 800]
l2_regs = [10, 20, 40, 80, 160, 320]
max_bins = [75, 150, 255]

#we will start with default model here to evaluate pickup from hyperparam tuning
ensemble_candidates = [gbc()]

for i in learning_rates:
    for j in max_iter:
        for k in max_leaf_nodes:
            for l in l2_regs:

                ensemble_candidates.append(gbc(learning_rate = i,
                                                max_iter = j,
                                                max_leaf_nodes = k,
                                                l2_regularization = l))

########################################################################
#query adjustment layer params
learning_rates = [0.05, 0.1, 0.25, 0.5]
max_leaf_nodes = [20, 31, 40, 80]
max_iter = [200, 800]
l2_regs = [10, 20, 40, 80, 160]

#we will start with default model here to evaluate pickup from hyperparam tuning
query_adjustment_candidates = [gbc()]

for i in learning_rates:
    for j in max_iter:
        for k in max_leaf_nodes:
            for l in l2_regs:

                query_adjustment_candidates.append(gbc(learning_rate = i,
                                                            max_iter = j,
                                                            max_leaf_nodes = k,
                                                            l2_regularization = l))