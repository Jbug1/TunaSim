#config file to pass all parameters onto script
from funcTrainer import specSimTrainer
from sklearn.ensemble import HistGradientBoostingClassifier as gbc


#logging
log_path = '/Users/jonahpoczobutt/projects/TunaRes/network_logs/ten_sim_speed'

#datasetBuilder params
build_datasets = False
dataset_names = ['train', 'val_1', 'val_2', 'test']
#dataset_max_sizes = [5e6, 5e6, 10, 10]
dataset_max_sizes = [1e7, 5e6, 5e6, 10e6]
query_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/nist23_full.pkl'
target_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/nist23_full.pkl'
ppm_match_window = 10
identity_column = 'inchi_base'
match_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results_speed'
ms2_da = 0.05
ms2_ppm = None

#first tunasim parameterization funcs
bounds = {'add_norm_b': (0, 2),
          'mult_add_norm_b': (0, 2),
          'dif_add_norm_b': (0, 2),
          'mult_b': (1e-10, 2),
          'add_norm_a': (1e-10, 3),
          'dif_b': (1e-10, 2),
          'dif_a':(-3,3),
          'mult_a': (-3,3),
          'add_norm_int': (0, 3),
           'target_intensity_int': (-0.2,1),
           'query_intensity_int': (-0.2,1),
          'target_intensity_a': (1e-10,2),
          'query_intensity_a': (1e-10,2),
          'target_intensity_b': (1e-10,2),
          'query_intensity_b': (1e-10,2),
          'target_intensity_c': (1e-10,2),
          'query_intensity_c': (1e-10,2)
          }

init_vals = {
    'mult_a' : 0.001,
    'mult_b': 1,
    'dif_a': 0.001,
    'dif_b':1,
    'add_norm_b' : 1,
    'target_intensity_a': 0.1,
    'query_intensity_a': 0.1,
    'target_intensity_b': 0.1,
    'query_intensity_b': 0.1,
    }

n_tunasims_final = 10
tunasims_n_iter = 3e5
residual_downsampling_percentile = 50
tunaSim_balance_column = 'score'
tunaSim_groupby_column = ['queryID', 'inchi_base']
learning_rate = 0.001
intermediate_outputs_path = f'{match_directory}/intermediate_ouputs'

tunaSim_trainers = list()
for i in range(n_tunasims_final):
    
    tunaSim_trainers.append(specSimTrainer(f'tuna_{i}',
                                init_vals = init_vals,
                                bounds = bounds,
                                max_iter = tunasims_n_iter,
                                learning_rates = learning_rate,
                                balance_column = tunaSim_balance_column,
                                groupby_column = tunaSim_groupby_column))

########################################################################
#tuna aggreagtion layer params
aggregator_selection_method = 'top'
learning_rates = [0.05, 0.1, 0.25, 0.5]
max_leaf_nodes = [20, 31, 40, 80]
max_iter = [200, 800]
l2_regs = [10, 20, 40, 80, 160]

#we will start with default model here to evaluate pickup from hyperparam tuning
tunaSim_aggregation_candidates = [gbc()]

for i in learning_rates:
    for j in max_iter:
        for k in max_leaf_nodes:
            for l in l2_regs:

                tunaSim_aggregation_candidates.append(gbc(learning_rate = i,
                                                            max_iter = j,
                                                            max_leaf_nodes = k,
                                                            l2_regularization = l))

########################################################################
#reweight layer params
init_vals = {
    'mult_a' : 0.001,
    'mult_b': 1,
    'dif_a': 0.001,
    'dif_b':1,
    'add_norm_b' : 1,
    'target_intensity_a': 0.1,
    'query_intensity_a': 0.1,
    'target_intensity_b': 0.1,
    'query_intensity_b': 0.1,
    }

# n_tunasims_final = 5
# tunasims_n_iter = 2e6
# residual_downsample_percentile = 50
# balance_column_tuna = 'score'
# groupby_column_tuna = ['queryID', 'target_base']
# learning_rate = 0.001

scoreByQuery_trainers = list()
scoreByGroup_aggregation_candidates = list()
# func_obs = list()
# for i in range(n_tunasims_final):
    
#     func_obs.append(reweightTrainer(f'tuna_{i}',
#                                 init_vals = init_vals,
#                                 bounds = bounds,
#                                 max_iter = tunasims_n_iter,
#                                 learning_rates = learning_rate,
#                                 balance_column= balance_column_tuna,
#                                 groupby_column = groupby_column_tuna))

    






