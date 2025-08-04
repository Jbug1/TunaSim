#config file to pass all parameters onto script
from funcTrainer import specSimTrainer, reweightTrainer


#logging
log_path = str()

#datasetBuilder params
build_datasets = True
dataset_names = list()
dataset_sizes = list()
query_path = str()
target_path = str()
ppm_window = int()
identity_column = str()
output_directory = str()

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

n_tunasims_final = 5
tunasims_n_iter = 2e6
residual_downsample_percentile = 50
balance_column_tuna = 'score'
groupby_column_tuna = ['queryID', 'target_base']
learning_rate = 0.001

func_obs = list()
for i in range(n_tunasims_final):
    
    func_obs.append(specSimTrainer(f'tuna_{i}',
                                init_vals = init_vals,
                                bounds = bounds,
                                max_iter = tunasims_n_iter,
                                learning_rates = learning_rate,
                                balance_column= balance_column_tuna,
                                groupby_column = groupby_column_tuna))

########################################################################
#tuna consolidation layer params
consolidation_models = list()

selection_method = 'top'

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

n_tunasims_final = 5
tunasims_n_iter = 2e6
residual_downsample_percentile = 50
balance_column_tuna = 'score'
groupby_column_tuna = ['queryID', 'target_base']
learning_rate = 0.001

func_obs = list()
for i in range(n_tunasims_final):
    
    func_obs.append(reweightTrainer(f'tuna_{i}',
                                init_vals = init_vals,
                                bounds = bounds,
                                max_iter = tunasims_n_iter,
                                learning_rates = learning_rate,
                                balance_column= balance_column_tuna,
                                groupby_column = groupby_column_tuna))

    






