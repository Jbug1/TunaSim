#config for evaluateIdentityNetwork.py
from sklearn.ensemble import HistGradientBoostingClassifier as gbc

#logging
log_path = '/Users/jonahpoczobutt/projects/TunaRes/network_logs_newmenthod'
results_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results_newmethod'

#path to network pickle file
network_path = '/Users/jonahpoczobutt/projects/TunaRes/network_results_newmethod/network.pkl'

#mandatory dataset params
dataset_names = ['test']

#optional dataset building params
build_datasets = False

query_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/nist23_metlin_overlap_noprec.pkl'
target_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/nist23_metlin_overlap_noprec.pkl'
dataset_max_sizes = [1e7]

ppm_match_window = 10
identity_column = 'inchi_base'
tolerance = 0.01
units_ppm = False
identity_column = 'inchi_base'

evaluate_old_metrics = True

matches_input_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results_newmethod'
write_intermediates = True
reweighted = True

network_performance_attribution = True

#network performance attribution needs intermediate results
if network_performance_attribution:

    write_intermediates = True

#additional feature eval
additional_feature_eval = True
train_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results_newmethod/intermediate_outputs'

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
