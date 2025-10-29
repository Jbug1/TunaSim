#config for evaluateIdentityNetwork.py

#logging
log_path = '/Users/jonahpoczobutt/projects/TunaRes/network_logs_metlin_bs_only/'
results_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results_metlin_bs_only/'

#path to network pickle file
network_path = '/Users/jonahpoczobutt/projects/TunaRes/network_results_bs_only/network.pkl'

#mandatory dataset params
dataset_names = ['metlin_test']

#optional dataset building params
build_datasets = True

query_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/metlin_noprec_clean.pkl'
target_input_path = '/Users/jonahpoczobutt/projects/raw_data/db_csvs/metlin_noprec_clean.pkl'
dataset_max_sizes = [1e7]

ppm_match_window = 10
identity_column = 'inchi_base'
tolerance = 0.01
units_ppm = False
identity_column = 'inchi_base'

evaluate_old_metrics = True

matches_input_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results_metlin_bs_only/'
write_intermediates = True
reweighted = True

network_performance_attribution = True

#network performance attribution needs intermediate results
if network_performance_attribution:

    write_intermediates = True