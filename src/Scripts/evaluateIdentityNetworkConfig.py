#config for evaluateIdentityNetwork.py

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