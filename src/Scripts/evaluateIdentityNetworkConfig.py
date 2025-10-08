#config for evaluateIdentityNetwork.py

#logging
log_path = '/Users/jonahpoczobutt/projects/TunaRes/network_logs'
results_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results/eval'

network_path = '/Users/jonahpoczobutt/projects/TunaRes/network_results/network.pkl'

build_datasets = False
dataset_names = ['test_stub']

evaluate_old_metrics = True

matches_input_directory = '/Users/jonahpoczobutt/projects/TunaRes/network_results/matched'
write_intermediates = True
reweighted = True

network_performance_attribution = True

#network performance attribution needs intermediate results
if network_performance_attribution:

    write_intermediates = True

