#config for evaluateIdentityNetwork.py
from sklearn.ensemble import HistGradientBoostingClassifier as gbc

#logging
log_path: str = ''
results_directory: str = ''

#path to network pickle files
network_path: str = ''

#mandatory dataset params
dataset_names: list[str] = ['']

#optional dataset building params
build_datasets: bool = False

query_input_path: str = ''
target_input_path: str = ''
dataset_max_sizes: list[int] = [1e7]

ppm_match_window: float = 10
identity_column: str = ''
tolerance: float = 0.01
units_ppm: bool = False
identity_column: str = ''

evaluate_old_metrics: bool = True

matches_input_directory: str = ''
write_intermediates: bool = True
reweighted: bool = True

network_performance_attribution: bool = True

#network performance attribution needs intermediate results
if network_performance_attribution:

    write_intermediates = True

#additional feature eval
additional_feature_eval: bool = False
train_directory: str = ''

selection_method: str = 'top'
learning_rates: list[float] = []
max_leaf_nodes: list[int] = []
max_iter: list[int] = []
l2_regs: list[int] = []
max_bins: list[int] = []

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
