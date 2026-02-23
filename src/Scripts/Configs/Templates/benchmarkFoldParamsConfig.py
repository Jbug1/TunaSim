from rdkit.Chem.rdRascalMCES import RascalOptions
import numpy as np

combined_path: str
output_directory: str
log_path: str = output_directory
param_sets = list()
max_groups = 8

options_sets = list()

options_0 = RascalOptions()
options_0.timeout = 10
options_0.maxBondMatchPairs = 2000
options_0.similarityThreshold = 0.9

options_1 = RascalOptions()
options_1.timeout = 1
options_1.maxBondMatchPairs = 10000
options_1.similarityThreshold = 0.9

options_2 = RascalOptions()
options_2.timeout = 1
options_2.maxBondMatchPairs = 2000
options_2.similarityThreshold = 0.9

options_3 = RascalOptions()
options_3.timeout = 1
options_3.maxBondMatchPairs = 2000
options_3.similarityThreshold = 0.8

options_4 = RascalOptions()
options_4.timeout = 1
options_4.maxBondMatchPairs = 2000
options_4.similarityThreshold = 0.7

for opts in [options_1]:
    for jobs in [1,2,4,8]:

        options_sets.append((opts,10,2,jobs, f'{output_directory}/{jobs}_jobs'))


# for opts in [options_0, options_1, options_2, options_3, options_4]:

#     options_sets.append((opts, 20, 2, ))



