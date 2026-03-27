# config for training_datasets asset

dataset_max_sizes = {'train': 1.2e7,
                     'val1': 1.2e7,
                     'val2': 1.2e7,
                     'test': 1e9}

identity_column: str = 'inchikey_base'
ppm_match_window: int = 10
tolerance: float = 0.01
units_ppm: bool = False
break_datasets_by_source = True
