# config for training_datasets asset

dataset_names: list[str] = ['train', 'val_1', 'val_2']
dataset_max_sizes: list[float] = [1e7, 1e7, 1e7]
identity_column: str = 'inchi_base'
ppm_match_window: int = 10
tolerance: float = 0.01
units_ppm: bool = False
