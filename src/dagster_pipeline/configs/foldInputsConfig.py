# config for fold_assignments asset
n_jobs: int = 4
mces_cutoff: int = 90
fold_names: list[str] = ['train', 'val1', 'val2', 'test']

# source name -> fold assignment (None = distributed by algorithm)
fold_mappings: dict = {
    'metlin': 'test',
}
