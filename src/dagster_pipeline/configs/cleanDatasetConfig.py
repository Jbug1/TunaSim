# config for cleaned_dataset asset
base_output_dir="/Users/jonahpoczobutt/projects/TunaRes/pipeline_output",
raw_input_path="/Users/jonahpoczobutt/projects/raw_data/highres_pickles/nist23_msg_train_combined.pkl",
noise_threshold: float = 0.01
precursor_removal_window_mz: float = 2.0
deisotoping_gaps: list[float] = []
isotope_mz_tolerance: float = 0.
