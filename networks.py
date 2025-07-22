import pandas as pd
import numpy as np
from logging import getLogger


class coreMatchNetwork:
    """ 
    trains all layers of a tunaSim network to predict whether two spectra come from the same molecule

    optionally tracks validation and test performance
    """

    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 scratch_path: str,
                 score_column: str,
                 tunaSim_funcObs: list,
                 sim_consolidation_candidates: list,
                 score_by_group_funcObs: list,
                 output_layer_candidates: list,
                 track_test_performance: bool = False,
                 track_val_performance: bool = False,
                 residual_downsampling_percentile = lambda x: np.percentile(x, 50),
                 ):
        
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.scratch_path = scratch_path
        self.score_column = score_column
        self.tunaSim_funcObs = tunaSim_funcObs
        self.sim_consolidation_candidates = sim_consolidation_candidates
        self.score_by_group_funcObs = score_by_group_funcObs
        self.output_layer_candidates = output_layer_candidates
        self.track_test_performance = track_test_performance
        self.track_val_performance = track_val_performance
        self.residual_downsamping_percentile = residual_downsampling_percentile
        
        self.log = getLogger()


    def fit(self):

        #read in train data
        train = pd.read_pickle(self.train_path)

        #fit 
        for model in self.tunaSim_funcObs:

            model.fit(train)

            



        

        
