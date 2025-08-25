from sklearn.metrics import roc_auc_score
from numpy import argmax
from typing import List
from logging import getLogger
#from scipy.stats import entropy
from time import time
from numba import njit
import numpy as np
from pandas import concat

class modelLayer:

    def __init__(self,
                 candidates: List,
                 selection_method: str = 'top'):
        
        self.candidates = candidates
        self.selection_method = selection_method
        self.log = getLogger(__name__)

        #maybe add a performance metric param here

    def predict(self, data):

        return self.final_model.predict_proba(data)[:,1]
    

    def select_model(self, train, val):

        """ 
        fit the first aggregation layer tunasims to a score by tunasim groupby column
        """

        self.train_performance = list()
        self.val_performance = list()

        counter = 0
        for model in self.candidates:

            #exclude groupby and label from inputs
            model.fit(train.iloc[:,:-1], train.iloc[:,-1])

            #generate validation preds
            train_preds = model.predict_proba(train.iloc[:,:-1])[:,1]
            val_preds = model.predict_proba(val.iloc[:,:-1])[:,1]

            #track validation performance
            self.train_performance.append(roc_auc_score(train.iloc[:,-1], train_preds))
            self.val_performance.append(roc_auc_score(val.iloc[:,-1], val_preds))

            counter +=1 
            if counter % 100 == 0:
                self.log.info(f'tested {counter} aggregator models')

        if self.selection_method == 'top':

            self.final_model = self.candidates[argmax(self.val_performance)]

        else:

            #logic from ng paper goes here
            pass


class groupAdjustmentLayer:
    """
    queryAdjustmentLayer recalibrates scores based on our confidence on a query level
    how likely is the query to have a top hit that is correct, based on train data?
    inputs are: 
        - the distance from the top hit to the next hit (aggregated score) and entropy 
        - entropy of the score distribution
    adjust scores with low confidence in top
    """

    def __init__(self,
                 candidates: List = None,
                 selection_method: str = 'top',
                 groupby_column: str = 'queryID',
                 jobs: int = 1):
        
        self.candidates = candidates
        self.selection_method = selection_method
        self.groupby_column = groupby_column
        self.jobs = jobs

        self.log = getLogger(__name__)
        
        self.model_layer = modelLayer(candidates = candidates,
                                      selection_method = selection_method)
        
        
    @property
    def final_model(self):

        return self.model_layer.final_model
    
    
    @property
    def train_performance(self):

        return self.model_layer.train_performance
    
    
    @property
    def val_performance(self):

        return self.model_layer.val_performance
    
        
    def process_input_data(self, data):
        """ 
        1) Separate instances where we have multiple hits per query from single hits
        2) return both with initial index still intact
        """

        start = time()
        grouped = data.groupby(self.groupby_column).size()

        multi_hit_ids = set()
        single_hit_ids = set()
        for id, size in zip(grouped.index, grouped):

            if size > 1:
                multi_hit_ids.add(id)

            else:
                single_hit_ids.add(id)

        multi_hit_indices = list()
        single_hit_indices = list()
        self.groupby_col = data[self.groupby_column].to_numpy()
        for i, id in enumerate(data[self.groupby_column].to_numpy()):

            if id in multi_hit_ids:
                multi_hit_indices.append(i)

            else:
                single_hit_indices.append(i)

        multi_hit = data.iloc[multi_hit_indices]
        single_hit = data.iloc[single_hit_indices]
        
        #get dif from top and score entropy by groupBy column group
        groupvar_input = [group[1].to_numpy() for group in multi_hit.groupby(self.groupby_column)['preds']]
        top_from_next, entropy = groupAdjustmentLayer.get_group_vars(groupvar_input)

        #flatten arrays
        top_from_next = [x for xs in top_from_next for x in xs]
        entropy = [x for xs in entropy for x in xs]

        #add group variables to multi hit
        multi_hit.insert(0, 'top_from_next', top_from_next)
        multi_hit.insert(0, 'entropy', entropy)

        self.log.info(f'processed input data in {round((time() - start) / 360, 4)} seconds')

        #if score is present, then we are in training mode
        if 'score' in multi_hit.columns:

            return multi_hit[['preds', 'top_from_next', 'entropy', 'score']], None
        
        #otherwise we are in inference mode
        else:

            return multi_hit[['preds', 'top_from_next', 'entropy']], single_hit['preds']
    
    @staticmethod
    @njit
    def get_group_vars(preds_agg):

        top_from_next = list()
        entropy = list()

        for preds in preds_agg:

            top_from_next.append(np.zeros(preds.shape[0]) + preds[0] - preds[1])

            preds = preds / np.sum(preds)

            entropy.append(np.zeros(preds.shape[0]) - np.sum(np.log(preds) * preds))

        return top_from_next, entropy
    

    def select_model(self, train, val):
        """ 
        1) get multi hit data for train and val
        2) select best model for multi hit
        """

        #transform input to be compatible with this layer
        #don't need single hits for training
        train_, _ = self.process_input_data(train)
        val_, _ = self.process_input_data(val)

        self.train = train_

        self.model_layer.select_model(train_, val_)


    def predict(self, data):
        """ 
        1) separate single and multi hits
        2) adjust multi hits
        3) return combined predictions for single and multi
        """

        multi_hits, single_hits = self.process_input_data(data)

        #update preds on multi hits
        multi_hits['preds'] = self.model_layer.predict(multi_hits)

        #join the dfs on original index order
        return concat((multi_hits['preds'], single_hits)).reindex(data.index).to_numpy()
