from sklearn.metrics import roc_auc_score
from numpy import argmax
from typing import List
from logging import getLogger
from joblib import Parallel, delayed
from scipy.stats import entropy
from time import time


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
        for model in self.candidate_models:

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


class queryAdjustmentLayer:

    def __init__(self,
                 candidates: List = None,
                 selection_method: str = 'top',
                 groupby_column: List = ['queryID'],
                 jobs: int = -1):
        
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
        Only want to take in instances where there is more than 1 hit
        """

        start = time.time()
        grouped = data.groupby(self.groupby_column).size()

        multi_hit_ids = set()
        single_hit_indices = set()
        for id, size in zip(grouped.index, grouped):

            if size > 1:
                multi_hit_ids.add(id)

            else:
                single_hit_indices.add(id)

        multi_hit_indices = list()
        single_hit_indices = list()
        for i, id in enumerate(data[self.groupby_column]):

            if id in multi_hit_ids:
                multi_hit_indices.append(i)

            else:
                single_hit_indices.append(i)

        multi_hit = data.iloc[multi_hit_indices]
        single_hit = data.iloc[single_hit_indices]

        groupvars = Parallel(n_jobs = self.jobs)(delayed(self.get_group_vars)(group[1]['preds']) for group in multi_hit.groupby(self.groupby_column))

        top_from_next = [i for group in groupvars for i in group[0]]
        entropies = [i for group in groupvars for i in group[1]]

        multi_hit.insert(-2, 'top_from_next', top_from_next)
        multi_hit.insert(-2, 'entropy', entropies)

        self.logger.info(f'processed input data in {round((time.time() - start) / 360, 4)} seconds')

        return multi_hit[['preds', 'top_from_next', 'entropy']], single_hit['preds']
    
    
    def get_group_vars(self, preds):

        top_from_next = preds[0] - preds[1]
        entropy_ = entropy(preds)

        return [top_from_next for _ in range(len(preds))], [entropy_ for _ in range(len(preds))]
    

    def select_model(self, train, val):

        train, _ = self.process_input_data(train)
        val, _ = self.process_input_data(val)

        self.model_layer.select_model(train, val)


    def predict(self, data):

        multi_hits, single_hits = self.process_input_data(data)

        #update preds on multi hits
        multi_hits['preds'] = self.model_layer.predict(multi_hits)

        return multi_hits.join(single_hits)['preds'].to_numpy()
