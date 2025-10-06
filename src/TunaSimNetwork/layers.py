from sklearn.metrics import roc_auc_score
from typing import List
from numba import njit, typed, types
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

class ensembleLayer:

    def __init__(self,
                 candidates: List,
                 selection_method: str = 'top',
                 data_column_str: str = 'tuna'):
        
        self.candidates = candidates
        self.selection_method = selection_method
        self.data_column_str = data_column_str

        #maybe add a performance metric param here

    def predict(self, data):

        #get scores only
        pred_data = data[[i for i in data.columns if self.data_column_str in i.lower()]]

        #get other columns
        data = data[[i for i in data.columns if self.data_column_str not in i.lower()]]

        #add predictions to other values
        data['preds'] = self.final_model.predict_proba(pred_data)[:,1]

        return data
    

    def fit(self, train, val):

        """ 
        fit the first aggregation layer tunasims to a score by tunasim groupby column
        """

        self.train_performance = list()
        self.val_performance = list()

        train_score = train['score'].to_numpy()
        val_score = val['score'].to_numpy()

        train = train[[i for i in train.columns if self.data_column_str in i.lower()]]
        val = val[[i for i in val.columns if self.data_column_str in i.lower()]]

        train['score'] = train_score
        val['score'] = val_score

        for model in self.candidates:

            #exclude groupby and label from inputs
            model.fit(train.iloc[:,:-1], train.iloc[:,-1])

            #generate validation preds
            train_preds = model.predict_proba(train.iloc[:,:-1])[:,1]
            val_preds = model.predict_proba(val.iloc[:,:-1])[:,1]

            #track validation performance
            self.train_performance.append(roc_auc_score(train.iloc[:,-1], train_preds))
            self.val_performance.append(roc_auc_score(val.iloc[:,-1], val_preds))

        if self.selection_method == 'top':

            self.final_model = self.candidates[np.argmax(self.val_performance)]

        else:

            #logic from ng paper goes here
            pass

        return train, val


class groupAdjustmentLayer:
    """
    wrapper around ensemble layer, as the only difference is input transformation
    groupAdjustmentLayer recalibrates scores based on our confidence on a query level
    how likely is the query to have a top hit that is correct, based on train data?
    inputs are: 
        - the distance from the top hit to the next hit (aggregated score) and entropy 
        - entropy of the score distribution
    adjust scores with low confidence in top
    """

    def __init__(self,
                 candidates: List = None,
                 selection_method: str = 'top',
                 groupby_column: list = ['queryID'],
                 data_column_str: str = 'top_from_next'):
        
        self.candidates = candidates
        self.selection_method = selection_method
        self.groupby_column = groupby_column
        self.data_column_str = data_column_str
        
        self.model_layer = ensembleLayer(candidates = candidates,
                                         selection_method = selection_method,
                                         data_column_str = data_column_str)
        
        
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

        #grouped object that we will use repeatedly
        grouped = data.groupby(self.groupby_column)

        sizes = grouped.size()

        multi_hit_ids = set()
        single_hit_ids = set()
        for id, size in zip(sizes.index, sizes):

            if size > 1:
                multi_hit_ids.add(id)

            else:
                single_hit_ids.add(id)

        multi_hit_indices = list()
        
        for i, id in enumerate(data[self.groupby_column].to_numpy().flatten()):

            if id in multi_hit_ids:
                multi_hit_indices.append(i)

        multi_hit = data.iloc[multi_hit_indices]

        #ensure that we are sorted top to bottom match within groups
        multi_hit.sort_values(by = self.groupby_column + ['preds'], ascending = False, inplace = True)

        grouped = multi_hit.groupby(self.groupby_column)
        
        #get dif from top and score entropy by groupBy column group
        groupvar_input = [group[1].to_numpy() for group in grouped['preds']]
        top_from_next, top_from_next_pct, top_from_next_dif = groupAdjustmentLayer.get_group_vars(groupvar_input)

        #grad top hit labels
        top_hits = grouped.first()

        multi_df = pd.DataFrame({'_'.join(self.groupby_column): top_hits.index,
                                    'top_from_next': top_from_next,
                                    'top_from_next_pct': top_from_next_pct,
                                    'top_from_next_dif' : top_from_next_dif,
                                    'preds': top_hits['preds'].to_numpy()})
        
        #if we are in train mode, retain score
        if 'score' in top_hits.columns:
            multi_df['score'] = top_hits['score'].to_numpy()

        return multi_df
    
    @staticmethod
    @njit
    def get_group_vars(preds_agg):

        top_from_next = np.zeros(len(preds_agg))
        top_from_next_pct = np.zeros(len(preds_agg))
        top_from_next_dif = np.zeros(len(preds_agg))

        for i, preds in enumerate(preds_agg):

            top_from_next_ = preds[0] - preds[1]
            top_from_next[i] = top_from_next_
            top_from_next_pct[i] = top_from_next_ / preds[0]
            top_from_next_dif[i] = preds[0] - top_from_next_

        return top_from_next, top_from_next_pct, top_from_next_dif

    def fit(self, train, val):
        """ 
        1) get multi hit data for train and val
        2) select best model for multi hit
        """

        #transform input to be compatible with this layer
        #don't need single hits for training
        train = self.process_input_data(train)
        val = self.process_input_data(val)

        return self.model_layer.fit(train, val)

    def predict(self, data):
        """ 
        1) separate single and multi hits
        2) adjust multi hits
        3) return combined predictions for single and multi
        """

        multi_hits = self.process_input_data(data)

        #build adjustment dictionary
        #this dictionary contains the estimated probability of top hit being correct
        #numba friendly, also want to retain this dict as part of network
        self.adjustment_dict = typed.Dict.empty(
            key_type = types.int64,
            value_type = types.float64,
        )

        #populate dictionary
        adjustment_preds = self.model_layer.predict(multi_hits)

        for id, pred in zip(adjustment_preds[self.groupby_column].to_numpy(), adjustment_preds['preds'].to_numpy()):

            self.adjustment_dict[id] = pred

        adjusted = groupAdjustmentLayer.apply_adjustments(data[self.groupby_column].to_numpy(),
                                                      data['preds'].to_numpy(),
                                                      self.adjustment_dict)
        
        data['preds'] = adjusted
        return data
    
    @staticmethod
    @njit
    def apply_adjustments(groups: list,
                          preds: list,
                          adjustment_dict: dict):
        
        """
        adjust prior predictions if they appear in the adjustment dict
        """

        output_array = np.zeros(groups.shape[0])
        index = 0
        for group, pred in zip(groups, preds):

            if group in adjustment_dict:

                output_array[index] = pred * adjustment_dict[group]

            else:
                output_array[index] = pred

            index += 1

        return output_array
    
class tunaSimLayer:

    def __init__(self,
                 trainers,
                 residual_downsample_percentile,
                 inference_jobs = 1,
                 inference_chunk_size = 1e6):
        
        self.trainers = trainers
        self.residual_downsample_percentile = residual_downsample_percentile
        self.inference_jobs = inference_jobs
        self.inference_chunk_size = int(inference_chunk_size)
        
    def residual_downsample_tunasims(self, dataset, trainer):
        """ 
        keep only groupby column groups where residual is above threshold
        """

        dataset['preds'] = trainer.function.predict_for_dataset(dataset)
        dataset['residual'] = abs(dataset['score'] - dataset['preds'])

        if trainer.balance_column is not None:

            temp = dataset[trainer.groupby_column + list(set(['score','residual', trainer.balance_column]))].groupby(trainer.groupby_column).min()

            #gather ids above thresh for both values of balance col
            zeros = temp[temp[trainer.balance_column] == 0]
            ones = temp[temp[trainer.balance_column] == 1]

            zero_ids = self.get_ids_above_thresh(zeros)
            one_ids = self.get_ids_above_thresh(ones)

            bad_ids = zero_ids.union(one_ids)

        else:

            #we will base this off minimum residual (best score) by group
            temp = dataset[trainer.groupby_column + ['score','residual']].groupby(trainer.groupby_column).min()

            bad_ids = self.get_ids_above_thresh(temp)


        #go back and grab the indexes of all groups that were above residual threshold
        #the ids we want to compare are row level tuples of any groupby column
        residual_inds = list()
        for index, id in zip(list(range(dataset.shape[0])), 
                            [tuple(i) for i in dataset[trainer.groupby_column].to_numpy()]):

            if id in bad_ids:

                residual_inds.append(index)

        dataset = dataset.iloc[residual_inds][trainer.groupby_column + ['query', 'target', 'score']]
    
        return dataset
    
    def get_ids_above_thresh(self, dataframe):

        residual_thresh = np.percentile(dataframe['residual'], self.residual_downsample_percentile)

        #track groups where we are above residual threshold to train on in next round
        #index of temp is now tuples of the groupby column
        bad_ids = set()
        for residual, id in zip(dataframe['residual'], 
                                       dataframe.index):

            if residual >= residual_thresh:
                bad_ids.add(id)

        return bad_ids


    def fit(self, dataset):

        #fit and update train performance for each round of residuals
        for trainer in self.trainers:

            #fit on remaining train data
            trainer.fit(dataset)
            trainer.trained = True

            #downsample from train before fitting next model
            dataset = self.residual_downsample_tunasims(dataset, trainer)

        self.trainers = [trainer for trainer in self.trainers if trainer.trained == True]

    def predict(self, dataset):
        """ 
        generate predictions on full datasets
        """

        #to minimize params, we will infer the groupby column from trainers
        groupby_column = self.trainers[0].groupby_column
        groupby_column_values = dataset[groupby_column].to_numpy()

        #for memory management purposes, break dataset into smaller chunks
        n_chunks = (dataset.shape[0] // self.inference_chunk_size) + 2
        chunk_inds = [i * self.inference_chunk_size for i in range(n_chunks)]
        score_outputs = list()
        for i in range(n_chunks - 1):

            start_ind, end_ind = chunk_inds[i], chunk_inds[i + 1]
        
            #collect preds by model by dataset
            preds = Parallel(n_jobs = self.inference_jobs)(delayed(trainer.final_function.predict_for_dataset)
                                                           (dataset.iloc[start_ind:end_ind][['query', 'target']])
                                                            for trainer in self.trainers)
            
            #convert to dictionary
            preds = {name: pred_array for name, pred_array in zip([i.name for i in self.trainers], preds)}

            #keep prediction subset
            score_outputs.append(pd.DataFrame(preds))

        preds = pd.concat(score_outputs, axis = 0, ignore_index = True)
        preds[groupby_column] = groupby_column_values

        if 'score' in dataset.columns:
            preds['score'] = dataset['score'].to_numpy()

        return preds.groupby(groupby_column).max().reset_index(drop = False)
    