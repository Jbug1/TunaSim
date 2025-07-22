import pandas as pd
from numpy import percentile
from logging import getLogger
from sklearn.metrics import roc_auc_score
import time


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
                 groupby_column: str,
                 tunaSim_funcObs: list,
                 sim_consolidation_candidates: list,
                 score_by_group_funcObs: list,
                 output_layer_candidates: list,
                 track_test_performance: bool = False,
                 track_val_performance: bool = False,
                 residual_downsampling_percentile = 50,
                 ):
        
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.scratch_path = scratch_path
        self.score_column = score_column
        self.groupby_column = groupby_column
        self.tunaSim_funcObs = tunaSim_funcObs
        self.sim_consolidation_candidates = sim_consolidation_candidates
        self.score_by_group_funcObs = score_by_group_funcObs
        self.output_layer_candidates = output_layer_candidates
        self.track_test_performance = track_test_performance
        self.track_val_performance = track_val_performance
        self.residual_downsamping_percentile = residual_downsampling_percentile
        
        self.log = getLogger()

    def update_performance_tracker(self, name, dataset):

        try:
            old = pd.read_csv(f'{self.scratch_path}/{name}.csv')
            temp = dataset.groupby(by=self.groupby_column).apply(lambda x: x[x['preds'] == max(x['preds'])].iloc[0])
            new = pd.concat((old, {'name': name, 'score': roc_auc_score(temp['score'] , temp['preds'])}))

        except:
            temp = dataset.groupby(by=self.groupby_column).apply(lambda x: x[x['preds'] == max(x['preds'])].iloc[0])
            new = pd.DataFrame({'name' : name, 'score' : roc_auc_score(temp['score'] , temp['preds'])})

        new.to_csv(f'{self.scratch_path}/{name}.csv')

    def residual_downsample(self, dataset):
        """ 
        keep only groupby column groups where residual is above threshold
        """

        dataset['residual'] = abs(dataset['score'] - dataset['preds'])

        temp = dataset.groupby(self.groupby_column).min()

        residual_thresh = percentile(temp['residual'], self.residual_downsamping_percentile)
        self.log.info(f'percent positive before downsample: {round(len(temp[temp['score'] == 1])/len(temp) * 100,4)}')

        pos = 0
        neg = 0
        bad_ids = list()
        for i in range(len(temp)):

            if temp.iloc[i]['residual'] >= residual_thresh:
                bad_ids.append(temp.iloc[i][self.groupby_column])

                if temp.iloc[i]['score'] == 1:
                    pos+=1

                else:
                    neg+=1

        bad_ids = set(bad_ids)

        residual_inds = list()

        for i in range(len(dataset)):

            if dataset.iloc[i]['queryID_target_base'] in bad_ids:

                residual_inds.append(i)

        dataset = dataset.iloc[residual_inds]
        self.log.info(f'new train length: {len(dataset)}')
        self.log.info(f'percent positive after downsample: {pos / (pos + neg)}')

        return dataset

    def fit_tunasims(self):

        #read in train data
        train = pd.read_pickle(self.train_path)

        original_train_len = train.shape[0]

        #fit and update train performance for each round of residuals
        for model in self.tunaSim_funcObs:

            start = time.time()

            #fit on remaining train data
            model.fit(train)
            
            #note performance
            train['preds'] = model.predict_for_dataset(train)
            self.update_performance_tracker('residual_train', train)

            #downsample from train before fitting next model
            train = self.residual_downsample(train)

            self.log.info(f'trained {model.name} in {round((time.time() - start) / 60), 4} minutes')

    def get_tunasim_preds_and_performances(self):

        dataset_names = ['train']
        datasets = [self.train_path]

        if self.track_val_performance:
            dataset_names.append('val')
            datasets.append(self.val_path)

        if self.track_test_performance:
            dataset_names.append('test')
            datasets.append(self.test_path)
        
        for i in range(len(dataset_names)):

            dataset = pd.read_pickle(datasets[i])

            









        
















            



        

        
