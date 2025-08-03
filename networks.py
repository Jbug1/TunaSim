import pandas as pd
from numpy import percentile, argmax
from logging import getLogger
from sklearn.metrics import roc_auc_score
import time


class IdentityMatchNetwork:
    """ 
    trains all layers of a tunaSim network to predict whether two spectra come from the same molecule

    optionally tracks validation and test performance
    """

    def __init__(self,
                 train_path: str,
                 val_1_path: str,
                 val_2_path: str,
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
        self.val_1_path = val_1_path
        self.val_2_path = val_2_path
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

    def fit_network(self):
        """ 
        function that runs script to fit the whole network
        """

        overall_start = time.time()

        #fit initial tunasims
        self.log.info('beginning initial tunasims')
        self.fit_tunasims(pd.read_pickle(self.train_path))

        #create tunasim preds
        self.log.info('creating tunasim preds')
        if self.track_test_performance:
            test_tunasim_preds = self.create_tunasim_preds(pd.read_pickle(self.test_path))
            test_tunasim_preds.to_csv(f'{self.scratch_path}/tunasims_top_test.csv')
            del(test_tunasim_preds)

        train_tunasim_preds = self.create_tunasim_preds(pd.read_pickle(self.train_path))
        train_tunasim_preds.to_csv(f'{self.scratch_path}/tunasims_top_train.csv')
        del(train_tunasim_preds)

        val_1_tunasim_preds = self.create_tunasim_preds(pd.read_pickle(self.val_1_path))
        val_1_tunasim_preds.to_csv(f'{self.scratch_path}/tunasims_top_val_1.csv')
        del(val_1_tunasim_preds)

        val_2_tunasim_preds = self.create_tunasim_preds(pd.read_pickle(self.val_2_path))
        val_2_tunasim_preds.to_csv(f'{self.scratch_path}/tunasims_top_val_2.csv')
        del(val_2_tunasim_preds)

        #select a tunasim aggregator from among candidates
        self.log.info('beginning tunasim aggregator')
        self.select_tunasim_aggregator(train = pd.read_csv(f'{self.scratch_path}/tunasims_top_train.csv'), 
                                       val = pd.read_csv(f'{self.scratch_path}/tunasims_top_val_1.csv'))

        #persist tunasim pred aggregations
        self.log.info('creating aggregated tunasim preds')
        if self.track_test_performance:
            test_aggregated_preds = self.create_tunasim_aggregated_preds(pd.read_csv(f'{self.scratch_path}/tunasims_top_test.csv'))
            test_aggregated_preds.to_csv(f'{self.scratch_path}/tunasims_aggregated_top_test.csv')
            del(test_aggregated_preds)

        #create train aggregated preds
        train_aggregated_preds = self.create_tunasim_aggregated_preds(pd.read_csv(f'{self.scratch_path}/tunasims_top_train.csv'))
        train_aggregated_preds.to_csv(f'{self.scratch_path}/tunasims_aggregated_top_train.csv')
        del(train_aggregated_preds)

        #create val 1 aggregated preds
        val_1_aggregated_preds = self.create_tunasim_aggregated_preds(pd.read_csv(f'{self.scratch_path}/tunasims_top_val_1.csv'))
        val_1_aggregated_preds.to_csv(f'{self.scratch_path}/tunasims_aggregated_top_val_1.csv')
        del(val_1_aggregated_preds)

        #create val 2 aggregated preds
        val_2_aggregated_preds = self.create_tunasim_aggregated_preds(pd.read_csv(f'{self.scratch_path}/tunasims_top_val_2.csv'))
        val_2_aggregated_preds.to_csv(f'{self.scratch_path}/tunasims_aggregated_top_val_2.csv')
        del(val_2_aggregated_preds)

        print(yool)

        #fit group adjustment layer
        #train dataset now includes the first validation dataset
        self.log.info('beginning adjustment tunasims')
        self.fit_adjustment_tunasims(pd.concat([pd.read_csv(f'{self.scratch_path}/tunasims_aggregated_top_train.csv'),
                                                pd.read_csv(f'{self.scratch_path}/tunasims_aggregated_top_train.csv')]),
                                     val = pd.read_csv(f'{self.scratch_path}/tunasims_aggregated_top_train.csv'))

        #create adjustment layer preds
        self.log.info('creating adjustment preds')
        self.create_adjustment_preds()

        #fit adjustment aggregator
        self.log.info('begining adjustment aggregator')
        self.fit_adjustment_aggregator()

        #throw in final preds by default for good measure
        self.log.info('creating final preds')
        self.create_final_preds()

        self.log.info(f'network training complete in {round((time.time() - overall_start) / 60), 4} minutes')


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

    def fit_tunasims(self, dataset):

        #fit and update train performance for each round of residuals
        for model in self.tunaSim_funcObs:

            start = time.time()

            #fit on remaining train data
            model.fit(dataset)
            
            #note performance
            dataset['preds'] = model.predict_for_dataset(dataset)
            self.update_performance_tracker('residual_train', dataset)

            #downsample from train before fitting next model
            dataset = self.residual_downsample(dataset)

            self.log.info(f'trained {model.name} in {round((time.time() - start) / 60), 4} minutes')

    def create_tunasim_preds(self, dataset):
        """ 
        generate predictions on full datasets
        """

        groupby_column = self.tunaSim_funcObs[0].groupby_column
        
        #collect preds by model by dataset
        preds = list()
        for model in self.tunaSim_funcObs:

            preds.append(model.sim_func.predict_for_dataset(dataset))

        #write out dataset from collected preds
        preds = pd.DataFrame(preds, columns = [_.name for _ in self.tunaSim_funcObs])
        preds[groupby_column] = dataset[groupby_column]
        preds['score'] = dataset['score']
        return preds.groupby(groupby_column).max()
        

    def select_tunasim_aggregator(self):
        """ 
        fit the first aggregation layer tunasims to a score by tunasim groupby column
        """

        #these should both be small enough to hold at once
        train = f'{self.scratch_path}/tunasim_top_train.csv'
        val = f'{self.scratch_path}/tunasim_top_val.csv'

        val_performance = list()

        for model in self.sim_consolidation_candidates:

            #exclude groupby and label from inputs
            model.fit(train.iloc[:,:-1], train.iloc[:,-1])

            #generate validation preds
            val_preds = model.predict(val.iloc[:,:-1])

            #track validation performance
            val_performance.append(roc_auc_score(val.iloc[:,-1], val_preds))

        if self.val_performance_selector == 'max':

            self.sim_consolidation_model = self.sim_consolidation_candidates[argmax(val_performance)]

        else:

            #logic from ng paper goes here
            pass 


    def create_tunasim_aggregated_preds(self):
        """ 
        aggregate tunasims and generate next layer of data
        """

        aggregated




    def fit_query_adjustment_tunasims(self):
        """ 
        fitting of 'curve' adjustment for each query
        """




    def evaluate_tunasim_performance(self):
        """ 
        
        """

        pass



















        
















            



        

        
