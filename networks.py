import pandas as pd
from numpy import percentile, argmax
from logging import getLogger
from sklearn.metrics import roc_auc_score
import time
import oldMetrics
from os import makedirs
from pickle import dump

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
                 intermediate_outputs_path: str,
                 tunaSim_trainers: list,
                 tunaSim_aggregation_candidates: list,
                 scoreByGroup_trainers: list,
                 scoreByGroup_aggregation_candidates: list,
                 residual_downsampling_percentile: int = 50,
                 aggregator_selection_method: str = 'top'
                 ):
        
        self.train_path = train_path
        self.val_1_path = val_1_path
        self.val_2_path = val_2_path
        self.test_path = test_path
        self.intermediate_outputs_path = intermediate_outputs_path
        self.tunaSim_trainers = tunaSim_trainers
        self.tunaSim_aggregation_candidates = tunaSim_aggregation_candidates
        self.scoreByGroup_trainers = scoreByGroup_trainers
        self.scoreByGroup_aggregation_candidates = scoreByGroup_aggregation_candidates
        self.residual_downsamping_percentile = residual_downsampling_percentile
        self.aggregator_selection_method = aggregator_selection_method

        makedirs(self.intermediate_outputs_path, exist_ok = True)
        makedirs(f'{self.intermediate_outputs_path}/pickled_objects', exist_ok = True)
        makedirs(f'{self.intermediate_outputs_path}/performance', exist_ok = True)
        
        self.log = getLogger(__name__)

    def fit(self):
        """ 
        function that runs script to fit the whole network
        """

        # overall_start = time.time()

        #fit initial tunasims
        self.log.info('beginning initial tunasims')
        self.fit_tunasims(pd.read_pickle(self.train_path))

        #create tunasim preds
        self.log.info('creating tunasim preds train')
        train_tunasim_preds = self.create_tunasim_preds(pd.read_pickle(self.train_path))
        train_tunasim_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_top_train.csv')
        del(train_tunasim_preds)

        self.log.info('creating tunasim preds val_1')
        val_1_tunasim_preds = self.create_tunasim_preds(pd.read_pickle(self.val_1_path))
        val_1_tunasim_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_1.csv')
        del(val_1_tunasim_preds)

        # self.log.info('creating tunasim preds val_2')
        # val_2_tunasim_preds = self.create_tunasim_preds(pd.read_pickle(self.val_2_path))
        # val_2_tunasim_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_2.csv')
        # del(val_2_tunasim_preds)

        #select a tunasim aggregator from among candidates
        self.log.info('beginning tunasim aggregator')
        self.tunasim_aggregator, train_aucs, val_aucs = self.select_aggregator(train = pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_train.csv').iloc[:,2:], 
                                                         val = pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_1.csv').iloc[:,2:],
                                                         candidate_models = self.tunaSim_aggregation_candidates)
        
        pd.DataFrame({'train': train_aucs,
                    'val': val_aucs}).to_csv(f'{self.intermediate_outputs_path}/performance/tuna_agg_performances.csv')
        
        self.log.info('tunasim aggreagtor selected')
        
        #create train aggregated preds
        train_aggregated_preds = self.create_tunasim_aggregated_preds(pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_train.csv'))
        train_aggregated_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_train.csv')
        del(train_aggregated_preds)

        #create val 1 aggregated preds
        val_1_aggregated_preds = self.create_tunasim_aggregated_preds(pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_1.csv'))
        val_1_aggregated_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_val_1.csv')
        del(val_1_aggregated_preds)

        #create val 2 aggregated preds
        val_2_aggregated_preds = self.create_tunasim_aggregated_preds(pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_2.csv'))
        val_2_aggregated_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_val_2.csv')
        del(val_2_aggregated_preds)

        print(stop)

        #fit group adjustment layer
        #train dataset now includes the first validation dataset
        self.log.info('beginning adjustment tunasims')
        self.fit_adjustment_tunasims(pd.concat([pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_train.csv'),
                                                pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_train.csv')]),
                                     val = pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_train.csv'))

        #create adjustment layer preds
        self.log.info('creating adjustment preds')
        self.create_adjustment_preds()

        #fit adjustment aggregator
        self.log.info('begining adjustment aggregator')
        self.fit_adjustment_aggregator()

        self.log.info(f'network training complete in {round((time.time() - overall_start) / 60), 4} minutes')

    def residual_downsample_tunasims(self, dataset, trainer):
        """ 
        keep only groupby column groups where residual is above threshold
        """

        dataset['preds'] = trainer.function.predict_for_dataset(dataset)
        dataset['residual'] = abs(dataset['score'] - dataset['preds'])

        #we will base this off minimum residual (best score) by group
        temp = dataset[trainer.groupby_column + ['score','residual']].groupby(trainer.groupby_column).min()

        residual_thresh = percentile(temp['residual'], self.residual_downsamping_percentile)
        self.log.info(f"percent positive before downsample: {round(len(temp[temp['score'] == 1])/len(temp) * 100,4)}")

        #track balance column distribution from groups above residual thresh
        pos = 0
        neg = 0

        #track groups where we are above residual threshold to train on in next round
        #index of temp is now tuples of the groupby column
        bad_ids = set()
        for residual, id, score in zip(temp['residual'], 
                                       temp.index,
                                        temp['score']):

            if residual >= residual_thresh:
                bad_ids.add(id)

                if score == 1:
                    pos+=1

                else:
                    neg+=1

        #go back and grab the indexes of all groups that were above residual threshold
        #the ids we want to compare are row level tuples of any groupby column
        residual_inds = list()
        for index, id in zip(list(range(dataset.shape[0])), 
                            [tuple(i) for i in dataset[trainer.groupby_column].to_numpy()]):

            if id in bad_ids:

                residual_inds.append(index)

        dataset = dataset.iloc[residual_inds][trainer.groupby_column + ['query', 'target', 'score']]
        self.log.info(f'new train length post {trainer.name}: {len(dataset)}')
        self.log.info(f'percent positive after downsample post {trainer.name}: {round(pos / (pos + neg) * 100, 4)}')

        return dataset

    def fit_tunasims(self, dataset):

        #fit and update train performance for each round of residuals
        for trainer in self.tunaSim_trainers:

            start = time.time()

            #fit on remaining train data
            trainer.fit(dataset)

            #downsample from train before fitting next model
            dataset = self.residual_downsample_tunasims(dataset, trainer)

            dataset.to_pickle(f'{self.intermediate_outputs_path}/{trainer.name}_post_downsample.pkl')

            with open(f'{self.intermediate_outputs_path}/pickled_objects/{trainer.name}.pkl', 'wb') as handle:

                dump(trainer.function, handle)

            self.log.info(f'trained {trainer.name} in {round((time.time() - start) / 60, 4)} minutes')

            #check that we have two classes left
            remaining_labels = list(set(dataset['score']))
            if  len(remaining_labels) == 1:

                self.log.info(f'ran out of {not remaining_labels[0]} lables. Exiting tunasim fitting')
                break

    def create_tunasim_preds(self, dataset):
        """ 
        generate predictions on full datasets
        """

        groupby_column = self.tunaSim_trainers[0].groupby_column
        
        #collect preds by model by dataset
        preds = dict()
        for trainer in self.tunaSim_trainers:

            preds[trainer.name] = trainer.function.predict_for_dataset(dataset)
            self.log.info(f'created preds for {trainer.name}')

        #write out dataset from collected preds
        preds = pd.DataFrame(preds)
        preds[groupby_column] = dataset[groupby_column]
        preds['score'] = dataset['score']
        return preds.groupby(groupby_column).max()
        

    def select_aggregator(self,
                          train,
                          val,
                          candidate_models):
        
        """ 
        fit the first aggregation layer tunasims to a score by tunasim groupby column
        """

        train_performance = list()
        val_performance = list()

        counter = 0
        for model in candidate_models:

            #exclude groupby and label from inputs
            model.fit(train.iloc[:,:-1], train.iloc[:,-1])

            #generate validation preds
            train_preds = model.predict_proba(train.iloc[:,:-1])[:,1]
            val_preds = model.predict_proba(val.iloc[:,:-1])[:,1]

            #track validation performance
            train_performance.append(roc_auc_score(train.iloc[:,-1], train_preds))
            val_performance.append(roc_auc_score(val.iloc[:,-1], val_preds))

            counter +=1 
            if counter % 100 == 0:
                self.log.info(f'tested {counter} aggregator models')

        if self.aggregator_selection_method == 'top':

            aggregation_model = self.tunaSim_aggregation_candidates[argmax(val_performance)]

        else:

            #logic from ng paper goes here
            pass

        return aggregation_model, train_performance, val_performance


    def create_tunasim_aggregated_preds(self):
        """ 
        aggregate tunasims and generate next layer of data
        """
        pass


    def fit_query_adjustment_tunasims(self):
        """ 
        fitting of 'curve' adjustment for each query
        """
        pass




    def evaluate_tunasim_performance(self):
        """ 
        
        """

        pass

    def evaluate_old_metric_performance(self):
        pass
