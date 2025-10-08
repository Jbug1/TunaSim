import pandas as pd
from logging import getLogger
import time
from os import makedirs

class IdentityMatchNetwork:
    """ 
    trains all layers of a tunaSim network to predict whether two spectra come from the same molecule
    logging done at this level
    """

    def __init__(self,
                 train_path: str,
                 val_1_path: str,
                 val_2_path: str,
                 intermediate_outputs_path: str,
                 tunaSim_layer,
                 ensemble_layer,
                 query_adjustment_layer
                 ):
        
        self.train_path = train_path
        self.val_1_path = val_1_path
        self.val_2_path = val_2_path
        self.intermediate_outputs_path = intermediate_outputs_path
        self.tunaSim_layer = tunaSim_layer
        self.ensemble_layer = ensemble_layer
        self.query_adjustment_layer = query_adjustment_layer

        makedirs(self.intermediate_outputs_path, exist_ok = True)
        
        self.log = getLogger(__name__)

    def fit(self):
        """ 
        function that runs script to fit the whole network
        """

        overall_start = time.time()

        #fit initial tunasims
        self.log.info('beginning tunaSim training')
        self.tunaSim_layer.fit(pd.read_pickle(self.train_path))

        #create tunasim preds
        self.log.info('creating tunasim predictions train')
        train_tunasim_preds = self.tunaSim_layer.predict(pd.read_pickle(self.train_path))
        train_tunasim_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_top_train.csv', index = False)

        self.log.info('creating tunasim predicitons val_1')
        val_1_tunasim_preds = self.tunaSim_layer.predict(pd.read_pickle(self.val_1_path))
        val_1_tunasim_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_1.csv', index = False)

        self.log.info('creating tunasim predictions val_2')
        val_2_tunasim_preds = self.tunaSim_layer.predict(pd.read_pickle(self.val_2_path))
        val_2_tunasim_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_2.csv', index = False)

        #select a tunasim aggregator from among candidates
        self.log.info('beginning ensemble layer')
        self.ensemble_layer.fit(train = pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_train.csv'), 
                                val = pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_1.csv'))
        
        #create train aggregated preds
        self.log.info('beginning train aggregated predictions')
        train_aggregated_preds = self.ensemble_layer.predict(pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_train.csv'))
        train_aggregated_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_train.csv', index = False)
        
        #create val 1 aggregated preds
        self.log.info('beginning train consolidated predictions')
        val_1_aggregated_preds = self.ensemble_layer.predict(pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_1.csv'))
        val_1_aggregated_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_val_1.csv', index = False)

        #create val 2 aggregated preds
        self.log.info('beginning train aggregated predictions')
        val_2_aggregated_preds = self.ensemble_layer.predict(pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_top_val_2.csv'))
        val_2_aggregated_preds.to_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_val_2.csv', index = False)

        #fit group adjustment layer
        #train dataset now includes the first validation dataset
        self.log.info('beginning query adjustment')
        adjustment_train, adjustment_val = self.query_adjustment_layer.fit(train = pd.concat([pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_train.csv'),
                                                                            pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_val_1.csv')]),
                                                                            val = pd.read_csv(f'{self.intermediate_outputs_path}/tunasims_aggregated_top_val_2.csv'))

        adjustment_train.to_csv(f'{self.intermediate_outputs_path}/adjustment_train.csv', index = False)
        adjustment_val.to_csv(f'{self.intermediate_outputs_path}/adjustment_val.csv', index = False)

        self.log.info(f'network training complete in {round((time.time() - overall_start) / 60, 4)} minutes')


    def predict(self, dataset, write_intermediates = False):
        """ 
        pass between layers with no need for intermediate outputs
        break the dataset on your own if you must
        """

        #tunasim layer
        layer_output = self.tunaSim_layer.predict(dataset)

        if write_intermediates:
            layer_output.to_csv(f'{self.intermediate_outputs_path}/tunaSim_output.csv', index = False)

        #ensemble layer
        layer_output = self.ensemble_layer.predict(layer_output)

        if write_intermediates:
            layer_output.to_csv(f'{self.intermediate_outputs_path}/ensemble_output.csv', index = False)

        #query adjustment layer
        layer_output = self.query_adjustment_layer.predict(layer_output)

        return layer_output
