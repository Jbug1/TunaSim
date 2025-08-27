import tunas
import numpy as np
from typing import List
import copy
from collections import deque
import time
from logging import getLogger
from numba import njit

@njit
def square_loss_grad(x, y):

        return 2 * (x - y)


class funcTrainer:
    ''' 
    name: what to call this func ob 
    function: tuna sim function that maps input spectra to 0-1 interval
    init_vals: initial values for TUNABLE PARAMETERS only
    params: tunable parameters by name
    param_grad: first derivative of the loss function (y-y_hat)
    param_grad2: second derivative of the loss function (y-y_hat)
    regularization_func: self explanatory
    regularization_grad: easily applyable grad of reg func
    loss_func: any transformation of Y- y_hat
    regularization_name: name of func
    loss_name: name of func
    solver: name of method to use
    lambdas: update step size
    '''

    def __init__(
            self,
            name: str,
            init_vals: dict,
            fixed_vals: dict = None,
            learning_rate: float = 0.01,
            max_iter: int = 1e5,
            bounds: list = None,
            learning_rate_scheduler: str = None,
            learning_beta: float = 0.5,
            ad_int: float = 0.8,
            ad_slope: float  = 0.3,
            scale_holdover_vals: int = 2,
            groupby_column: str = None,
            balance_column: str = None
    ):
        
        self.name = name
        self.init_vals = init_vals
        self.bounds = bounds
        self.max_iter = int(max_iter)
        self.n_iter = 0
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_beta = learning_beta
        self.ad_int = ad_int
        self.ad_slope = ad_slope
        self.scale_holdover_vals = scale_holdover_vals
        self.groupby_column = groupby_column
        self.balance_column = balance_column

        self.balance_flag = 0

        self.log = getLogger(__name__)

        self.input_lengths = list()

        #set scheduling dictionaries
        self.accumulated_gradients = {key: 0 for key in self.init_vals}
        self.accumulated_scales = {key: 0 for key in self.init_vals}
        self.scale_holdovers = {key: deque([1 for i in range(self.scale_holdover_vals)]) for key in self.init_vals}

        #we will start squared accumulated with large value for small steps
        self.squared_accumulated = {key: 1 for key in self.init_vals.keys()}

        #grad directions begins at zero, signifying a change in neither direction
        self.accumulated_directions = {key: 0 for key in self.init_vals.keys()}
        
        self.init_vals = init_vals
        self.fixed_vals = fixed_vals
        self.function = self.function(**init_vals)
        
        self.trained_values = copy.deepcopy(self.init_vals)
    
    def fit(self, train_data):

        start = time.time()
        self.build_inds_dict(train_data)
        self.log.info(f'created inds dict in {round((time.time() - start) / 60,4)} minutes')

        start = time.time()
        self.stoch_descent(train_data)
        self.log.info(f'trained function in {round((time.time() - start) / 60,4)} minutes')
        

    def build_inds_dict(self,
                        data):
        """ 
        if we are not balancing, we will always sample from the 0 dict
        """

        if self.balance_column is None:
            balances = np.zeros(len(data))
        else:
            balances = data[self.balance_column].tolist()

        if self.groupby_column is None:
            groups = list(range(len(data)))

        else:
            #ensure that groupby with multiple columns is a hashable type
            groups = [tuple(i) for i in data[self.groupby_column].to_numpy()]

        indices = list(range(len(data)))
    
        self.num_0 = 0
        self.num_1 = 0

        self.key_to_ind_0 = dict()
        self.key_to_ind_1 = dict()

        self.zeros_dict = dict()
        self.ones_dict = dict()

        for balance, group, index in zip(balances, groups, indices):

            #this will tell us which dictionary to go to
            #depending on whether we are balancing on some column or not
            if balance == 0:

                if group in self.key_to_ind_0.keys():
                    
                    #retrieve key from the conversion dict
                    num_key = self.key_to_ind_0[group]

                    #update the zeros dict with index
                    self.zeros_dict[num_key].append(index)
                
                else:

                    #add this key and unique value to the conversion dict
                    self.key_to_ind_0[group] = self.num_0

                    #add new key to zeros dict
                    self.zeros_dict[self.num_0] = [index]

                    #increment counter
                    self.num_0 += 1

            else:

                if group in self.key_to_ind_1.keys():
                    
                    #retrieve key from the conversion dict
                    num_key = self.key_to_ind_1[group]

                    #update the zeros dict with index
                    self.ones_dict[num_key].append(index)
                
                else:

                    #add this key and unique value to the conversion dict
                    self.key_to_ind_1[group] = self.num_1

                    #add new key to zeros dict
                    self.ones_dict[self.num_1] = [index]

                    #increment counter
                    self.num_1 += 1

        if self.balance_column is None:
            self.ones_dict = self.zeros_dict
            self.num_1 = self.num_0


    def get_match_rows(self):

        #flip balance flag
        self.balance_flag = abs(self.balance_flag - 1)

        if self.balance_flag == 0:

            return self.zeros_dict[np.random.randint(self.num_0)]

        else:

            return self.ones_dict[np.random.randint(self.num_1)]


    def stoch_descent(self, train_data):
        """ 
        Implement gradient descent for model tuning
        func must take: match, query, target
        """

        self.unmatched = 0
        self.top_wrong = 0
        for _ in range(int(self.max_iter)):

            index = self.get_match_rows()

            self.input_lengths.append(len(index))

            sub = train_data.iloc[index]
                                                           
            #update with the score of choice and funcOb's loss function
            score, pred_val = self.get_match_grad_components(sub)

            self.step(score, pred_val)    

            self.n_iter += 1
    
    def calculate_learning_rate(self, param, grad):

        if self.learning_rate_scheduler is None:

            return  self.learning_rates[param]

        elif self.learning_rate_scheduler == 'sag':

            #add scale contribution to holdover
            self.scale_holdovers[param].append(abs(grad))

            #use the most recent grad to get exp decaying net gradient
            self.accumulated_gradients[param] = self.learning_beta * self.accumulated_gradients[param] + (1 - self.learning_beta) * grad

            #get accumulated scale with the popped holdover value
            self.accumulated_scales[param] = self.accumulated_scales[param] * self.learning_beta + self.scale_holdovers[param].popleft() * (1 - self.learning_beta)

            #update the learning rate
            self.learning_rates[param] = self.learning_rates[param] * (self.ad_int + self.ad_slope * abs(self.accumulated_gradients[param]) / self.accumulated_scales[param])
            
            return max(1e-7, self.learning_rates[param])

    def step(self, score, pred_val):

        #convert gradient of f^ to gradient of loss func
        loss_grad = square_loss_grad(pred_val, score)
        
        for key, grad, bounds in zip(self.function.grad_names, self.function.grad_vals, self.bounds):

            #chain rule to get gradient w.r.t loss func
            #use np.dot in order to accomodate vector and float vals
            step = grad * loss_grad * self.learning_rate

            updated = getattr(self.function, key) - step

            #set updated value given constraints1
            setattr(self.function, key, min(max(bounds[0], updated), bounds[1]))


class tunaSimTrainer(funcTrainer):

    def __init__(self,
            name: str,
            init_vals: dict,
            fixed_vals: dict = None,
            learning_rate: List[float] = 0.01,
            max_iter: int = 1e5,
            bounds: dict = None,
            learning_rate_scheduler: str = None,
            learning_beta: float = 0.5,
            ad_int: float = 0.8,
            ad_slope: float  = 0.3,
            scale_holdover_vals: int = 2,
            groupby_column: str = None,
            balance_column: str = None):
        
        self.function = tunas.tunaSim
        
        super().__init__(
            name = name,
            init_vals = init_vals,
            fixed_vals = fixed_vals,
            learning_rate = learning_rate,
            max_iter = max_iter,
            bounds = bounds,
            learning_rate_scheduler = learning_rate_scheduler,
            learning_beta = learning_beta,
            ad_int = ad_int,
            ad_slope = ad_slope,
            scale_holdover_vals = scale_holdover_vals,
            groupby_column = groupby_column,
            balance_column = balance_column
        )

        self.bounds = [bounds[key] for key in self.init_vals]

    def get_match_grad_components(self, sub_df):

        #in the first round, we want to pick the index with the highest similarity scores
        if sub_df.shape[0] > 1:

            sims = self.function.predict_for_dataset(sub_df)
        
            #then, update gradients based on the best match for this grouping column value
            best_match_index = np.argmax(sims)

        else:
            best_match_index = 0

        return sub_df.iloc[best_match_index]['score'], self.function.predict(sub_df.iloc[best_match_index]['query'], 
                                                                          sub_df.iloc[best_match_index]['target'],
                                                                          grads = True)       


class tunaQueryTrainer(funcTrainer):

    def __init__(
            self,
            name: str,
            init_vals: dict,
            fixed_vals: dict = None,
            learning_rates: List[float] = 0.01,
            max_iter: int = 1e5,
            bounds: dict = None,
            learning_rate_scheduler: str = None,
            learning_beta: float = 0.5,
            ad_int: float = 0.8,
            ad_slope: float  = 0.3,
            scale_holdover_vals: int = 2,
            groupby_column: str = None,
            balance_column: str = None,
            identity_column: str = None,
            fixed: set = set()):
        
        self.identity_column = identity_column
        self.function = tunas.tunaTop

        super().__init__(name = name,
            init_vals = init_vals,
            fixed_vals = fixed_vals,
            learning_rates = learning_rates,
            max_iter = max_iter,
            bounds = bounds,
            learning_rate_scheduler = learning_rate_scheduler,
            learning_beta = learning_beta,
            ad_int = ad_int,
            ad_slope = ad_slope,
            scale_holdover_vals = scale_holdover_vals,
            groupby_column = groupby_column,
            balance_column = balance_column,
            fixed = fixed) 

    def get_match_grad_components(self, sub_df):
        """ 
        match the format of get match grad for similarities
        """
        
        #we only want the label of the top hit
        label = np.array(sub_df['score'] == True).astype(np.int64)[0] 

        #keep all scores to move top based on shape
        preds = np.array(sub_df['preds'], dtype = np.float64) 

        #top hit only for now
        return label, self.function.predict(preds, grads = True)[0] 