import TunaSims
import numpy as np
from functools import partial
from typing import Callable, List
import copy
from collections import Counter
from collections import deque


class funcTrainer:
    ''' 
    name: what to call this func ob 
    sim_func: tuna sim function that maps input spectra to 0-1 interval
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
            loss_grad: Callable = lambda x,y: 2 * (x - y),
            learning_rates: List[float] = 0.01,
            max_iter: int = 1e5,
            bounds: dict = None,
            learning_rate_scheduler: str = None,
            learning_beta: float = 0.5,
            ad_int: float = 0.8,
            ad_slope: float  = 0.3,
            scale_holdover_vals: int = 2,
            groupby_column: str = None,
            balance_column: str = None
    ):
        self.name = name
        self.loss_grad = loss_grad
        self.init_vals = init_vals
        self.bounds = bounds
        self.max_iter = int(max_iter)
        self.n_iter = 0
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_beta = learning_beta
        self.ad_int = ad_int
        self.ad_slope = ad_slope
        self.scale_holdover_vals = scale_holdover_vals
        self.groupby_column = groupby_column
        self.balance_column = balance_column

        self.balance_flag = 0

        #set scheduling dictionaries
        self.accumulated_gradients = {key: 0 for key in self.init_vals}
        self.accumulated_scales = {key: 0 for key in self.init_vals}
        self.scale_holdovers = {key: deque([1 for i in range(self.scale_holdover_vals)]) for key in self.init_vals}


        if type(learning_rates) == float or type(learning_rates) == int:
            self.learning_rates = dict()
            for key in self.init_vals:
                self.learning_rates[key] = learning_rates

        else:
            self.learning_rates = learning_rates

        #we will start squared accumulated with large value for small steps
        self.squared_accumulated = {key: 1 for key in self.init_vals.keys()}

        #grad directions begins at zero, signifying a change in neither direction
        self.accumulated_directions = {key: 0 for key in self.init_vals.keys()}

        if len(self.learning_rates) != len(self.init_vals):
            raise ValueError('lambda and init vals len must match')
        
        self.init_vals = init_vals
        self.fixed_vals = fixed_vals
        inits = dict()
        inits.update(fixed_vals)
        inits.update(init_vals)
        self.sim_func = self.sim_func(**inits)
        
        self.trained_values = copy.deepcopy(self.init_vals)
    
    def fit(self, train_data, verbose=None):

        train_data = train_data.sample(frac = 1)

        self.build_inds_dict(train_data)

        self.stoch_descent(train_data, verbose)

    def build_inds_dict(self, data):
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
            groups = data[self.groupby_column].tolist()

        indices = list(range(len(data)))
    
        self.num_0 = 0
        self.num_1 = 0

        key_to_ind_0 = dict()
        key_to_ind_1 = dict()

        self.zeros_dict = dict()
        self.ones_dict = dict()

        for balance, group, index in zip(balances, groups, indices):

            #this will tell us which dictionary to go to
            #depending on whether we are balancing on some column or not
            if balance == 0:

                if group in key_to_ind_0.keys():
                    
                    #retrieve key from the conversion dict
                    num_key = key_to_ind_0[group]

                    #update the zeros dict with index
                    self.zeros_dict[num_key].append(index)
                
                else:

                    #add this key and unique value to the conversion dict
                    key_to_ind_0[group] = self.num_0

                    #add new key to zeros dict
                    self.zeros_dict[self.num_0] = [index]

                    #increment counter
                    self.num_0 += 1

            else:

                if group in key_to_ind_1.keys():
                    
                    #retrieve key from the conversion dict
                    num_key = key_to_ind_1[group]

                    #update the zeros dict with index
                    self.ones_dict[num_key].append(index)
                
                else:

                    #add this key and unique value to the conversion dict
                    key_to_ind_1[group] = self.num_1

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


    def stoch_descent(self, train_data, verbose = None):
        """ 
        Implement gradient descent for model tuning
        func must take: match, query, target
        """

        for _ in range(int(self.max_iter)):

            index = self.get_match_rows()

            #select only what we are interested in grouping and retrieve grads
            score, pred_val = self.get_match_grad(train_data.iloc[index])
            
            #update with the score of choice and funcOb's loss function
            self.step(score, pred_val)    

            if (_ + 1) % (verbose or 1e12) == 0:
                print(f'completed {_ + 1} iterations')

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
        loss_grad = self.loss_grad(pred_val, score)
    
        if np.isnan(loss_grad):
            raise ValueError("loss grad is nan")
        
        for key, value, bounds in zip(self.sim_func.grad_names, self.sim_func.grad_vals, self.bounds):

            if key not in self.init_vals:
                continue

            #chain rule to get gradient w.r.t loss func
            #use np.dot in order to accomodate vector and float vals
            step = np.dot(value, loss_grad)
                
            #adjust lambda according to scheduler
            learning_rate = self.calculate_learning_rate(key, step)

            step = learning_rate * step
            updated = getattr(self.sim_func, key) - step

            #set updated value given constraints
            setattr(self.sim_func, key, min(max(bounds[0], updated), bounds[1]))

            if np.isnan(updated):
                raise ValueError(f'updated value is Nan for {key, value}')

class specSimTrainer(funcTrainer):

    def __init__(self,
            name: str,
            init_vals: dict,
            fixed_vals: dict = None,
            loss_grad: Callable = lambda x,y: 2 * (x - y),
            learning_rates: List[float] = 0.01,
            max_iter: int = 1e5,
            bounds: dict = None,
            learning_rate_scheduler: str = None,
            learning_beta: float = 0.5,
            ad_int: float = 0.8,
            ad_slope: float  = 0.3,
            scale_holdover_vals: int = 2,
            groupby_column: str = None,
            balance_column: str = None):
        
        self.sim_func = TunaSims.speedyTuna
        
        super().__init__(
            name = name,
            init_vals = init_vals,
            fixed_vals = fixed_vals,
            loss_grad = loss_grad,
            learning_rates = learning_rates,
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

        self.bounds = [self.bounds[key] for key in self.sim_func.grad_names]
        

class scoreByQueryTrainer(funcTrainer):

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
    ):
        
        self.sim_func = TunaSims.ScoreByQuery

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
            groupby_column = groupby_column) 
        

    def get_match_grad(self, sub_df):

        #in the first round, we want to pick the index with the highest similarity scores
        if sub_df.shape[0] > 1:

            sims = sub_df.apply(lambda x: self.sim_func.predict(x['query'], x['target'], grads = False), 
                    axis = 1, 
                    result_type = 'expand')
        
        
            #then, update gradients based on the best match for this grouping column value
            best_match_index = np.argmax(sims)

        else:
            best_match_index = 0

        return sub_df.iloc[best_match_index]['score'], self.sim_func.predict(sub_df.iloc[best_match_index]['query'], 
                                                                          sub_df.iloc[best_match_index]['target'],
                                                                          grads = True) 
        

    def loss_grad(self, pred_value, score):

        label_ind = np.where(pred_value[1] == score)

        output = np.zeros(len(pred_value[0]))
        output[label_ind] = 1 / pred_value[0][label_ind]

        return output