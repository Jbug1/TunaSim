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

        self.ones = 0
        self.zeros = 0

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

        self.train_data_shape = train_data.shape[0]

        self.train_data = train_data.sample(frac = 1)

        if self.balance_column is not None:

            self.balance_flag = 0

            if self.groupby_column is None:
                self.train_data.sort_values(by = self.balance_column, inplace = True)

            else:
                self.train_data.sort_values(by = [self.balance_column, self.groupby_column], inplace = True)

            counts = Counter(train_data['score'])
            if len(counts) != 2 or counts[0] < 1 or counts[1] < 1:
                raise ValueError("Can't balance this dataset")
            
            self.n_zeros = counts[0]
            self.n_ones = counts[1]

        self.stoch_descent(verbose)

        self.trained_vals = self.init_vals

    def get_index(self):

        if self.balance_column is not None:

            if self.balance_flag == 0:

                index = np.random.randint(self.n_zeros)

            else:

                index = self.n_zeros + np.random.randint(self.n_ones)

            #we want this to rotate between zero and 1
            self.balance_flag = abs(self.balance_flag - 1)

        else:

            index = np.random.randint(self.train_data_shape)

        return index

    def single_match_grad(self):

        index = self.get_index()

        if self.train_data.iloc[index]['score'] == 1:
            self.ones +=1
        else:
            self.zeros +=1

        #call predict method from Tuna Sim which updates gradients
        return self.train_data.iloc[index]['score'], self.sim_func.predict(self.train_data.iloc[index]['query'], 
                                                self.train_data.iloc[index]['target'], 
                                                self.train_data.iloc[index]['precquery'], 
                                                self.train_data.iloc[index]['prectarget'])
    
    def grouped_match_grad(self):

        index = self.get_index()

        #select only what we are interested in grouping
        sub = self.train_data[self.train_data[self.groupby_column] == self.train_data.iloc[index][self.groupby_column]]

        #in the first round, we want to pick the index with the highest similarity scores
        sims = sub.apply(lambda x: self.sim_func.predict(x['query'], x['target'], grads = False), 
                  axis = 1, 
                  result_type = 'expand')
        
        
        #then, update gradients based on the best match for this grouping column value
        best_match_index = np.argmax(sims)

        return sub.iloc[best_match_index]['score'], self.sim_func.predict(sub.iloc[best_match_index]['query'], 
                                                                          sub.iloc[best_match_index]['target'],
                                                                          grads = True)

    def stoch_descent(self, verbose = None):
        """ 
        Implement gradient descent for model tuning
        func must take: match, query, target
        """

        for _ in range(int(self.max_iter)):

            if self.groupby_column is None:

                score, pred_val = self.single_match_grad()

            else:
                score, pred_val = self.grouped_match_grad()
            
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
    
    def step(self, score, pred_val, verbose = False):

        #convert gradient of f^ to gradient of loss func
        loss_grad = self.loss_grad(pred_val, score)
        print(f'{loss_grad=}')
    
        if np.isnan(loss_grad):
            raise ValueError("loss grad is nan")
        
        for key in self.init_vals:

            #chain rule to get gradient w.r.t loss func
            #use np.dot in order to accomodate vector and float vals
            step = np.dot(self.sim_func.grads1[key], loss_grad)
                
            #adjust lambda according to scheduler
            learning_rate = self.calculate_learning_rate(key, step)

            step = learning_rate * step
            updated = getattr(self.sim_func, key) - step

            if verbose and self.n_iter % 10 == 0:
                print(f"{self.n_iter=}, {key=}, {self.accumulated_gradients[key]}, {self.accumulated_scales[key]}, {current_value=}, {updated=}, {learning_rate=}")

            if key in self.bounds:
                bounds = self.bounds[key]
                setattr(self.sim_func, key, min(max(bounds[0], updated), bounds[1]))

            else:
                setattr(self.sim_func, key, updated)

            if np.isnan(updated):
                raise ValueError('updated value is Nan')


    def trained_func(self):
        if self.trained_vals is None:
            raise ValueError('function has not been trained')

        else:
            return partial(self.sim_func,**self.trained_vals)

class specSimTrainer(funcTrainer):

    def __init__(self,
            name: str,
            init_vals: dict,
            fixed_vals: dict = None,
            loss_grad: Callable = lambda x,y: 2 * abs(x - y),
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
        

    def loss_grad(self, pred_value, score):

        label_ind = np.where(pred_value[1] == score)

        output = np.zeros(len(pred_value[0]))
        output[label_ind] = 1 / pred_value[0][label_ind]

        return output