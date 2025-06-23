
import numpy as np
from functools import partial
from typing import Callable, List
import copy
from collections import Counter
import pandas as pd


class func_ob:
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
            sim_func: Callable,
            init_vals: dict,
            fixed_vals: dict = None,
            regularization_grad: Callable = lambda x: 0,
            loss_grad: Callable = lambda x: 2 * x,
            regularization_name: str = '',
            loss_name: str = 'l2',
            solver: str = 'stoch',
            learning_rates: List[float] = 0.01,
            max_iter: int = 1e5,
            momentum_beta: float = 0.8,
            momentum_type: str = None,
            running_grad_start: float = 1e5,
            bounds: dict = None,
            tol: float = 0.0,
            balance_classes: bool = True,
            learning_rate_scheduler: str = None,
            learning_beta: float = 0.5,
            groupby_column: str = None,
    ):
        self.name = name
        self.sim_func = sim_func
        self.regularization_name = regularization_name
        self.regularization_grad = regularization_grad
        self.loss_grad = loss_grad
        self.loss_name = loss_name
        self.init_vals = init_vals
        self.solver = solver
        self.bounds = bounds
        self.max_iter = int(max_iter)
        self.momentum_beta = momentum_beta
        self.momentum_type = momentum_type
        self.running_grad_start = running_grad_start
        self.n_iter = 0
        self.tol = tol
        self.balance_classes = balance_classes
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_beta = learning_beta
        self.groupby_column = groupby_column

        self.grad = None
        self.converged = None
        self.running_grad = None
        self.trained_vals = None
        self.objective_value = None

        self.ones = 0
        self.zeros = 0

        #set accumulated gradients dictionary in case we are implementing momentum
        self.accumulated_gradient = {key: 0 for key in self.init_vals.keys()}

        if type(learning_rates) == float or type(learning_rates) == int:
            self.learning_rates = dict()
            for key in self.init_vals:
                self.learning_rates[key] = learning_rates

        else:
            self.learning_rates = learning_rates

        #we will start squared accumulated with large value for small steps
        self.squared_accumulated = {key: 1 for key in self.init_vals.keys()}

        #grad directions begins at zero, signifying a change in neither direction
        self.grad_directions = {key: 1 for key in self.init_vals.keys()}

        if len(self.learning_rates) != len(self.init_vals):
            raise ValueError('lambda and init vals len must match')
        
        self.init_vals = init_vals
        self.fixed_vals = fixed_vals
        inits = dict()
        inits.update(fixed_vals)
        inits.update(init_vals)
        self.sim_func = self.sim_func(**inits)
        
        if self.tol<0:
            raise ValueError('early stop must be geq 0')
        
        self.trained_values = copy.deepcopy(self.init_vals)
    
    def fit(self, train_data, verbose=None):

        self.train_data_shape = train_data.shape[0]

        self.converged = False
        self.running_grad = self.running_grad_start

        if self.balance_classes:

            self.train_data = train_data.sample(frac = 1)

            if self.groupby_column is None:
                self.train_data.sort_values(by = 'score', inplace = True)

            else:
                self.train_data.sort_values(by = ['score', self.groupby_column], inplace = True)

            counts = Counter(train_data['score'])
            if len(counts) != 2 or counts[0] < 1 or counts[1] < 1:
                raise ValueError("Can't balance this dataset")
            
            self.n_zeros = counts[0]
            self.n_ones = counts[1]

        if self.solver == 'stoch':

            self.stoch_descent(verbose)

        else:
            self.scipy_solver_estimate(train_data)

        self.trained_vals = self.init_vals

    def get_index(self):

        if self.balance_classes:

            if np.random.binomial(1, 0.5) == 0:

                index = np.random.randint(self.n_zeros)

            else:

                index = self.n_zeros + np.random.randint(self.n_ones)

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

        if self.train_data.iloc[index]['score'] == 1:
            self.ones +=1
        else:
            self.zeros +=1

        #select only what we are interested in grouping
        sub = self.train_data[self.train_data[self.groupby_column] == self.train_data.iloc[index][self.groupby_column]]

        if len(set(sub['score'])) > 1:
            print('argh')

        #in the first round, we want to pick the index with the highest similarity scores
        sims = sub.apply(lambda x: self.sim_func.predict(x['query'], x['target'], x['precquery'], x['prectarget'], grads = False), 
                  axis = 1, 
                  result_type = 'expand')
        
        
        #then, update gradients based on the best match for this grouping column value
        best_match_index = np.argmax(sims)

        return sub.iloc[best_match_index]['score'], self.sim_func.predict(sub.iloc[best_match_index]['query'], 
                                                sub.iloc[best_match_index]['target'], 
                                                sub.iloc[best_match_index]['precquery'], 
                                                sub.iloc[best_match_index]['prectarget'])



    def stoch_descent(self, verbose = None):
        """ 
        Implement gradient descent for model tuning
        func must take: match, query, target
        """

        self.running_grad = self.running_grad_start

        for _ in range(int(self.max_iter)):

            if self.converged:
                break

            if self.groupby_column is None:

                score, pred_val = self.single_match_grad()

            else:
                score, pred_val = self.grouped_match_grad()
            
            #update with the score of choice and funcOb's loss function
            self.step(score, pred_val)    

            #update object based on results
            self.converged = self.running_grad < self.tol

            if verbose is not None:

                if (_ + 1) % verbose == 0:
                    print(f'completed {_ + 1} iterations')

        self.n_iter += _

    def calculate_unweighted_step(self, grad, param):

        if self.momentum_type is None:

                step = grad
    
        elif self.momentum_type == 'simple':

            #first calculate the new step which takes accumulated grad into consideration
            step = (self.momentum_beta * self.accumulated_gradient[param]) + (1 - self.momentum_beta) * grad
            
            #update accumulated_gradient
            self.accumulated_gradient[param] = step
            
        elif self.momentum_type == 'nag':

            adjustment = (1 - self.momentum_beta) * (grad)
            overshoot = (self.momentum_beta * self.accumulated_gradient[param]) + adjustment

            #by adjusting the grad for beta again, we can get the next round step in
            # the end result will be wrong but we could always just take the overshoot back out...who cares
            step = adjustment + self.momentum_beta * overshoot

            self.accumulated_gradient[param] = overshoot

        return step
    
    def calculate_learning_rate(self, param, grad):

        if self.learning_rate_scheduler is None:

            learning_rate = self.learning_rates[param]
        
        elif self.learning_rate_scheduler == 'rms':

            self.squared_accumulated[param] = self.learning_beta * self.squared_accumulated[param] + (1 - self.learning_beta) * grad ** 2
            learning_rate = self.learning_rates[param] / (1e-7 + np.sqrt(self.squared_accumulated[param]))

        elif self.learning_rate_scheduler == 'ad':

            self.grad_directions[param] = self.learning_beta * self.grad_directions[param] + (1 - self.learning_beta) * int(grad > 0)
            learning_rate = self.learning_rates[param]  * abs(self.grad_directions[param])

        elif self.learning_rate_scheduler == 'ad_mult':

            self.grad_directions[param] = self.learning_beta * self.grad_directions[param] + (1 - self.learning_beta) * int(grad > 0)
            self.learning_rates[param] = self.learning_rates[param] * 2 * abs(self.grad_directions[param])
            learning_rate = self.learning_rates[param]

        #utilize both AD and RMS
        elif self.learning_rate_scheduler == 'hybrid':

            self.squared_accumulated[param] = self.learning_beta * self.squared_accumulated[param] + (1 - self.learning_beta) * grad ** 2
            self.grad_directions[param] = self.learning_beta * self.grad_directions[param] + (1 - self.learning_beta) * int(grad > 0)
            learning_rate = self.learning_rates[param]  * abs(self.grad_directions[param]) / (1e-7 + np.sqrt(self.squared_accumulated[param]))

        return learning_rate
    
    def step(self, score, pred_val, verbose = True):
            
        #collector for running grad across all variables
        running_grad_temp = 0

        #convert gradient of f^ to gradient of loss func
        loss_grad = self.loss_grad(pred_val - score)

        #print(f"{pred_val=}, {loss_grad=}, {score=}")
    
        if np.isnan(loss_grad):
            raise ValueError("loss grad is nan")
        
        for key in self.init_vals:

            #chain rule to get gradient w.r.t loss func
            grad = self.sim_func.grads1_score_agg[key] * loss_grad

            #update running gradient
            running_grad_temp += abs(grad)

            current_value = getattr(self.sim_func, key)

            #add gradient w.r.t. regualrization function
            grad += self.regularization_grad(current_value)

            #calculate direction and mag of unweighted step
            unweighted_step = self.calculate_unweighted_step(grad, key)
                
            #adjust lambda according to scheduler
            learning_rate = self.calculate_learning_rate(key, grad)

            step = learning_rate * unweighted_step
            updated = current_value - step

            if verbose:
                print(f"{key=}, {current_value=}, {updated=}, {learning_rate=}, {unweighted_step=}, {grad=}, {step=}")

            if key in self.bounds:
                bounds = self.bounds[key]
                setattr(self.sim_func, key, min(max(bounds[0], updated), bounds[1]))

            else:
                setattr(self.sim_func, key, updated)

            if np.isnan(updated):
                print(key, current_value, updated)
                raise ValueError('updated value is Nan')

        self.running_grad = self.momentum_beta * self.running_grad + (1 - self.momentum_beta) * running_grad_temp/len(self.init_vals)

    def trained_func(self):
        if self.trained_vals is None:
            raise ValueError('function has not been trained')

        else:
            return partial(self.sim_func,**self.trained_vals)
        