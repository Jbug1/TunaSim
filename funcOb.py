
import numpy as np
from functools import partial
from typing import Callable, List
from scipy.optimize import minimize as mini
from scipy.optimize import approx_fprime as approx


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
            init_vals: List[float] = 0.1,
            params: List[str] = None,
            param_grad: List[Callable] = None,
            param_grad2: List[Callable] = None,
            regularization_func: Callable = lambda x: 0,
            regularization_grad: Callable = None,
            loss_func: Callable = lambda x: x**2,
            regularization_name: str = '',
            loss_name: str = 'l2',
            solver: str = 'stoch',
            lambdas: List[float] = 1,
            max_iter: int = 1e5,
            tol: float = 1e-10,
            momentum_weights: List[float] = [0.8,0.2],
            momentum_type: str = 'None',
            running_grad_start: float = 1e5,
            rand: bool = False,
            scheduler: Callable = None,
            bounds: dict = {
                "query_da_thresh":[0,np.inf],
                "target_da_thresh":[0,np.inf],
                "match_tolerance":[0,np.inf]
            }
    ):
        self.name = name
        self.sim_func = sim_func
        self.regularization_name = regularization_name
        self.regularization_func = regularization_func
        self.regularization_grad = regularization_grad
        self.loss_func = loss_func
        self.loss_name = loss_name
        self.init_vals = init_vals
        self.params = params
        self.param_grad = param_grad
        self.param_grad2 = param_grad2
        self.solver = solver
        self.bounds = bounds
        self.lambdas = lambdas
        self.max_iter = max_iter
        self.momentum_weights = momentum_weights
        self.momentum_type = momentum_type
        self.running_grad_start = running_grad_start
        self.rand=rand
        self.scheduler = scheduler
        self.n_iter = 0
        self.tol=tol

        self.grad = None
        self.converged = None
        self.running_grad = None
        self.converged = None
        self.trained_vals = None
        self.objective_value = None
        

    @property
    def objective_func(self):
        
        return partial(self.objective, 
                        loss_func = self.loss_func, 
                        reg_func = self.regularization_func, 
                        sim_func = self.sim_func)
    
    def set_array_params(self):

        if type(self.init_vals) == float:
            self.init_vals = np.array([self.init_vals for i in range(len(self.params))])
            self.init_vals_ = self.init_vals

        else:
            self.init_vals = np.array(self.init_vals)
            self.init_vals_ = np.array(self.init_vals)

        if type(self.lambdas) == float or type(self.lambdas) == int:
            self.lambdas = np.array([self.lambdas for i in range(len(self.params))])

        else:
            self.lambdas = np.array(self.lambdas)

        if type(self.epsilon) == float or type(self.epsilon) == int:
            self.epsilon = np.array([self.epsilon for i in range(len(self.params))])

        else:
            self.epsilon = np.array(self.epsilon)

        #set arrays for bounds
        self.min_bounds = np.array([-np.inf for i in range(len(self.params))])
        self.max_bounds = np.array([np.inf for i in range(len(self.params))])

        for key, value in self.bounds.items():
            ind = np.where(self.params == key)
            self.min_bounds[ind] = value[0]
            self.max_bounds[ind] = value[1]
    
    def fit(self, train_data, verbose=None):

        self.set_array_params()

        if self.solver == 'stoch':

            self.stoch_descent(train_data, verbose)

        else:
            self.scipy_solver_estimate(train_data)
    
    def scipy_solver_estimate(self, train_data, warm_start=False):

        if warm_start:
            init_vals = self.trained_vals
        else:
            init_vals=self.init_vals

        scipy_res = mini(fun=self.objective_func, x0=init_vals, args=[self.params, train_data], tol=self.tol, bounds = self.bounds, method = self.solver)

        #update values based on results
        self.converged = scipy_res.success
        self.trained_vals = scipy_res.x
        self.n_iter += scipy_res.nfev
        self.objective_value = scipy_res.fun

    def stoch_descent(self, train_data, verbose=None):
        """ 
        Implement gradient descent for model tuning
        func must take: match, query, target
        """

        if sum(self.momentum_weights)!=1:
            raise ValueError('sum of stop props must equal 1')
        
        if self.tol<0:
            raise ValueError('early stop must be geq 0')

        if len(self.init_vals) != len(self.params) or len(self.params)!= len(self.lambdas):
            raise ValueError('all input vectors must have same first dimension')

        #set index at 0 and initial running grad so that we don't trigger early stop
        i=0
        self.running_grad = np.zeros(len(self.params)) + (self.running_grad_start/len(self.params))

        while i<self.max_iter and sum(np.abs(self.running_grad))>self.tol:

            #grab individual row
            if self.rand:
                index = np.random.randint(train_data.shape[0])
            else:
                index=i%train_data.shape[0]
            
            #estimate gradient and update values
            if self.momentum_type == 'None':
                
                self.grad = approx(self.init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])

                if np.any(np.isnan(self.grad)) or np.any(np.isinf(self.grad)):
                    print('bad grad')
                    i+=1
                    continue

                self.init_vals -= self.lambdas * self.grad
                
                self.running_grad = self.momentum_weights[0] * self.running_grad + self.momentum_weights[1] * self.grad

            elif self.momentum_type == 'simple':

                self.grad = approx(self.init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])
                if np.any(np.isnan(self.grad)) or np.any(np.isinf(self.grad)):
                    print('bad grad')
                    i+=1
                    continue

                self.running_grad = self.momentum_weights[0] * self.running_grad + self.momentum_weights[1] * self.grad
                self.init_vals -= self.lambdas * self.running_grad

            elif self.momentum_type == 'jonie':

                self.init_vals -= self.lambdas * self.momentum_weights[0] * self.running_grad
                self.grad = approx(self.init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])
                if np.any(np.isnan(self.grad)) or np.any(np.isinf(self.grad)):
                    print('bad grad')
                    i+=1
                    continue
                
                self.init_vals -= self.lambdas * self.momentum_weights[1] * self.grad
                self.running_grad = self.momentum_weights[0] * self.running_grad + self.momentum_weights[1]* self.grad
            
            
            self.init_vals = np.clip(self.init_vals, self.min_bounds, self.max_bounds)

            i+=1
            if verbose is not None:
                if i%verbose == 0:
                    print(f'completed {i} updates')

            #update to epsilon values
            if self.zero_grad_epsilon_boost != 0:

                zero_inds = np.where(self.grad == 0)
                self.epsilon[zero_inds] *= self.zero_grad_epsilon_boost
                self.epsilon=np.clip(self.epsilon,0,10)

            if self.lambda_schedule is not None:

                self.lambdas = self.lambda_schedule(self.lambdas)

        #update object based on results
        self.n_iter += i
        self.converged = sum(self.running_grad)<self.tol
        self.running_grad = self.running_grad 
        self.trained_vals = self.init_vals

    def trained_func(self):
        if self.trained_vals is None:
            raise ValueError('functon has not been trained')

        else:
            kwargs = {k:v for k,v in zip(self.params,self.trained_vals)}
            return partial(self.sim_func,**kwargs)
        