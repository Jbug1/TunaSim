
import numpy as np
from functools import partial
from typing import Callable, List
from scipy.optimize import minimize as mini
from scipy.optimize import approx_fprime as approx
import copy


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
            regularization_func: Callable = lambda x: 0,
            regularization_grad: Callable = None,
            loss_func: Callable = lambda x: x**2,
            loss_grad: Callable = lambda x: -2 * x,
            regularization_name: str = '',
            loss_name: str = 'l2',
            solver: str = 'stoch',
            lambdas: List[float] = 0.01,
            max_iter: int = 1e5,
            momentum_weights: List[float] = [0.8,0.2],
            momentum_type: str = 'None',
            running_grad_start: float = 1e5,
            rand: bool = False,
            scheduler: Callable = None,
            bounds: dict = None,
            tol: float = 0.0
    ):
        self.name = name
        self.sim_func = sim_func
        self.regularization_name = regularization_name
        self.regularization_func = regularization_func
        self.regularization_grad = regularization_grad
        self.loss_func = loss_func
        self.loss_grad = loss_grad
        self.loss_name = loss_name
        self.init_vals = init_vals
        self.solver = solver
        self.bounds = bounds
        self.max_iter = max_iter
        self.momentum_weights = momentum_weights
        self.momentum_type = momentum_type
        self.running_grad_start = running_grad_start
        self.rand=rand
        self.scheduler = scheduler
        self.n_iter = 0
        self.tol = tol

        self.grad = None
        self.converged = None
        self.running_grad = None
        self.converged = None
        self.trained_vals = None
        self.objective_value = None

        if type(lambdas) == float or type(lambdas) == int:
            self.lambdas = dict()
            for key in self.init_vals:
                self.lambdas[key] = lambdas

        else:
            self.lambdas = lambdas

        if len(self.lambdas) != len(self.init_vals):
            raise ValueError('lambda and init vals len must match')
        
        self.init_vals = init_vals
        self.fixed_vals = fixed_vals
        inits = dict()
        inits.update(fixed_vals)
        inits.update(init_vals)
        self.sim_func = self.sim_func(**inits)

        if sum(self.momentum_weights)!=1:
            raise ValueError('sum of stop props must equal 1')
        
        if self.tol<0:
            raise ValueError('early stop must be geq 0')
        
        self.trained_values = copy.deepcopy(self.init_vals)
        

    # @property
    # def objective_func(self):
        
    #     return partial(self.objective, 
    #                     loss_func = self.loss_func, 
    #                     reg_func = self.regularization_func, 
    #                     sim_func = self.sim_func)
    
    
    def fit(self, train_data, verbose=None):

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

    def stoch_descent(self, train_data, verbose = None):
        """ 
        Implement gradient descent for model tuning
        func must take: match, query, target
        """

        #set index at 0 and initial running grad so that we don't trigger early stop
        i=0
        self.running_grad = self.running_grad_start

        while i < self.max_iter and self.running_grad > self.tol:

            #grab individual row
            if self.rand:
                index = np.random.randint(train_data.shape[0])
            else:
                index = i % train_data.shape[0]

            #call predict method from Tuna Sim which updates gradients
            pred_val = self.sim_func.predict(train_data.iloc[index]['query'], train_data.iloc[index]['target'])

            print(pred_val)

            #update with the score of choice and funcOb's loss function
            self.step(train_data.iloc[i]['score'], pred_val)

            #update object based on results
            self.n_iter += i
            self.converged = self.running_grad < self.tol

            if verbose is not None:

                if self.n_iter % verbose == 0:
                    print(f'completed {self.n_iter} iterations')

    def step(self, score, pred_val):
            
        running_grad_temp = 0
        for key in self.init_vals:

            value = self.sim_func.grads1_score_agg[key]

            print(key, value)

            running_grad_temp += abs(value)

            lambda_ = self.lambdas[key]
            current = getattr(self.sim_func, key)
            loss_grad = self.loss_grad(score - pred_val)
            print(f'loss grad: {loss_grad}')
            print(f'step {lambda_ * loss_grad * value}')

            if self.momentum_type == 'None':
                setattr(self.sim_func, key, current - lambda_ * loss_grad * value)
    
            elif self.momentum_type == 'simple':

                self.grad = approx(self.init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])
                if np.any(np.isnan(self.grad)) or np.any(np.isinf(self.grad)):
                    print('bad grad')
                    i+=1

                self.running_grad = self.momentum_weights[0] * self.running_grad + self.momentum_weights[1] * self.grad
                self.init_vals -= self.lambdas * self.running_grad

            elif self.momentum_type == 'jonie':

                self.init_vals -= self.lambdas * self.momentum_weights[0] * self.running_grad
                self.grad = approx(self.init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])
                if np.any(np.isnan(self.grad)) or np.any(np.isinf(self.grad)):
                    print('bad grad')
                    i+=1
                
                self.init_vals -= self.lambdas * self.momentum_weights[1] * self.grad
                self.running_grad = self.momentum_weights[0] * self.running_grad + self.momentum_weights[1]* self.grad

        self.running_grad = self.momentum_weights[0] * self.running_grad + self.momentum_weights[1] * running_grad_temp

    def trained_func(self):
        if self.trained_vals is None:
            raise ValueError('function has not been trained')

        else:
            kwargs = {k:v for k,v in zip(self.params,self.trained_vals)}
            return partial(self.sim_func,**kwargs)
        