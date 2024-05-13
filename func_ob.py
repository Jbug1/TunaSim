
import numpy as np
from functools import partial
from scipy.optimize import minimize as mini
from scipy.optimize import approx_fprime as approx

def objective(x, args, loss_func, reg_func, distance_func):

    kwargs = {k:v for k,v in zip(args[0],x)}
    return np.sum(args[1].apply(lambda i: loss_func(i['match'] - distance_func(i['query'], i['target'],**kwargs)),axis=1) + reg_func(x))

class func_ob:
    def __init__(
            self,
            name,
            distance_func,
            regularization_func,
            loss_func,
            init_vals,
            params,
            constraints = None,
            solver = 'stoch',
            bounds = None,
            lambdas = 1,
            max_iter = 1e5,
            tol = 1e-10,
            momentum_weights = [0.8,0.2],
            epsilon = 1.4901161193847656e-08,
            momentum_type = None,
            rand = False
    ):
        self.name = name
        self.distance_func = distance_func
        self.regularization_func = regularization_func
        self.loss_func = loss_func
        self.init_vals = init_vals
        self.constraints=constraints
        self.params = params
        self.solver=solver
        self.bounds = bounds
        self.lambdas = lambdas
        self.max_iter = max_iter
        self.momentum_weights = momentum_weights
        self.epsilon = epsilon
        self.momentum_type = momentum_type
        self.rand=rand
        self.n_iter = 0
        self.tol=tol
        self.converged = None
        self.trained_vals = None
        self.running_grad = None
        self.converged = None
        self.trained_vals = None
        self.objective_value = None
        

    @property
    def objective_func(self):
        
        return partial(objective, 
                        loss_func = self.loss_func, 
                        reg_func = self.regularization_func, 
                        distance_func = self.distance_func)
    
    def fit(self, train_data, warm_start=False, verbose=None):

        if self.solver == 'stoch':

            self.stoch_descent(train_data, warm_start, verbose)

        else:
            self.scipy_solver_estimate(train_data, warm_start)
    
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


    def stoch_descent(self, train_data, warm_start=False, verbose=None):
        """ 
        Implement gradient descent for model tuning
        func must take: match, query, target
        """
        if warm_start:
            init_vals = self.trained_vals
        else:
            init_vals=self.init_vals

        if sum(self.momentum_weights)!=1:
            raise ValueError('sum of stop props must equal 1')
        
        if self.tol<0:
            raise ValueError('early stop must be geq 0')

        if type(self.lambdas)==float or type(self.lambdas) == int:
            self.lambdas = np.array([self.lambdas for _ in range(len(self.params))])

        if len(self.init_vals) != len(self.params) or len(self.params)!= len(self.lambdas):
            raise ValueError('all input vectors must have same first dimension')
        
        if self.constraints is not None and len(self.params) != len(self.constraints):
            raise ValueError('all input vectors must have same first dimension')

        if self.constraints is not None:
            mins = np.array([i[0] for i in self.bounds])
            maxs = np.array([i[1] for i in self.bounds])

        #set index at 0 and initial running grad so that we don't trigger early stop
        i=0
        running_grad = np.zeros(len(self.params))+10*(self.tol+1e-10)

        while i<self.max_iter and sum(np.abs(running_grad))>self.tol:

            #grab individual row
            if self.rand:
                index = np.random.randint(len(train_data))
            else:
                index=i
            
            #estimate gradient and update values
            if self.momentum_type is None:

                grad = approx(self.init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])
                init_vals -= self.lambdas * grad
                running_grad = self.momentum_weights[0] * running_grad + self.momentum_weights[1] * grad

            elif self.momentum_type == 'simple':

                grad = approx(init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])
                running_grad = self.momentum_weights[0] * running_grad + self.momentum_weights[1] * grad
                init_vals -= self.lambdas * running_grad

            elif self.momentum_type == 'jonie':

                init_vals -= self.lambdas * self.momentum_weights[0] * running_grad
                grad = approx(init_vals, self.objective_func, self.epsilon, [self.params, train_data.iloc[index:index+1]])
                init_vals -= self.lambdas * self.momentum_weights[1] * grad
                running_grad = self.momentum_weights[0] * running_grad + self.momentum_weights[1]* grad

            
            if self.bounds is not None:
                init_vals = np.clip(init_vals, mins, maxs)

            i+=1
            if verbose is not None:
                if i%verbose == 0:
                    print(f'completed {i} updates')

        #update object based on results
        self.n_iter += i
        self.converged = sum(running_grad)>self.tol
        self.running_grad = running_grad 
        self.trained_vals = init_vals

    def trained_func(self):
        if self.trained_vals is None:
            raise ValueError('functon has not been trained')

        else:
            kwargs = {k:v for k,v in zip(self.params,self.trained_vals)}
            return partial(self.distance_func,**kwargs)
        