
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import minimize as mini
from scipy.optimize import approx_fprime as approx

def scipy_solver_estimate(objective_func, data, init_vals, var_names, bounds=None, method='L-BFGS-B'):


    return mini(fun=objective_func, x0=init_vals, args=[var_names, data], tol=1e-20, bounds = bounds, method = method)


def stoch_descent(objective_func, 
                 data,
                 init_vals, 
                 params,  
                 constraints = None, 
                 lambdas = 1, 
                 max_iter = 1e6,
                 early_stop_thresh = 1e-5,
                 momentum_weights=[0.8, 0.2],
                 epsilon = 1.4901161193847656e-08,
                momentum_type=None ):
    """ 
    Implement gradient descent for model tuning
    func must take: match, query, target
    """

    if sum(momentum_weights)!=1:
        raise ValueError('sum of stop props must equal 1')
    
    if early_stop<0:
        raise ValueError('early stop must be geq 0')

    if type(lambdas)==float or type(lambdas) == int:
        lambdas = np.array([lambdas for i in range(len(params))])

    if len(init_vals) != len(params) or len(params)!= len(lambdas):
        raise ValueError('all input vectors must have same first dimension')
    
    if constraints is not None and len(params) != len(constraints):
        raise ValueError('all input vectors must have same first dimension')

    if constraints is not None:
        mins = np.array([i[0] for i in constraints])
        maxs = np.array([i[1] for i in constraints])

    #set index at 0 and initial running grad so that we don't trigger early stop
    i=0
    running_grad = np.zeros(len(params))+10*(early_stop_thresh+1e-10)

    while i<max_iter and sum(np.abs(running_grad))>early_stop_thresh:

        #grab individual row
        index = i % len(data)
        
        #estimate gradient and update values
        if momentum_type is None:

            grad = approx(init_vals, objective_func, epsilon, [params, data.iloc[index:index+1]])
            init_vals -= lambdas * grad
            running_grad = momentum_weights[0] * running_grad + momentum_weights[1] * grad

        elif momentum_type == 'simple':

            grad = approx(init_vals, objective_func, epsilon, [params, data.iloc[index:index+1]])
            running_grad = momentum_weights[0] * running_grad + momentum_weights[1] * grad
            init_vals -= lambdas * running_grad

        elif momentum_type == 'jonie':

            init_vals -= lambdas*momentum_weights[0]*running_grad
            grad = approx(init_vals, objective_func, epsilon, [params, data.iloc[index:index+1]])
            init_vals -= lambdas * momentum_weights[1] * grad
            running_grad = momentum_weights[0] * running_grad + momentum_weights[1]* grad

        
        if constraints is not None:
            init_vals = np.clip(init_vals, mins, maxs)

        i+=1

    #return number of iterations, whether we stopped early, and the final values
    return (i, sum(running_grad)>early_stop, running_grad, init_vals)