
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from scipy import stats
from scipy.optimize import minimize as mini
from scipy.optimize import approx_fprime as approx
import TunaSims

def scipy_solver_estimate(objective_func, init_vals, var_names, data):


    return mini(fun=objective_func, x0=init_vals, args=[], tol=1e-10, method = 'L-BFGS-B')


def stoch_descent(obj_func, 
                 data, 
                 params, 
                 init_vals, 
                 constraints, 
                 loss_func, 
                 reg_func, 
                 distance_func,
                 lambdas = 1, 
                 max_iter = 1e6,
                 early_stop=0,
                 stop_props=[0.8,0.2],
                 epsilon = 1.4901161193847656e-08,
                momentum_type=None ):
    """ 
    Implement gradient descent for model tuning
    func must take: match, query, target
    """

    if sum(stop_props)!=1:
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

    #specify objective function
    objective_ = partial(obj_func, 
                         loss_func=loss_func, 
                         reg_func=reg_func, 
                         distance_func=distance_func,
                         )

    i=0
    running_grad = np.zeros(len(params))

    while i<max_iter and sum(running_grad)>early_stop:

        #grab individual row
        index = i % len(data)
        
        #estimate gradient and update values
        if momentum_type is None:

            grad = approx(init_vals, objective_, [params, data.iloc[index:index+1]])
            init_vals -= lambdas * grad
            running_grad = stop_props[0]*running_grad + stop_props[1]*np.abs(grad)

        elif momentum_type == 'simple':

            grad = approx(init_vals, objective_, epsilon, [params, data.iloc[index:index+1]])
            running_grad = stop_props[0]*running_grad + stop_props[1]*np.abs(grad)
            init_vals -= lambdas * running_grad

        elif momentum_type == 'jonie':

            init_vals -= lambdas*stop_props[0]*running_grad
            grad = approx(init_vals, objective_, epsilon, [params, data.iloc[index:index+1]])
            init_vals -= lambdas * stop_props[1]*grad
            running_grad = stop_props[0]*running_grad + stop_props[1]*np.abs(grad)

        
        if constraints is not None:
            init_vals = np.clip(init_vals, mins, maxs)

        i+=1

    return (sum(running_grad)>early_stop,init_vals)


stoch_descent(TunaSims.objective,
             df_ex,
             ['a','b'],
             [2,3],
             None,
             lambda x: x**2,
             lambda x: 0.1*sum(np.abs(x)),
             TunaSims.tuna_dif_distance,
             max_iter = 1e4,
             )