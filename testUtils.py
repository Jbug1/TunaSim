import TunaSims
import func_ob
import spectral_similarity

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc


def create_all_funcs_stoch(reg_funcs,
                    reg_names, 
                    losses, 
                    loss_names, 
                    momentums, 
                    params, 
                    mom_weights, 
                    lambdas=[1], 
                    init_vals=None,
                    max_iters=[1e5]):
    
    funcs=list()

    for  i in range(len(reg_funcs)):
        for j in range(len(losses)):
            for momentum in momentums:
                for name_, params_ in params.items():
                    if momentum is not None:
                        for mom_weight in mom_weights:
                            for lam in lambdas:
                                for max_iter in max_iters:
                                
                                    if init_vals is None:
                                        funcs.append(func_ob.func_ob(name=f'{name_}_{reg_names[i]}_{loss_names[j]}_{momentum}_{mom_weight}',
                                                            distance_func = TunaSims.tuna_dif_distance,
                                                            regularization_func = reg_funcs[i],
                                                            loss_func = losses[j],
                                                            init_vals = np.ones(len(params_))/10,
                                                            params = params_,
                                                            lambdas = lam,
                                                            solver = 'stoch',
                                                            max_iter = max_iter,
                                                            momentum_weights = mom_weight,
                                                            momentum_type = momentum
                                        ))

                                    else:
                                        for init_val in init_vals:
                                            funcs.append(func_ob.func_ob(name=f'{name_}_{reg_names[i]}_{loss_names[j]}_{momentum}_{mom_weight}',
                                                            distance_func = TunaSims.tuna_dif_distance,
                                                            regularization_func = reg_funcs[i],
                                                            loss_func = losses[j],
                                                            init_vals = init_val,
                                                            params = params_,
                                                            lambdas = lam,
                                                            solver = 'stoch',
                                                            max_iter = max_iter,
                                                            momentum_weights = mom_weight,
                                                            momentum_type = momentum
                                        ))
    
    return funcs

def trained_res_to_df(trained, test_data):

    out = list()
    for i in trained:
        
        trained_func = trained[i].trained_func()
        pred_res = test_data.apply(lambda x: trained_func(x['query'],x['target']),axis=1,result_type='expand').to_numpy()
        res_auc = auc(test_data['match', pred_res])
        out.append([
            trained[i].name,
            trained[i].distance_func,
            trained[i].loss_func,
            trained[i].reg_func,
            trained[i].init_vals,
            trained[i].params,
            trained[i].lambdas,
            trained[i].max_iter,
            trained[i].init_vals,
            res_auc
        ])

    return pd.DataFrame(out,columns=['name','disance_func','loss_func','reg_func','init_vals','params','lambdas','max_iter','init_vals'])

def orig_metric_to_df(metrics, test_data):

    out=list()
    for metric in metrics:

        pred_res = test_data.apply(lambda x: 1 - spectral_similarity.distance_sep(x['query'],x['target'],method=metric), axis=1, result_type='expand')
        res_auc = auc(test_data['match', pred_res])

        out.append([metric,res_auc])

    return pd.DataFrame(out, columns=['metric','AUC'])


#table other solvers for now
# solvers = ['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell']
# for reg_func in reg_funcs:
#     for loss in losses:
#         for name_, params_ in params.items():
#             for solver in solvers:

#                 funcs.append(func_ob.func_ob(name=f'{name_}_{reg_func}_{loss}_{solver}',
#                                             distance_func = TunaSims.tuna_dif_distance,
#                                             regularization_func = reg_func,
#                                             loss_func = loss,
#                                             init_vals = np.ones(len(params_))/10,
#                                             params = params_,
#                                             solver = solver,
#                 ))