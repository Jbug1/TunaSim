import TunaSims
import func_ob
import spectral_similarity

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc


def dict_combine(dict1,dict2):

    out=None

    if dict1 is not None:
        out=dict()
        for key,val in dict1.items():
            out[key]=val

    if dict2 is not None:
        if out is None:
            out=dict()
        for key,val in dict2.items():
            out[key]=val

    return out


def create_all_funcs_stoch(reg_funcs,
                    reg_names, 
                    losses, 
                    loss_names, 
                    momentums, 
                    params, 
                    mom_weights, 
                    lambdas=1, 
                    init_vals=None,
                    max_iters=[1e5],
                    func=None):
    
    funcs=list()

    for  i in range(len(reg_funcs)):
        for j in range(len(losses)):
            for momentum in momentums:
                for name_, params_ in params.items():
                    if momentum != 'none':
                        for mom_weight in mom_weights:
                            for lam in lambdas:
                                for max_iter in max_iters:
                                
                                    if init_vals is None:
                                        funcs.append(func_ob.func_ob(name=f'{name_}_{reg_names[i]}_{loss_names[j]}_{momentum}_[{mom_weight}]_{lam}_{max_iter}',
                                                            distance_func = func,
                                                            regularization_func = reg_funcs[i],
                                                            loss_func = losses[j],
                                                            init_vals = np.ones(len(params_[0]))/10,
                                                            bounds=params_[1],
                                                            params = params_[0],
                                                            lambdas = lam,
                                                            solver = 'stoch',
                                                            max_iter = max_iter,
                                                            momentum_weights = mom_weight,
                                                            momentum_type = momentum
                                        ))

                                    #broken...reimpliment
                                    else:
                                        for init_val in init_vals:
                                            funcs.append(func_ob.func_ob(name=f'{name_}_{reg_names[i]}_{loss_names[j]}_{momentum}_{mom_weight}',
                                                            distance_func = func,
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
                                            
                    else:
                        for lam in lambdas:
                            for max_iter in max_iters:
                                if init_vals is None:
                                    funcs.append(func_ob.func_ob(name=f'{name_}_{reg_names[i]}_{loss_names[j]}_{momentum}_none_{lam}_{max_iter}',
                                                        distance_func = func,
                                                        regularization_func = reg_funcs[i],
                                                        loss_func = losses[j],
                                                        init_vals = np.ones(len(params_[0]))/10,
                                                        bounds=params_[1],
                                                        params = params_[0],
                                                        lambdas = lam,
                                                        solver = 'stoch',
                                                        max_iter = max_iter,
                                                        momentum_type = momentum
                                    ))
    
    return funcs

def trained_res_to_df(trained, test_data):

    out = list()
    for i in range(len(trained)):

        if np.any(np.isnan(trained[i].trained_vals)):
            print(f'{trained[i].name} has nans')
            continue

        trained_func = trained[i].trained_func()
        pred_res = test_data.apply(lambda x: trained_func(x['query'],x['target']),axis=1,result_type='expand').to_numpy()

        if np.any(np.isnan(pred_res)) or np.any(np.abs(pred_res)>1e6):
            print('nans present in pred res')
            continue
        
        res_auc = auc(test_data['match'], pred_res)
        temp = trained[i].name.split('_')[-7:]
        name = '_'.join(trained[i].name.split('_')[:-7])
        temp=[name]+temp +[res_auc]
        out.append(temp)

    return pd.DataFrame(out,columns=['name','reg','alpha','loss_func','momentum','weights','lambdas','max_iter','auc'])

def orig_metric_to_df(metrics, test_data):
    """
    also returns metric scores by match for comparison
    """

    out=list()
    raw_sims = list()
    for metric in metrics:

        pred_res = test_data.apply(lambda x: 1 - spectral_similarity.distance_sep(x['query'],x['target'],method=metric), axis=1, result_type='expand').tolist()
        for i in range(len(pred_res)):
            if np.isnan(pred_res[i]):
                print(f'{metric}_{i}')
        res_auc = auc(test_data['match'], pred_res)
        raw_sims.append(pred_res)

        out.append([metric,res_auc])

    raw_sims=pd.DataFrame(raw_sims).transpose()
    raw_sims.columns = metrics
    
    return (pd.DataFrame(out, columns=['metric','AUC']),raw_sims)


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