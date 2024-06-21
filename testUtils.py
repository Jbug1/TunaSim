import TunaSims
import func_ob
import spectral_similarity
import tools

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

def get_func_dist(df, pred_func, name):

    pred_res = df.apply(lambda x: pred_func(x['query'], x['target']), axis=1)
    score_res = df['match'].to_numpy()

    methods_range = {
        "proportional_entropy": [0, 1],
        "proportional_manhattan": [0, 1],
        "proportional_lorentzian": [0,1],
        "entropy": [0, np.log(4)],
        "dot_product": [0,1],
        #"cosine": [0,1],
        "absolute_value": [0, 2],
        "bhattacharya_1": [0, np.arccos(0) ** 2],
        "bhattacharya_2": [0, np.inf],
        "canberra": [0, np.inf],
        #"clark": [0, np.inf],
        #"avg_l": [0, 1.5],
        "divergence": [0, np.inf],
        #"euclidean": [0, np.sqrt(2)],
        "hellinger": [0, np.inf],
        #"improved_similarity": [0, np.inf],
        "lorentzian": [0, np.inf],
        "mod_lorentzian": [0, np.inf],
        "manhattan": [0, 2],
        "matusita": [0, np.sqrt(2)],
        "mean_character": [0, 2],
        "motyka": [-0.5, 0],
        "pearson_correlation": [-1, 1],
        #"penrose_shape": [0, np.sqrt(2)],
        #"penrose_size": [0, np.inf],
        "probabilistic_symmetric_chi_squared": [0, 1],
        "squared_chord": [0, 2],
        "mod_squared_chord": [0, 2],
        "mod2_squared_chord": [0, 2],
        "proportional_squared_chord" : [0, 2],
        "squared_euclidean": [0, 2],
        "symmetric_chi_squared": [0, 0.5 * np.sqrt(2)],
        "whittaker_index_of_association": [0, np.inf],
        #"perc_peaks_in_common": [0, 1],
        #"rbf": [0, 1],
        "chi2": [0, 1],
        #"cosine_kernel": [0, 1],
        "laplacian": [0, 1],
        #"minkowski": [0, 1],
        #"correlation": [0, 1],
        "jensenshannon": [0, 1],
        #"sqeuclidean": [0, 1],
        #"gini": [0, 1],
        #"l2": [0, 1],
        "common_mass": [0, 1],
        "cross_ent": [0, np.inf],
        "proportional_cross_ent": [0, np.inf],
        "braycurtis": [0, 1],
        "binary_cross_ent": [0, np.inf],
        "kl": [0, 1],
        #"chebyshev": [0, 1],
        "fidelity": [0, 1],
        "harmonic_mean": [0, 1],
        "ruzicka": [0, 1],
        "roberts": [0, 1],
        "intersection": [0, 1],}

    score_range = methods_range[name]

    pred_res = np.array([tools.normalize_distance(i, score_range) for i in pred_res])
    score_res = np.array([tools.normalize_distance(i, score_range) for i in score_res])

    return np.abs(pred_res-score_res)

def create_all_funcs_stoch(reg_funcs,
                    reg_names, 
                    losses, 
                    loss_names, 
                    momentums, 
                    params, 
                    mom_weights, 
                    inits,
                    lambdas=1, 
                    max_iters=[1e5],
                    func=None,
                    ):
    
    
    funcs=list()

    for  _ in range(len(reg_funcs)):
        for __ in range(len(losses)):
            for momentum in momentums:
                for name_, params_ in params.items():
                    if momentum != 'none':
                        for mom_weight in mom_weights:
                            for lam in lambdas:
                                for max_iter in max_iters:
                    
                                    funcs.append(func_ob.func_ob(name=f'{name_}_{reg_names[_]}_{loss_names[__]}_{momentum}_[{mom_weight}]_{lam}_{max_iter}',
                                                        sim_func = func,
                                                        regularization_func = reg_funcs[_],
                                                        loss_func = losses[__],
                                                        init_vals = [inits[x] for x in params_[0]],
                                                        bounds=params_[1],
                                                        params = params_[0],
                                                        lambdas = lam,
                                                        solver = 'stoch',
                                                        max_iter = max_iter,
                                                        momentum_weights = mom_weight,
                                                        momentum_type = momentum
                                    ))
                                            
                    else:
                        for lam in lambdas:
                            for max_iter in max_iters:
                                funcs.append(func_ob.func_ob(name=f'{name_}_{reg_names[_]}_{loss_names[__]}_{momentum}_none_{lam}_{max_iter}',
                                                    sim_func = func,
                                                    regularization_func = reg_funcs[_],
                                                    loss_func = losses[__],
                                                    init_vals = [inits[x] for x in params_[0]],
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
        pred_res = test_data.apply(lambda x: trained_func(x['precursor'],x['mzs'],x['query'],x['target']),axis=1,result_type='expand').to_numpy()

        if np.any(np.isnan(pred_res)) or np.any(np.abs(pred_res)>1e6):
            print(f'{trained[i].name}: nans present in pred res')
            continue
        
        res_auc = auc(test_data['match'], pred_res)
        temp = trained[i].name.split('_')[-7:]
        name = '_'.join(trained[i].name.split('_')[:-7])
        temp=[name]+temp +[res_auc]
        out.append(temp)

    return pd.DataFrame(out,columns=['name','reg','alpha','loss_func','momentum','weights','lambdas','max_iter','auc'])

def orig_metric_to_df(metrics, test_data, unnnormalized=False):
    """
    also returns metric scores by match for comparison
    """

    out=list()
    raw_sims = list()
    unnorm_dists=list()
    for metric in metrics:

        
        pred_res = test_data.apply(lambda x: 1 - spectral_similarity.distance_sep(x['query'],x['target'],method=metric), axis=1, result_type='expand').tolist()
        for i in range(len(pred_res)):
            if np.isnan(pred_res[i]):
                print(f'{metric}_{i}')
        res_auc = auc(test_data['match'], pred_res)
        raw_sims.append(pred_res)

        out.append([metric,res_auc])
        
        #also get unnormalized distance score if needed
        if unnnormalized:
            dist_res = test_data.apply(lambda x: spectral_similarity.distance_sep(x['query'],x['target'],method=metric, need_normalize_result=False), axis=1, result_type='expand').tolist()

            for i in range(len(dist_res)):
                if np.isnan(dist_res[i]):
                    print(f'{metric}_{i}')

            unnorm_dists.append(dist_res)

    raw_sims=pd.DataFrame(raw_sims).transpose()
    raw_sims.columns = metrics

    if unnnormalized:
        unnorm_dists=pd.DataFrame(unnorm_dists).transpose()
        unnorm_dists.columns = metrics
    
    if unnnormalized:
        return (pd.DataFrame(out, columns=['metric','AUC']),raw_sims, unnorm_dists)
    else:
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