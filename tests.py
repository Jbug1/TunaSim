import tools
import spectral_similarity
import numpy as np
import time
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
import datasetBuilder
import copy
import os
from sklearn import base


def roc_curves_models(preds, trues):
    
    #get total true and false
    tot_true = np.sum(trues)
    tot_false = len(trues)-tot_true

    #sort results by predicted sim val
    sort_order = np.argsort(preds)
    preds=preds[sort_order]
    trues = trues[sort_order]

    running_pos=0
    running_neg=0

    ys_ = np.zeros(len(trues))
    xs_ = np.zeros(len(trues))

    for j in range(len(trues)):
        
        if trues[j]==1:
            running_pos+=1

        else:
            running_neg+=1

        ys_[j]=1- (running_pos/tot_true) #1- FNR
        xs_[j]=1- (running_neg/tot_false) #1-TNR

    return (xs_, ys_)

def fdr_curves_models(preds, trues):
    

    #sort results by predicted sim val in descending
    sort_order = np.argsort(preds)[::-1]
    preds=preds[sort_order]
    trues = trues[sort_order]

    #true labels
    running_neg=0

    ys_ = np.zeros(len(trues))

    for j in range(len(trues)):
        
        if trues[j]==0:
            running_neg+=1

        ys_[j]= running_neg/(j+1)

    return (preds[:-1000], ys_[:-1000])

def roc_curves_select_metrics(df, inds):

    "plot ROC curves for all metrics given"

    xs=list()
    ys=list()

    tot_true = np.sum(df['match'])
    tot_false = len(df)-tot_true
    for i in inds:

        df.sort_values(by=df.columns[i], inplace=True)

        running_pos=0
        running_neg=0

        ys_ = np.zeros(len(df))
        xs_ = np.zeros(len(df))

        for j in range(len(df)):
            
            if df.iloc[j]['match']==True:
                running_pos+=1

            else:
                running_neg+=1

            ys_[j]=1- (running_pos/tot_true) #1- FNR
            xs_[j]=1- (running_neg/tot_false) #1-TNR

        ys.append(ys_)
        xs.append(xs_)

    return (np.array(df.columns[inds]), xs, ys)



def break_data_into_quantiles(input, col_names, quantile_num, output_path):

    for i in col_names:

        os.mkdir(f'{output_path}/{i}')

        input.sort_values(by=i, inplace=True)

        chunk_size = int(len(input)/quantile_num)
        for j in range(quantile_num):
            
            quant_data = input.iloc[j*chunk_size:(j+1)*chunk_size]
            quant_data.to_pickle(f'{output_path}/{i}/{j}.pkl')

def aucs_by_quantile(input_folder,metrics):

    quantile_aucs=dict()
    for i in os.listdir(input_folder):

        quant=i.split('.')[0]

        try:
            quant=int(quant)
        except:
            continue

        quantile_aucs[quant]=list()

        quantile_res = pd.read_pickle(f'{input_folder}/{i}')

        for metric in metrics:
            quantile_res.sort_values(by=metric, inplace=True)
            quantile_aucs[quant].append((metric, auc(quantile_res['match'].to_numpy(),quantile_res[metric].to_numpy())))

    return quantile_aucs

def zero_one_loss(preds, true):
    
    preds=np.array(preds).squeeze()
    true=np.array(true).squeeze()
    
    return sum(abs(true-preds))/len(true)

def best_model_select(models, train, val, test):
    
    model_aucs = np.zeros(len(models))
    train['match']=train['match'].astype(int)
    val['match']=val['match'].astype(int)
    test['match']=test['match'].astype(int)

    best_model = None
    best_auc=0

    i=0
    best_true_ind=0
    for model in models:

        clf = base.clone(model)
        clf.fit(train.iloc[:,:-1], train.iloc[:,-1])
        true_index = np.where(clf.classes_==1)[0][0]
        true_probs = clf.predict_proba(val.iloc[:,:-1])[:,true_index].squeeze()
        val_auc = auc(val.iloc[:,-1:].to_numpy(),true_probs)

        if val_auc > best_auc:
            best_model=copy.deepcopy(clf)
            best_auc = val_auc
            best_true_ind = true_index

        del(clf)
        model_aucs[i]=val_auc
        i+=1
        
    true_probs = best_model.predict_proba(test.iloc[:,:-1])[:,best_true_ind].squeeze()
    return (best_model, auc(test.iloc[:,-1].to_numpy(),true_probs), model_aucs)

def best_models_by_subset(cols, train_sizes, models, train, val, test):

    res_dict=dict()
    for key, value in cols.items():

        for size in train_sizes:
    
            res_dict[key] = best_model_select(models, train.iloc[:size,value+[-1]], val.iloc[:,value+[-1]], test.iloc[:,value+[-1]])
            print(f'finished {key} for {size}')

    return res_dict

def create_variable_comparisons(noise_threshes, centroid_threshes, centroid_types, powers, sim_methods, prec_removes, matches, top_hit_only, outpath):

    for j in noise_threshes:
        for k in range(len(centroid_threshes)):
            for l in powers:
                for m in prec_removes:

                    cleaned = matches.apply(lambda x: datasetBuilder.clean_and_spec_features(x['query'],
                                                                                                x['precquery'],
                                                                                                x['target'],
                                                                                                x['prectarget'],
                                                                                                noise_thresh=j,
                                                                                                centroid_thresh = centroid_threshes[k],
                                                                                                centroid_type=centroid_types[k],
                                                                                                reweight_method=l,
                                                                                                prec_remove=m
                                                                                                ), 
                                                                                                axis=1,
                                                                                                result_type='expand')
                    
                    cleaned = cleaned.iloc[:,-2:]
                    cleaned.columns= ['query','library']

                    #gather all similarities   
                    try:                                                      
                        sims = run_metrics_sims(sim_methods,cleaned, tol_thresh = centroid_threshes[k], tol_type=centroid_types[k])
                        aucs = sims_to_auc(sims, matches.iloc[:,-1].to_numpy(), top_hit_only)
                        aucs['clean_specs'] =  f'n:{j}, c:{centroid_threshes[k]}{centroid_types[k]}, p:{l}, pr:{m}'
                        aucs.to_csv(outpath, mode='a', header=False)
                        print(f'added n:{j}, c:{centroid_threshes[k]}{centroid_types[k]}, p:{l}, pr:{m}')
                    except:
                        print(f'error on n:{j}, c:{centroid_threshes[k]}{centroid_types[k]}, p:{l}, pr:{m}')

def create_variable_comparisons_chunk(noise_threshes, centroid_threshes, centroid_types, powers, sim_methods, prec_removes, matches_folder, top_hit_only, outpath, original_order=True):

    sims_big = None
    for j in noise_threshes:
        for k in range(len(centroid_threshes)):
            for l in powers:
                for m in prec_removes:

                    count=1
                    match_labels=np.array([])
                    
                    for match in os.listdir(matches_folder):

                        if match[-3:] != 'pkl' or 'similarities' in match:
                            continue

                        #read pickled match df
                        matches = pd.read_pickle(f'{matches_folder}/{match}')
                        cleaned = matches.apply(lambda x: datasetBuilder.clean_and_spec_features(x['query'],
                                                                                                    x['precquery'],
                                                                                                    x['target'],
                                                                                                    x['prectarget'],
                                                                                                    noise_thresh=j,
                                                                                                    centroid_thresh = centroid_threshes[k],
                                                                                                    centroid_type=centroid_types[k],
                                                                                                    reweight_method=l,
                                                                                                    prec_remove=m,
                                                                                                    original_order=original_order
                                                                                                    ), 
                                                                                                    axis=1,
                                                                                                    result_type='expand')
                        
                        
                        match_labels = np.concatenate((match_labels,matches.iloc[:,-1].to_numpy()))
                        query_ids=matches['query_spec_ID'].tolist()
                        target_ids=matches['target_spec_ID'].tolist()
                        del(matches)
                        cleaned = cleaned.iloc[:,-2:]
                        cleaned.columns= ['query','library']

                        #gather all similarities                        
                        sims = run_metrics_sims(sim_methods,cleaned, tol_thresh = centroid_threshes[k], tol_type=centroid_types[k], original_order=original_order, reweight_method=l)
                        sims.insert(0,'query_spec_ID',query_ids)
                        sims.insert(1,'target_spec_ID',target_ids)
                        del(cleaned)
                        del(query_ids)
                        del(target_ids)

                        if sims_big is None:
                            sims_big=sims.copy()
                        else:
                            sims_big = pd.concat([sims_big,sims])

                        del(sims)
                        print(f'added {count} n:{j}, c:{centroid_threshes[k]}{centroid_types[k]}, p:{l}, pr:{m}')
                        count+=1

                    sims_big.to_pickle(f'{matches_folder}/similarities_{j}_{centroid_threshes[k]}_{centroid_types[k]}_{l}_{m}.pkl')
                    aucs = sims_to_auc(sims_big, match_labels, top_hit_only)
                    sims_big=None
                    aucs['clean_specs'] =  f'n:{j}, c:{centroid_threshes[k]}{centroid_types[k]}, p:{l}, pr:{m}'
                    aucs.to_csv(outpath, mode='a', header=False)
                    print('added to csv')


def run_metrics_sims(metrics, test, tol_thresh, tol_type, original_order=False, reweight_method=None):
    """
    create sims for all metrics
    """

    #make sure we have valid metric subset to look at
    if metrics is None:
        metrics=list()
        for i in spectral_similarity.methods_range:
            metrics.append(i)
            metrics.append('max_'+i)
           
        metrics.append('reverse_dot_product')
        metrics.remove('max_jensenshannon')

    if tol_type == 'da':
        sims = test.apply(lambda x: spectral_similarity.multiple_similarity(x['query'],x['library'],methods =metrics, ms2_da = tol_thresh,reweight_spectra=original_order, reweight_method=reweight_method), axis=1, result_type='expand')
    else:
        sims = test.apply(lambda x: spectral_similarity.multiple_similarity(x['query'],x['library'],methods =metrics, ms2_ppm = tol_thresh, reweight_spectra=original_order,reweight_method=reweight_method), axis=1, result_type='expand')

    return sims



def add_poisson_noise_to_spectrum(spec, precursor_mz, noise_peaks):

    # generate noise mzs and intensities to be added
    noise_intensities = np.random.poisson(lam=1, size=noise_peaks)
    noise_mzs = np.random.uniform(0, precursor_mz, size=noise_peaks)

    # sort noise_mzs
    noise_mzs.sort()

    # make noise spectrum
    noise_spec = list()
    for i in range(len(noise_mzs)):

        noise_spec.append([noise_mzs[i], noise_intensities[i]])

    noise_spec = np.array(noise_spec)

    # build the final spectrum with mzs and combined peaks
    out_spec = list()
    i = 0
    j = 0
    while i < len(spec) and j < len(len(noise_spec)):

        if spec[i][0] < noise_spec[j][0]:

            out_spec.append(spec[i])
            i += 1

        else:
            out_spec.append(noise_spec[j])
            j += 1

    while i < len(spec):

        out_spec.append(spec[i])
        i += 1

    while j < len(noise_spec):

        out_spec.append(noise_spec[j])
        j += 1

    return np.array(noise_spec)


def run_all_comparisons_models(test_dataset, precursor_mass_thresh, models):

    results_dict = dict()

    for i in range(len(models)):
        results_dict[i] = np.zeros((1000, 4))

    x = test_dataset[:, :-1]
    y = test_dataset[:, -1]

    for i in range(len(models)):
        y_hat = models[i].predict_proba(x)
        buckets = results_dict[i].shape[0] - 2

        for j in range(len(y_hat)):

            sim = y_hat[j][1]
            below_thresh_index = int(sim / (1 / buckets))

        if y[j] == True:

            results_dict[i][:below_thresh_index][:, 0] += 1
            results_dict[i][below_thresh_index:][:, 3] += 1

        else:

            results_dict[i][:below_thresh_index][:, 1] += 1
            results_dict[i][below_thresh_index:][:, 2] += 1


def run_all_comparisons(
    target_df,
    decoy_df=[],
    precursor_mass_thresh=10,
    metrics=None,
    compare_to_target=True,
    lim_rows=None,
):
    """
    This function compares all similarity metrics to each other, constructing a df of
    metrics used in evaluations for each one

    target_df: dataframe with actual spectra
    decoy_df: dataframe with decoy spectra
    precursor_mass_thresh: ppm num type
    results_dict:
    """
    results_dict = dict()

    if metrics == None:

        metrics = list()
        for i in spectral_similarity.methods_range:
            metrics.append(i)
            metrics.append("max_" + i)
            # metrics.append("reverse_" + i)
            metrics.append("min_" + i)
            metrics.append("ave_" + i)

    for i in metrics:
        results_dict[i] = np.zeros((1000, 4))

    start = time.time()
    for i in range(len(target_df)):

        if i == lim_rows:
            return results_dict

        if i % 1000 == 0:
            print(f"examined {i} rows")
            # print(time.time() - start)

        query_row = target_df.iloc[i]

        err = tools.ppm(query_row["precursor"], precursor_mass_thresh)
        upper = query_row["precursor"] + err
        lower = query_row["precursor"] - err

        # grab all spectra in precursor range of same ion type
        precursor_window_df_target = target_df[
            (target_df["precursor"] > lower)
            & (target_df["precursor"] < upper)
            & (target_df["precursor_type"] == query_row["precursor_type"])
        ]

        if len(decoy_df) > 0:

            # grab all spectra in precursor range of same ion type
            precursor_window_df_decoy = decoy_df[
                (decoy_df["precursor"] > lower)
                & (decoy_df["precursor"] < upper)
                & (decoy_df["precursor_type"] == query_row["precursor_type"])
            ]
        # print(i)
        if compare_to_target:
            for j in range(len(precursor_window_df_target)):

                target_row = precursor_window_df_target.iloc[j]

                # if target_row['index']!=query_row['index']:

                # create boolean match flag so we know if inchi key are the same
                match_flag = (
                    query_row["inchi"].split("-")[0]
                    == target_row["inchi"].split("-")[0]
                )

                # call update function to update dictionary
                update_results_dict(
                    target_row[0],
                    query_row[0],
                    match_flag,
                    metrics,
                    results_dict,
                )

        if len(decoy_df) > 0:
            for j in range(len(precursor_window_df_decoy)):

                target_row = precursor_window_df_decoy.iloc[j]

                # create boolean match flag so we know if inchi key are the same
                match_flag = (
                    query_row["inchi"].split("-")[0]
                    == target_row["inchi"].split("-")[0]
                )

                # call update function to update dictionary
                a = update_results_dict(
                    target_row[0],
                    query_row[0],
                    match_flag,
                    metrics,
                    results_dict,
                )

    return results_dict


def update_results_dict(spec_query, spec_reference, match_flag, metrics, results_dict):
    # set metrics to None to compute all similarities
    # grab all similarities with same parameters from the paper

    similarities = spectral_similarity.multiple_similarity(
        spec_query, spec_reference, methods=metrics, ms2_da=0.05
    )

    for i in similarities:

        sim = similarities[i]

        # verify number of buckets for ROC curve
        buckets = results_dict[i].shape[0] - 2
        try:
            below_thresh_index = int(sim / (1 / buckets)) + 1
        except:
            below_thresh_index = 0

        # these should be a match any negatives are false, any positives are true
        if match_flag:

            results_dict[i][:below_thresh_index][:, 0] += 1
            results_dict[i][below_thresh_index:][:, 3] += 1

        else:

            results_dict[i][:below_thresh_index][:, 1] += 1
            results_dict[i][below_thresh_index:][:, 2] += 1


def dict_to_df(dictionary):
    """ """
    # create helper of right shape for concatentation of results
    final = np.zeros((1, 5))
    for i in dictionary:

        entry_len = dictionary[i].shape[0]
        first = [i] * entry_len
        first = np.array(first)
        # first = np.array([[i] for i in first])
        first = first.reshape(-1, 1)

        out = np.concatenate((first, dictionary[i]), axis=1)
        final = np.concatenate((final, out))

    return pd.DataFrame(
        final[1:], columns=["metric", "true_pos", "false_pos", "true_neg", "false_neg"]
    )


def run_comparisons_with_noise(
    target,
    decoy,
    precursor_mass_thresh,
    results_dict,
    decoy_breaks,
    base_filepath,
    metrics=[],
):

    decoy = decoy.sample(frac=1).reset_index(drop=True)

    # do the first pass on target data only
    run_all_comparisons(
        target,
        [],
        precursor_mass_thresh,
        results_dict,
        compare_to_target=True,
        metrics=metrics,
    )

    out = dict_to_df(results_dict)
    out.to_csv(f"{base_filepath}_target.csv")

    print("*" * 20)
    print("Ran on target DF")
    print("*" * 20)

    # how big of a block of decoy df do we need to grab?
    inc = int(len(decoy) / decoy_breaks)
    for i in range(decoy_breaks):

        # run the comparisons only picking up on the noise dataframe
        run_all_comparisons(
            target,
            decoy[i * inc : (i + 1) * inc],
            precursor_mass_thresh,
            results_dict,
            compare_to_target=False,
        )

        out = dict_to_df(results_dict)
        out.to_csv(f"{base_filepath}_{i}.csv")

        print("*" * 20)
        print(f"Ran on decoy {i}")
        print("*" * 20)


def sims_to_auc(metrics, trues, top_hit_only):

    outdict=dict()
    outdict['metric']=list()
    outdict['auc']=list()

    #stick the match labels on if top hit only
    if top_hit_only:
        metrics['match']=trues

    for metric in metrics:

        if metric in ['queryID','match','query_spec_ID','target_spec_ID']:
            continue

        outdict['metric'].append(metric)

        #grab top hits if true
        #maybe fancier way to do with groupby but IDK
        if top_hit_only:
            temp = metrics.groupby(by='queryID').apply(lambda x: x[x[metric] == max(x[metric])])

            try:
                outdict['auc'].append(auc(temp['match'].to_numpy(), temp[metric].to_numpy()))
            except:
                outdict['auc'].append(-1)

        else:
            outdict['auc'].append(auc(trues, metrics[metric].to_numpy()))
        
    return pd.DataFrame(outdict)