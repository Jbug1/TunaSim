# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
import tools
import numpy as np
import scipy
import spectral_similarity
import copy
import time
import bisect
import os
import ms_entropy


def get_spec_features(spec_query,spec_target):

    #if we have invalid spectra, return all -1s...shouldn't happen
    if len(spec_query) == 0 or len(spec_target) == 0:
        return list(np.zeros(6)-1)
    
    outrow = np.zeros(6)

    # first get all peaks below precursor mz
    n_peaks = len(spec_query)
    ent = scipy.stats.entropy(spec_query[:, 1])

    outrow[0] = ent
    outrow[1] = n_peaks

    if n_peaks < 2:
        outrow[2] = -1
    else:
        outrow[2] = ent / np.log(n_peaks)

    n_peaks = len(spec_target)
    ent = scipy.stats.entropy(spec_target[:, 1])
    outrow[3] = ent
    outrow[4] = n_peaks
    if n_peaks < 2:
        outrow[5] = -1
    else:
        outrow[5] = ent / np.log(n_peaks)

    return list(outrow)


def add_non_spec_features(query_row, target_row):
    """
    query prec
    target prec
    query ce
    target ce
    instrument same
    ce ratio
    ce abs
    prec abs
    prec ppm
    mass reduction query
    mass reduction target
    mass reduction abs dif
    mass reduction ratio
    """

    outrow = np.zeros(13)

    # individual features
    outrow[0] = float(query_row["precursor"])
    outrow[1] = float(target_row["precursor"])
    outrow[2] = float(query_row["collision_energy"])
    outrow[3] = float(target_row["collision_energy"])

    # combined features
    outrow[4] = float(target_row["instrument"] == query_row["instrument"])

    if (
        float(target_row["collision_energy"]) > 0
        and float(query_row["collision_energy"]) > 0
    ):
        outrow[5] = max(
            float(target_row["collision_energy"]) / float(query_row["collision_energy"]),
            float(query_row["collision_energy"]) / float(target_row["collision_energy"]),
        )
    else:
        outrow[5] = 0

    outrow[6] = abs(
        float(target_row["collision_energy"]) - float(query_row["collision_energy"])
    )

    # precursor features
    outrow[7] = abs(query_row["precursor"] - target_row["precursor"])
    outrow[8] = abs(query_row["precursor"] - target_row["precursor"]) / tools.ppm(
        query_row["precursor"], 1
    )

    spec_query = query_row['spectrum']
    below_prec_indices = np.where(spec_query[:, 0] < (query_row["precursor"] - tools.ppm(query_row["precursor"], 3)))
    mass_reduc_query = np.sum(spec_query[below_prec_indices][:, 1]) / np.sum(spec_query[:, 1])
    outrow[9] = mass_reduc_query

    spec_target = target_row['spectrum']
    below_prec_indices = np.where(spec_target[:, 0] < (target_row["precursor"] - tools.ppm(target_row["precursor"], 3)))
    mass_reduc_target = np.sum(spec_target[below_prec_indices][:, 1]) / np.sum(spec_target[:, 1])
    outrow[10]=mass_reduc_target
    
    outrow[11] = abs(mass_reduc_query-mass_reduc_target)
    
    if mass_reduc_target==0:
        outrow[12]=-1
    else:
        outrow[12]=(mass_reduc_query/mass_reduc_target)

    return outrow

def clean_and_spec_features(
    spec1,
    prec1,
    spec2,
    prec2,
    noise_thresh,
    centroid_thresh,
    centroid_type="ppm",
    reweight_method=1,
    prec_remove=None,
):
    """
    Function to clean the query and target specs according to parameters passed. Returns only matched spec
    """

    if centroid_type == "ppm":

        spec1_ = tools.clean_spectrum(
            spec1,
            noise_removal = noise_thresh,
            ms2_ppm = centroid_thresh,
            standardize = False,
            max_mz=prec_remove(prec1),
        )
        spec2_ = tools.clean_spectrum(
            spec2,
            noise_threshold = noise_thresh,
            ms2_ppm = centroid_thresh,
            standardize = False,
            max_mz=prec_remove(prec2),
        )
    else:
        spec1_ = tools.clean_spectrum(
            spec1,
            noise_removal = noise_thresh,
            ms2_da = centroid_thresh,
            standardize = False,
            max_mz=prec_remove(prec1),
        )
        spec2_ = tools.clean_spectrum(
            spec2,
            noise_removal = noise_thresh,
            ms2_da = centroid_thresh,
            standardize = False,
            max_mz=prec_remove(prec2),
        )

    # reweight by given reweight_method
    if len(spec1_) > 0:
        spec1_[:,1] = tools.weight_intensity(spec1_[:,1], reweight_method)
    if len(spec2_) > 0:
        spec2_[:,1] = tools.weight_intensity(spec2_[:,1], reweight_method)
        

    # get new spec features
    spec_features = get_spec_features(spec1_,spec2_)

    return spec_features + [spec1_, spec2_] #spec1_features + spec2_features 


def get_sim_features(query, lib, methods, ms2_da=None, ms2_ppm=None):

    
    sims = spectral_similarity.multiple_similarity(
        query, lib, methods, ms2_da=ms2_da, ms2_ppm=ms2_ppm
    )
    
    return [sims[i] for i in methods]
    
def create_matches_df_chunk(query_df, 
                                 target_df, 
                                 precursor_thresh, 
                                 max_len, 
                                 chunk_size, 
                                 outpath, 
                                 logpath,
                                 adduct_match=False):

    """
    writes out to folder. No other nonspec info retained

    edited to also include inchiKey match as a field
    """

    start = time.perf_counter()

    num_trues_core = 0
    num_trues_key = 0
    target_df['spectrum'] = [np.array(i, dtype=np.float64) for i in target_df['spectrum']]
    query_df['spectrum'] = [np.array(i, dtype=np.float64) for i in query_df['spectrum']]

    #break into pos and neg, discard original
    target_pos = target_df[target_df['mode']=='+']
    target_neg = target_df[target_df['mode']=='-']

    #sort by precursor
    target_pos.sort_values(by='precursor', inplace=True)
    target_neg.sort_values(by='precursor', inplace=True)
    del(target_df)

    seen = 0
    seen_ = 0
    seen_chunk = 0
    unmatched=0
    chunk_counter=1
    pieces = list()
    cores_set=set()

    #precursors held here for first part of search
    precursors_pos = target_pos['precursor'].to_numpy()
    precursors_neg = target_neg['precursor'].to_numpy()
    for i in range(len(query_df)):

        seen_+=1
        cores_set.add(query_df.iloc[i]['inchi_base'])

        dif = tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)
            
        #get precursor boundaries and their corresponding indices
        upper = query_df.iloc[i]["precursor"] + dif
        lower = query_df.iloc[i]["precursor"] - dif

        #search against pos precursors if pos mode
        if query_df.iloc[i]['mode']=='+':
            lower_ind = bisect.bisect_right(precursors_pos, lower)
            upper_ind = bisect.bisect_left(precursors_pos,upper)
            within_range = target_pos.iloc[lower_ind:upper_ind]

        else:
            lower_ind = bisect.bisect_right(precursors_neg, lower)
            upper_ind = bisect.bisect_left(precursors_neg,upper)
            within_range = target_neg.iloc[lower_ind:upper_ind]

        if adduct_match:
            within_range = within_range[within_range['precursor_type'] == query_df.iloc[i]["precursor_type"]]

        #always exclude self match
        within_range = within_range[within_range['queryID'] != query_df.iloc[i]["queryID"]]

        #catch case where there are no precursor matches
        if within_range.shape[0]==0:
            unmatched+=1
            continue
        
        seen += within_range.shape[0]
        seen_chunk += within_range.shape[0]

        #add new columns and rename old, flush to csv
        within_range['precquery'] = [query_df.iloc[i]["precursor"] for x in range(len(within_range))]
        within_range['query'] = [query_df.iloc[i]["spectrum"] for x in range(len(within_range))]
        within_range["InchiCoreMatch"] = query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
        within_range["InchiKeyMatch"] = query_df.iloc[i]["inchi"] == within_range["inchi"]
        within_range['queryID'] = query_df.iloc[i]["queryID"]
        within_range.rename(columns={"precursor": "prectarget", "inchi_base": "target_base", 'spectrum':'target'}, inplace=True)
        within_range = within_range[['precquery','prectarget','query','target','queryID','target_base',"InchiCoreMatch","InchiKeyMatch"]]
        pieces.append(within_range)

        num_trues_core += sum(within_range["InchiCoreMatch"])
        num_trues_key += sum(within_range["InchiKeyMatch"])
    
        if seen_chunk >= chunk_size or seen > max_len:

            seen_chunk=0
            chunk_df = pd.concat(pieces)
            chunk_df.to_pickle(f'{outpath}/chunk_{chunk_counter}.pkl')
            chunk_counter+=1
            pieces = list()

            if chunk_counter % 10 == 0:
                with open(logpath,'a') as handle:
                    handle.write(f'finished {chunk_counter} chunks of size {chunk_size}\n')

            if seen > max_len:  
                with open(logpath,'a') as handle:

                    handle.write(f'matched prec thresh: {precursor_thresh}, max len:{max_len} adduct match: {adduct_match} in {time.perf_counter()-start}\n')
                    handle.write(f'total number of query spectra considered: {seen_}\n')
                    handle.write(f'total number of target spectra considered: {seen}\n')
                    handle.write(f'total inchicores seen: {len(cores_set)}\n')
                    handle.write(f'{unmatched} queries went unmatched\n')
                    handle.write(f'num true matches core: {num_trues_core} \n')
                    handle.write(f'num true matches key: {num_trues_key} \n')
                    
                return

    with open(logpath,'a') as handle:

        chunk_df = pd.concat(pieces)
        chunk_df.to_pickle(f'{outpath}/chunk_{chunk_counter}.pkl')
        
        handle.write(f'matched prec thresh: {precursor_thresh}, max len:{max_len} adduct match: {adduct_match} in {time.perf_counter()-start}\n')
        handle.write(f'total number of query spectra considered: {seen_}\n')
        handle.write(f'total number of target spectra considered: {seen}\n')
        handle.write(f'total inchicores seen: {len(cores_set)}\n')
        handle.write(f'{unmatched} queries went unmatched\n')
        handle.write(f'num true matches core: {num_trues_core} \n')
        handle.write(f'num true matches key: {num_trues_key} \n')

    return

def create_matches_and_model_data(query,
                                  target,
                                   matchesOutputPath,
                                   modelDataOutputPath,
                                   chunk_size,
                                   max_size,
                                   ppm_windows,
                                   noise_threshes,
                                   noise_names,
                                   centroid_tolerance_vals,
                                   centroid_tolerance_types,
                                   reweight_methods,
                                   reweight_names,
                                   sim_methods,
                                   prec_removes,
                                   prec_remove_names,
                                   adduct_match = False
                                   ):

    for i in ppm_windows:
        create_matches_df_chunk(query_df = query,
                                target_df = target,
                                precursor_thresh = i,
                                adduct_match = adduct_match,
                                max_len = max_size,
                                chunk_size= chunk_size,
                                outpath = f'{matchesOutputPath}/{i}_ppm',
                                logpath = f'{matchesOutputPath}/log_{i}_ppm.txt')


        create_model_dataset_chunk(
                                    input_path = f'{matchesOutputPath}/{i}_ppm',
                                    output_path = f'{modelDataOutputPath}/{i}_ppm',
                                    logpath = f'{modelDataOutputPath}/{i}_ppm/log.txt',
                                    sim_methods = sim_methods, 
                                    noise_threshes = noise_threshes,
                                    noise_names = noise_names, 
                                    centroid_tolerance_vals = centroid_tolerance_vals, 
                                    centroid_tolerance_types = centroid_tolerance_types,
                                    reweight_methods = reweight_methods,
                                    reweight_names = reweight_names,
                                    prec_removes = prec_removes,
                                    prec_remove_names = prec_remove_names
        )


def create_model_dataset_chunk(
    input_path,
    output_path,
    logpath,
    sim_methods=None,
    noise_threshes=[0.01],
    noise_names = ['1%'],
    centroid_tolerance_vals=[0.05],
    centroid_tolerance_types=["da"],
    reweight_methods=[None],
    reweight_names = ['none'],
    prec_removes=[None],
    prec_remove_names = ['none'],
):
    """ """

    start = time.perf_counter()
    # create helper vars
    spec_columns = [
        "ent_query",
        "npeaks_query",
        "normalent_query",
        "ent_target",
        "npeaks_target",
        "normalent_target",
    ]

    for chunk in os.listdir(input_path):

        if chunk[-3:] != 'pkl':
            continue

        matches_df = pd.read_pickle(f'{input_path}/{chunk}')
        pieces=list()
        pieces.append(matches_df.iloc[:,:2])
        # create initial value spec columns
        init_spec_df = matches_df.apply(
            lambda x: get_spec_features(
                x["query"], x["target"]
            ),
            axis=1,
            result_type="expand",
        )

        init_spec_df.columns = spec_columns
        pieces.append(init_spec_df)
        
        ticker = 0
        for remove in range(len(prec_removes)):
            for i in range(len(noise_threshes)):
                for j in range(len(reweight_methods)):
                    for k in range(len(centroid_tolerance_vals)):

                        spec_columns_ = [
                            f"{x}_n:{noise_names[i]} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{reweight_names[j]}, pr:{prec_remove_names[remove]}"
                            for x in spec_columns
                        ]


                        sim_columns_ = [
                            f"{x}_n:{noise_names[i]} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{reweight_names[j]}, pr:{prec_remove_names[remove]}"
                            for x in sim_methods
                        ]

                        # clean specs and get corresponding spec features
                        cleaned_df = matches_df.apply(
                            lambda x: clean_and_spec_features(
                                x["query"],
                                x["precquery"],
                                x["target"],
                                x["prectarget"],
                                noise_thresh=noise_threshes[i],
                                centroid_thresh=centroid_tolerance_vals[k],
                                centroid_type=centroid_tolerance_types[k],
                                reweight_method= reweight_methods[j],
                                prec_remove=prec_removes[remove],
                            ),
                            axis=1,
                            result_type="expand",
                        )


                        cleaned_df.columns = (
                            spec_columns_  + ["query", "target"]
                        )

                        pieces.append(cleaned_df.iloc[:,:-2])
                        
                        # create columns of similarity scores
                        if centroid_tolerance_types[k] == "ppm":
                            sim_df = cleaned_df.apply(
                                lambda x: get_sim_features(
                                    x["query"],
                                    x["target"],
                                    sim_methods,
                                    ms2_ppm = centroid_tolerance_vals[k]
                                ),
                                axis=1,
                                result_type="expand",
                            )

                        else:
                            
                            sim_df = cleaned_df.apply(
                                lambda x: get_sim_features(
                                    x["query"],
                                    x["target"],
                                    sim_methods,
                                    ms2_da = centroid_tolerance_vals[k]
                                ),
                                axis=1,
                                result_type="expand",
                            )
                            
                        sim_df.columns = sim_columns_
                        pieces.append(sim_df)

                        ticker += 1
                        if ticker % 10 == 0:
                            with open(logpath,'w') as handle:

                                handle.write(f"added {ticker} settings in {round(time.perf_counter() - start,2)}\n")
                                start = time.perf_counter()
                        
        out_df = pd.concat(pieces, axis=1)
        out_df["InchiCoreMatch"] = matches_df["InchiCoreMatch"]
        out_df["InchiKeyMatch"] = matches_df["InchiKeyMatch"]
        out_df.to_pickle(f'{output_path}/{chunk}')


def generate_keep_indices(noise_threshes,
                         centroid_tolerance_vals, 
                         reweight_methods, 
                         spec_features, 
                         sim_methods, 
                         prec_removes =[True],
                         any_=False, 
                         inits=False):

    if inits:
        keep_indices= list(range(8))
    else:
        keep_indices=list()

    ind = 8
    for _ in prec_removes:
        for i in noise_threshes:
            for k in reweight_methods:
                for j in centroid_tolerance_vals:
                  
                    for l in spec_features:
                        if any_:
                            if True in [i,j,k,l,_]:
                                keep_indices.append(ind)
                        else:
                            if i==j==k==l==_==True:
                                keep_indices.append(ind)
                        ind+=1

                    for l in sim_methods:
                        
                        if any_:
                            if True in [i,j,k,l,_]:
                                keep_indices.append(ind)
                        else:
                            if i==j==k==l==_==True:
                                keep_indices.append(ind)
                        ind+=1

    return keep_indices

