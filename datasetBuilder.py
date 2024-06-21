# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
import tools
import numpy as np
import scipy
import spectral_similarity
import copy


def get_adduct_subset(nist_df): 

    return nist_df[
        (nist_df["precursor_type"] == "[M+H]+")
        | (nist_df["precursor_type"] == "[M-H]-")
    ]

def add_gauss_noise_to_peaks(spec, scale_ratio):

    noises = np.zeros(len(spec))
    for i in range(len(spec)):
        noises[i] = np.random.normal(scale=spec[i][1] * scale_ratio)

    spec[:, 1] = spec[:, 1] + noises
    spec[:, 1] = np.clip(spec[:, 1], a_min=0, a_max=None)

    return spec


def add_noises_to_matches(matches, scale_ratio, mult):

    matches["query"] = matches.apply(
        lambda x: add_gauss_noise_to_peaks(x["query"], scale_ratio=scale_ratio), axis=1
    )
    matches["query"] = matches.apply(
        lambda x: add_beta_noise_to_spectrum(x["query"], x["query_prec"], mult=mult),
        axis=1,
    )

    return matches



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

    #eventually this should be added to non-spec features

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


def add_beta_noise_to_spectrum(spec, precursor_mz, mult, noise_peaks=None):

    if noise_peaks is None:
        noise_peaks = len(spec)

    # generate noise mzs and intensities to be added
    noise_spec = np.zeros((noise_peaks, 2))
    noise_spec[:, 1] = np.random.beta(a=1, b=5, size=noise_peaks) * mult
    noise_spec[:, 0] = np.random.uniform(0, precursor_mz, size=noise_peaks)

    # build the final spectrum with mzs and combined peaks
    spec = np.concatenate((spec, noise_spec))
    spec = spec[spec[:, 0].argsort()]
    return spec




def clean_and_spec_features(
    spec1,
    prec1,
    spec2,
    prec2,
    noise_thresh,
    centroid_thresh,
    centroid_type="ppm",
    reweight_method=1,
    prec_remove=True,
    original_order=False
):
    """
    Function to clean the query and target specs according to parameters passed. Returns only matched spec
    """

    if centroid_type == "ppm":

        if prec_remove:

            spec1_ = tools.clean_spectrum(
                spec1,
                noise_removal=noise_thresh,
                ms2_ppm=centroid_thresh,
                standardize=False,
                max_mz=prec1-tools.ppm(prec1,3),
            )
            spec2_ = tools.clean_spectrum(
                spec2,
                noise_removal=noise_thresh,
                ms2_ppm=centroid_thresh,
                standardize=False,
                max_mz=prec2-tools.ppm(prec2,3),
            )
        else:
            spec1_ = tools.clean_spectrum(
                spec1,
                noise_removal=noise_thresh,
                ms2_ppm=centroid_thresh,
                standardize=True,
                max_mz=None,
            )
            spec2_ = tools.clean_spectrum(
                spec2,
                noise_removal=noise_thresh,
                ms2_ppm=centroid_thresh,
                standardize=False,
                max_mz=None,
            )

    else:
        if prec_remove:
            spec1_ = tools.clean_spectrum(
                spec1, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=True, max_mz=prec1 -tools.ppm(prec1,3)
            )
            spec2_ = tools.clean_spectrum(
                spec2, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=True, max_mz=prec2-tools.ppm(prec2,3)
            )
        else:
            spec1_ = tools.clean_spectrum(
            spec1, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=True
            )
            spec2_ = tools.clean_spectrum(
                spec2, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=True
            )

    if not original_order:
        # reweight by given reweight_method
        spec1_[:,1]= tools.weight_intensity(spec1_[:,1], reweight_method)
        spec2_[:,1]= tools.weight_intensity(spec2_[:,1], reweight_method)

    # get new spec features
    spec_features = get_spec_features(spec1_,spec2_)

    return spec_features + [spec1_, spec2_] #spec1_features + spec2_features 


def get_sim_features(query, lib, methods, ms2_da=None, ms2_ppm=None, original_order=False, reweight_method=None):

    
    sims = spectral_similarity.multiple_similarity(
        query, lib, methods, ms2_da=ms2_da, ms2_ppm=ms2_ppm, reweight_spectra=original_order, reweight_method=reweight_method
    )
    
    return [sims[i] for i in methods]
    

def create_matches_df(query_df, target_df, precursor_thresh, max_rows_per_query, max_len, adduct_match):

    non_spec_columns = [
        "precquery",
        "prectarget",
        "cequery",
        "cetarget",
        "instsame",
        "ceratio",
        "ceabs",
        "prec_abs_dif",
        "prec_ppm_dif",
        "mass_reduction_query",
        "mass_reduction_target",
        "mass_reduc_abs",
        "mass_reduc_ratio"
    ]
   
    #to be sure...shuffle query
    #query_df = query_df.sample(frac=1)

    out = None
    #target_df = target_df.sample(frac=1)
    printy = 1e5

    target_df['spectrum'] = [np.array(i, dtype=np.float64) for i in target_df['spectrum']]
    query_df['spectrum'] = [np.array(i, dtype=np.float64) for i in query_df['spectrum']]

    seen=0
    seen_=0
    unmatched=0
    cores_set=set()
    query_num=list()
    for i in range(len(query_df)):

        seen_+=1
        
        cores_set.add(query_df.iloc[i]['inchi_base'])

        if adduct_match:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) 
                & (query_df.iloc[i]["precursor_type"]==target_df["precursor_type"])
                & (target_df["queryID"]!=query_df.iloc[i]["queryID"])
            ]

        else:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) 
                & (query_df.iloc[i]["mode"]==target_df["mode"])
                & (target_df["queryID"]!=query_df.iloc[i]["queryID"])
            ]

        #catch case where there are no precursor matches
        if within_range.shape[0]==0:
            unmatched+=1
            continue

        #within_range = within_range.sample(frac=1)[:max_rows_per_query]
        within_range.reset_index(inplace=True)
        query_num = query_num+[i for _ in range(len(within_range))]
        seen += len(within_range)

        if seen > printy:

            print(f"{seen} rows created")
            printy = printy + 1e5

        if out is None:
            out = within_range.apply(
                lambda x: add_non_spec_features(query_df.iloc[i], x),
                axis=1,
                result_type="expand"
            )
            
            out.columns = non_spec_columns
            out['query_spec_ID'] = [query_df.iloc[i]["ID"] for x in range(len(within_range))]
            out['target_spec_ID'] = within_range["ID"].tolist()
            out["query"] = [query_df.iloc[i]["spectrum"] for x in range(len(out))]
            out["target"] = within_range["spectrum"].tolist()
            out["match"] = (
                query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            )

        else:
            temp = within_range.apply(
                lambda x: add_non_spec_features(query_df.iloc[i], x),
                axis=1,
                result_type="expand"
            )
            
            temp.columns = non_spec_columns
            temp['query_spec_ID'] = [query_df.iloc[i]["ID"] for x in range(len(within_range))]
            temp['target_spec_ID'] =within_range["ID"].tolist()
            temp["query"] = [query_df.iloc[i]["spectrum"] for x in range(len(temp))]
            temp["target"] = within_range["spectrum"]
            temp["match"] = (
                query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            )
            out = pd.concat([out, temp])

        if len(out) >= max_len:
            print(f'total number of query spectra considered: {seen_}')
            print(f'total number of target spectra considered: {seen}')
            print(f'total inchicores seen: {len(cores_set)}')
            print(f'{unmatched} queries went unmatched')
            return out

    print(f'total number of query spectra considered: {seen_}')
    print(f'total number of target spectra considered: {seen}')
    print(f'total inchicores seen: {len(cores_set)}')
    print(f'{unmatched} queries went unmatched')
    return out

def create_bare_matches_df_chunk(query_df, target_df, precursor_thresh, max_rows_per_query, max_len, adduct_match, chunk_size, outpath):

    """
    writes out to folder. No other nonspec info retained
    """
    out = None
    target_df['spectrum'] = [np.array(i, dtype=np.float64) for i in target_df['spectrum']]
    query_df['spectrum'] = [np.array(i, dtype=np.float64) for i in query_df['spectrum']]

    seen=0
    seen_=0
    unmatched=0
    chunk_counter=1
    cores_set=set()
    query_num=list()
    for i in range(len(query_df)):

        seen_+=1
        
        cores_set.add(query_df.iloc[i]['inchi_base'])

        if adduct_match:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) 
                & (query_df.iloc[i]["precursor_type"]==target_df["precursor_type"])
                & (target_df["queryID"]!=query_df.iloc[i]["queryID"])
            ]

        else:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) 
                & (query_df.iloc[i]["mode"]==target_df["mode"])
                & (target_df["queryID"]!=query_df.iloc[i]["queryID"])
            ]

        #catch case where there are no precursor matches
        if within_range.shape[0]==0:
            unmatched+=1
            continue
        
        #within_range = within_range.sample(frac=1)[:max_rows_per_query]
        within_range.reset_index(inplace=True)
        query_num = query_num+[i for _ in range(len(within_range))]
        seen += len(within_range)

        if out is None:
            out = {'query_spec_ID': [query_df.iloc[i]["ID"] for x in range(len(within_range))],
                   'target_spec_ID': within_range["ID"].tolist(),
                   'query':[query_df.iloc[i]["spectrum"] for x in range(len(within_range))],
                   'precquery':[query_df.iloc[i]["precursor"] for x in range(len(within_range))],
                   "target": within_range["spectrum"].tolist(),
                   "prectarget": within_range["precursor"].tolist(),
                   "match": query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            }
            out= pd.DataFrame(out)
        else:
            temp = {'query_spec_ID': [query_df.iloc[i]["ID"] for x in range(len(within_range))],
                   'target_spec_ID': within_range["ID"].tolist(),
                   'query':[query_df.iloc[i]["spectrum"] for x in range(len(within_range))],
                    'precquery':[query_df.iloc[i]["precursor"] for x in range(len(within_range))],
                   "target": within_range["spectrum"].tolist(),
                   "prectarget": within_range["precursor"].tolist(),
                   "match": query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            }
            temp= pd.DataFrame(temp)
            out = pd.concat([out, temp])
    
        if len(out) >= chunk_size or seen > max_len:

            outlen = len(out)
            out.dropna(how='any', inplace=True)
            if len(out)!=outlen:
                print(f'lost {outlen-len(out)} rows')

            out.to_pickle(f'{outpath}/chunk_{chunk_counter}.pkl')
            print(f'finished chunk {chunk_counter}')
            chunk_counter+=1
            out = None

            if seen > max_len:
                print(f'total number of query spectra considered: {seen_}')
                print(f'total number of target spectra considered: {seen}')
                print(f'total inchicores seen: {len(cores_set)}')
                print(f'{unmatched} queries went unmatched')

                return 
            
    out.to_pickle(f'{outpath}/chunk_{chunk_counter}.pkl')
    print(f'finished chunk {chunk_counter}')
    print(f'total number of query spectra considered: {seen_}')
    print(f'total number of target spectra considered: {seen}')
    print(f'total inchicores seen: {len(cores_set)}')
    print(f'{unmatched} queries went unmatched')

    return 

def clean_and_spec_features_single(
    spec1,
    prec1,
    noise_thresh,
    centroid_thresh,
    centroid_type="ppm",
    reweight_method=1,
    verbose=False
):
    """
    Function to clean the query and target specs according to parameters passed. Returns only matched spec
    """

    if verbose:
        print(spec1)
    
    if centroid_type == "ppm":

        spec1_ = tools.clean_spectrum(
            spec1,
            noise_removal=noise_thresh,
            ms2_ppm=centroid_thresh,
            standardize=False,
            max_mz=prec1,
        )

    else:
        spec1_ = tools.clean_spectrum(
            spec1, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=False
        )

    if verbose:
        print(spec1_)
        

    # reweight by given reweight_method
    spec1_[:,1]= tools.weight_intensity(spec1_[:,1], reweight_method)
    # print(spec1_)

    # get new spec features
    spec_features = get_spec_features_single(spec1_, prec1)
    if verbose:
        print(spec_features)

    spec1_ = tools.standardize_spectrum(spec1_)

    return spec_features + [spec1_]


def get_spec_features_single(spec, precursor):

    if len(spec) == 0:
        spec = np.array([[1, 0]])

    outrow = np.zeros(4)

    # first get all peaks below precursor mz
    below_prec_indices = np.where(spec[:, 0] < (precursor - tools.ppm(precursor, 3)))
    mass_reduction = np.sum(spec[below_prec_indices][:, 1]) / np.sum(spec[:, 1])

    spec = spec[below_prec_indices]

    n_peaks = len(spec)
    ent = scipy.stats.entropy(spec[:, 1])

    outrow[0] = ent
    outrow[1] = n_peaks

    if n_peaks < 2:
        outrow[2] = -1
    else:
        outrow[2] = ent / np.log(n_peaks)
    outrow[3] = mass_reduction

    return list(outrow)


def get_sim_features_all(targets, queries, sim_methods, ms2_ppm=None, ms2_da=None):
    """
    This function calculates the similarities of the queries (one parameter setting) against all target specs
    """

    if ms2_da is None and ms2_ppm is None:
        raise ValueError("need either ms2da or ms2ppm to proceed")

    sims_out = None

    for i in range(targets.shape[1]):

        temp = pd.concat((targets.iloc[:, i : i + 1], queries), axis=1)

        col0 = temp.columns[0]
        col1 = temp.columns[1]

        sims = temp.apply(
            lambda x: get_sim_features(
                x[col0], x[col1], methods=sim_methods, ms2_da=ms2_da, ms2_ppm=ms2_ppm
            ), 
            axis=1,
            result_type="expand"
        )

        if sims_out is None:
            sims_out = sims
        else:
            sims_out = pd.concat((sims_out, sims), axis=1)

    return sims_out

def create_cleaned_df(
    matches_df,
    sim_methods=None,
    noise_threshes=[0.01],
    centroid_tolerance_vals=[0.05],
    centroid_tolerance_types=["da"],
    reweight_methods=['orig'],
    prec_removes=[True],
    original_order=False
):
    """ """
    # create helper vars
    out_df = None
    spec_columns = [
        "ent_query",
        "npeaks_query",
        "normalent_query",
        "ent_target",
        "npeaks_target",
        "normalent_target",
    ]

    # create initial value spec columns
    #
    out_df=None
    for remove in prec_removes:

        init_spec_df = matches_df.apply(
            lambda x: get_spec_features(
                x["query"], x["target"]
            ),
            axis=1,
            result_type="expand",
        )

        init_spec_df.columns = spec_columns

        ticker = 0
        for i in noise_threshes:
            for j in reweight_methods:
                for k in range(len(centroid_tolerance_vals)):

                    ticker += 1
                    if ticker % 10 == 0:
                        print(f"added {ticker} settings")

                    spec_columns_ = [
                        f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{remove}"
                        for x in spec_columns
                    ]


                    sim_columns_ = [
                         f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{remove}"
                        for x in sim_methods
                    ]

                    # clean specs and get corresponding spec features
                    cleaned_df = matches_df.apply(
                        lambda x: clean_and_spec_features(
                            x["query"],
                            x["precquery"],
                            x["target"],
                            x["prectarget"],
                            noise_thresh=i,
                            centroid_thresh=centroid_tolerance_vals[k],
                            centroid_type=centroid_tolerance_types[k],
                            reweight_method=j,
                            prec_remove=remove,
                            original_order=original_order
                        ),
                        axis=1,
                        result_type="expand",
                    ).iloc[:,-2:]
                    
                    cleaned_df.columns = ["query", "target"]

                    if centroid_tolerance_types[k]=='da':
                        clean_matches = cleaned_df.apply(lambda x: tools.match_peaks_in_spectra_separate(
                                x['query'],
                                x['target'], 
                                ms2_da=centroid_tolerance_vals[k]
                            ),
                            axis=1,
                            result_type='expand'
                         )
                    else:
                        clean_matches = cleaned_df.apply(lambda x: tools.match_peaks_in_spectra_separate(
                                x['query'],
                                x['target'], 
                                ms2_ppm=centroid_tolerance_vals[k]
                            ),
                            axis=1,
                            result_type='expand'
                         )
                        
                    clean_matches.columns=[f'mzs_{remove}_{i}_{j}_{k}',f'query_{remove}_{i}_{j}_{k}',f'target_{remove}_{i}_{j}_{k}']

                    out_df = pd.concat(
                        (   
                            out_df,
                            clean_matches,
                        ),
                        axis=1,
                    )

        out_df['precursor'] = (matches_df['precquery'] + matches_df['prectarget'])/2
        out_df['match']=matches_df['match']
        return out_df

def create_model_dataset(
    matches_df,
    sim_methods=None,
    noise_threshes=[0.01],
    centroid_tolerance_vals=[0.05],
    centroid_tolerance_types=["da"],
    reweight_methods=['orig'],
    prec_removes=[True],
    original_order=False
):
    """ """
    # create helper vars
    out_df = None
    spec_columns = [
        "ent_query",
        "npeaks_query",
        "normalent_query",
        "ent_target",
        "npeaks_target",
        "normalent_target",
    ]

    # create initial value spec columns
    #
    for remove in prec_removes:

        init_spec_df = matches_df.apply(
            lambda x: get_spec_features(
                x["query"], x["target"]
            ),
            axis=1,
            result_type="expand",
        )

        init_spec_df.columns = spec_columns

        ticker = 0
        for i in noise_threshes:
            for j in reweight_methods:
                for k in range(len(centroid_tolerance_vals)):

                    ticker += 1
                    if ticker % 10 == 0:
                        print(f"added {ticker} settings")

                    spec_columns_ = [
                        f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{remove}"
                        for x in spec_columns
                    ]


                    sim_columns_ = [
                         f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{remove}"
                        for x in sim_methods
                    ]

                    # clean specs and get corresponding spec features
                    cleaned_df = matches_df.apply(
                        lambda x: clean_and_spec_features(
                            x["query"],
                            x["precquery"],
                            x["target"],
                            x["prectarget"],
                            noise_thresh=i,
                            centroid_thresh=centroid_tolerance_vals[k],
                            centroid_type=centroid_tolerance_types[k],
                            reweight_method=j,
                            prec_remove=remove,
                            original_order=original_order
                        ),
                        axis=1,
                        result_type="expand",
                    )


                    cleaned_df.columns = (
                        spec_columns_  + ["query", "target"]
                    )

                    
                    # create columns of similarity scores
                    if centroid_tolerance_types[k] == "ppm":
                        sim_df = cleaned_df.apply(
                            lambda x: get_sim_features(
                                x["query"],
                                x["target"],
                                sim_methods,
                                ms2_ppm=centroid_tolerance_vals[k],
                                original_order=original_order,
                                reweight_method=j
                            ),
                            axis=1,
                            result_type="expand",
                        )

                        sim_df.columns = sim_columns_

                    else:
                        
                        sim_df = cleaned_df.apply(
                            lambda x: get_sim_features(
                                x["query"],
                                x["target"],
                                sim_methods,
                                ms2_da=centroid_tolerance_vals[k],
                                original_order=original_order,
                                reweight_method=j
                            ),
                            axis=1,
                            result_type="expand",
                        )
                        
                        sim_df.columns = sim_columns_

                    # add everything to the output df
                    if out_df is None:

                        out_df = pd.concat(
                            (
                                matches_df.iloc[:, :-3],
                                init_spec_df,
                                cleaned_df.iloc[:, :-2],
                                sim_df,
                            ),
                            axis=1,
                        )

                    else:

                        out_df = pd.concat(
                            (
                                out_df,
                                cleaned_df.iloc[:, :-2],
                                sim_df,
                            ),
                            axis=1,
                        )

    out_df["match"] = matches_df["match"]
    out_df.drop(['query_spec_ID','target_spec_ID'], axis=1, inplace=True)
    return out_df

def generate_keep_indices(noise_threshes, centroid_tolerance_vals, reweight_methods, spec_features, sim_methods, prec_removes =[True],any_=False, nonspecs=False, init_spec=False):

    if nonspecs:
        keep_indices= list(range(13))
    else:
        keep_indices=list()

    if init_spec:
        keep_indices+=list(range(13,19))

    ind=19
    for _ in prec_removes:
        for i in noise_threshes:
            for j in centroid_tolerance_vals:
                for k in reweight_methods:
                    
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

