# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
from tools_fast import match_spectrum
import numpy as np
import scipy
import copy
import time
import bisect
import os
#import ms_entropy

def ppm(base, ppm):
    """
    convert ppm threshold to dalton based on precursor exact mass (base)
    """

    return base * (ppm / 1e6)


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
    centroid_type="da",
    reweight_method=None,
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
            max_mz=prec_remove(prec1),
        )
        spec2_ = tools.clean_spectrum(
            spec2,
            noise_threshold = noise_thresh,
            ms2_ppm = centroid_thresh,
            max_mz=prec_remove(prec2),
        )
    else:
        spec1_ = tools.clean_spectrum(
            spec1,
            noise_removal = noise_thresh,
            ms2_da = centroid_thresh,
            max_mz=prec_remove(prec1),
        )
        spec2_ = tools.clean_spectrum(
            spec2,
            noise_removal = noise_thresh,
            ms2_da = centroid_thresh,
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
    
def create_matches_df(query_df, 
                        target_df, 
                        precursor_thresh, 
                        max_len, 
                        outpath):

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

        row = query_df.iloc[i]

        seen_+=1
        cores_set.add(row['inchi_base'])

        dif = ppm(row["precursor"], precursor_thresh)
            
        #get precursor boundaries and their corresponding indices
        upper = row["precursor"] + dif
        lower = row["precursor"] - dif

        #search against pos precursors if pos mode
        if row['mode']=='+':
            lower_ind = bisect.bisect_right(precursors_pos, lower)
            upper_ind = bisect.bisect_left(precursors_pos,upper)
            within_range = target_pos.iloc[lower_ind:upper_ind]

        else:
            lower_ind = bisect.bisect_right(precursors_neg, lower)
            upper_ind = bisect.bisect_left(precursors_neg,upper)
            within_range = target_neg.iloc[lower_ind:upper_ind]

        #always exclude self match
        within_range = within_range[within_range['queryID'] != query_df.iloc[i]["queryID"]]

        #catch case where there are no precursor matches
        if within_range.shape[0]==0:
            unmatched+=1
            continue
        
        seen += within_range.shape[0]
        seen_chunk += within_range.shape[0]

        #add new columns and rename old, flush to csv
        queries = list()
        targets = list()
        for target in within_range['target']:

            matched = match_spectrum(row['query'], 
                                     target, 
                                     ms2_da = self.ms2_da, 
                                     ms2_ppm = self.ms2_ppm)
            
            queries.append(matched[:,1] / np.sum(matched[:,1]))
            targets.append(matched[:,2] / np.sum(matched[:,2]))

        within_range['query'] = queries 
        within_range['target'] = targets

        within_range["score"] = row[self.match_column] == within_range[self.match_column]
        within_range['queryID'] = row["queryID"]
        pieces.append(within_range)

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

@dataclass
class trainSetBuilder:
    
    query_input_path: str
    target_input_path: str
    dataset_names: list
    dataset_max_sizes: list
    identity_column: str
    outputs_dir: str
    ppm_match_window: int

    def make_directory_structure(self):
        
        try:
            os.mkdir(self.outputs_dir, exist_ok = True)
            os.mkdir(f'{self.outputs_dir}/raw', exist_ok = True)
            os.mkdir(f'{self.outputs_dir}/raw/query', exist_ok = True)
            os.mkdir(f'{self.outputs_dir}/raw/target', exist_ok = True)
            os.mkdir(f'{self.outputs_dir}/matched', exist_ok = True)

            for name in self.dataset_names:
                os.mkdir(f'{self.outputs_dir}/matched/{name}', exist_ok = True)

        except Exception as e:
            self.log.error(f'unable to create directory structure: {e}')
            raise e
        
    def create_match_datasets(self):

        for dataset in os.listdir(f'{self.outputs_dir}/raw/query'):
         
            query = pd.read_pickle(f'{self.outputs_dir}/raw/query/{dataset}')
            target = pd.read_pickle(f'{self.outputs_dir}/raw/target/{dataset}')

            create_matches_df_chunk(query, 
                                    target,
                                    self.ppm_match_window,
                                    self.dataset_max_sizes[self.dataset_names.index(dataset)],
                                    None,
                                    f'{self.outputs_dir}/matched',
                                    f'loggy.log')

    def break_datasets(self):

        query = pd.read_pickle(self.query_input_path)

        query_identities = list(set(query[self.identity_column]))

        self.log.info(f'query length: {query.shape[0]}')
        self.log.info(f'query unique identities: {len(query_identities)}')

        target = pd.read_pickle(self.target_input_path)

        target_identities = list(set(target[self.identity_column]))

        self.log.info(f'target length: {query.shape[0]}')
        self.log.info(f'target unique identities: {len(query_identities)}')

        self.log.info(f'query length: {query.shape[0]}')
        self.log.info(f'query unique identities: {len(target_identities)}')

        all_identities = list(query_identities.union(target_identities))
        self.log.info(f'total unique identities: {len(target_identities)}')

        self.create_and_write_sub_dfs(all_identities,
                                      'target',
                                      target)
        
        self.log.info('wrote all sub dfs for target')

        self.create_and_write_sub_dfs(all_identities,
                                      'query',
                                      pd.read_pickle(self.query_input_path))
        
        self.log.info('wrote all sub dfs for target')

        self.create_match_datasets()


    def create_and_write_sub_dfs(self, all_identities, name, df):

        #build the different sets of identities
        identity_sets = list()
        assigned_inds = list()
        for i in range(len(self.dataset_names)):

            identity_sets.append(set(all_identities[int(i * len(all_identities) / len(self.dataset_names)): 
                                       int((i + 1) *len(all_identities) / len(self.dataset_names))]))
            
            assigned_inds.append(list())
            
        identities = df.iloc[self.identity_column]
        for i, identity in zip(list(range(len(identities))), identities):

            for j in range(len(identity_sets)):

                if identity in identity_sets[j]:

                    assigned_inds[j].append(i)
                    break

        for i in range(len(assigned_inds)):

            sub = df.iloc[assigned_inds[i]]
            sub.to_pickle(f'{self.outputs_dir}/{name}/{self.dataset_names[i]}.pkl')
        
