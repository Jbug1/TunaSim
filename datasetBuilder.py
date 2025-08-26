# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
from tools_fast import match_spectrum
import numpy as np
import os
import time
import bisect
from pathlib import Path
from dataclasses import dataclass
from logging import getLogger

def ppm(base, ppm):
    """
    convert ppm threshold to dalton based on precursor exact mass (base)
    """

    return base * (ppm / 1e6)

@dataclass
class trainSetBuilder:

    def __init__(self,
                 query_input_path: str,
                 target_input_path: str,
                 dataset_names: list,
                 identity_column: str,
                 dataset_max_sizes: list,
                 outputs_directory: str,
                 ppm_match_window: int,
                 self_search: bool = False,
                 ms2_da: float = None,
                 ms2_ppm: float = None):
    
        self.query_input_path = query_input_path
        self.target_input_path = target_input_path
        self.dataset_names = dataset_names
        self.identity_column = identity_column
        self.dataset_max_sizes = dataset_max_sizes
        self.outputs_directory = outputs_directory
        self.ppm_match_window = ppm_match_window
        self.self_search = self_search
        self.ms2_da = ms2_da
        self.ms2_ppm = ms2_ppm

        if self.query_input_path == self.target_input_path:
            self.self_search = True

        self.log = getLogger(__name__)
    
    def make_directory_structure(self):
        
        try:
            os.makedirs(self.outputs_directory, exist_ok = True)
            os.makedirs(f'{self.outputs_directory}/raw', exist_ok = True)
            os.makedirs(f'{self.outputs_directory}/raw/query', exist_ok = True)
            os.makedirs(f'{self.outputs_directory}/raw/target', exist_ok = True)
            os.makedirs(f'{self.outputs_directory}/matched', exist_ok = True)

            # for name in self.dataset_names:
            #     os.makedirs(f'{self.outputs_directory}/matched/{name}', exist_ok = True)

        except Exception as e:
            self.log.error(f'unable to create directory structure: {e}')
            raise 
        
    def create_match_datasets(self):

        for dataset in [i.split('.')[0] for i in os.listdir(f'{self.outputs_directory}/raw/query')]:
         
            query = pd.read_pickle(f'{self.outputs_directory}/raw/query/{dataset}.pkl')
            target = pd.read_pickle(f'{self.outputs_directory}/raw/target/{dataset}.pkl')

            #shuffling query ensures good coverage of different identities if we are limiting dataset size
            query = query.sample(frac = 1)
            
            self.create_matches_df(query, 
                                    target,
                                    self.dataset_max_sizes[self.dataset_names.index(dataset)],
                                    f'{self.outputs_directory}/matched/{dataset}.pkl')

    def break_datasets(self):

        query = pd.read_pickle(self.query_input_path)

        query_identities = set(query[self.identity_column])

        self.log.info(f'query length: {query.shape[0]}')
        self.log.info(f'query unique identities: {len(query_identities)}')

        target = pd.read_pickle(self.target_input_path)
        target = target[['precursor', 'mode', 'spectrum', self.identity_column]]
        
        if self.self_search:
            target['queryID'] = list(range(target.shape[0]))
        else:
            target['queryID'] = [-1 for _ in range(target.shape[0])]

        target_identities = list(set(target[self.identity_column]))

        self.log.info(f'target length: {query.shape[0]}')
        self.log.info(f'target unique identities: {len(query_identities)}')

        all_identities = list(query_identities.union(target_identities))
        self.log.info(f'total unique identities: {len(target_identities)}')

        self.create_and_write_sub_dfs(all_identities,
                                      'target',
                                      target)
        
        self.log.info('wrote all sub dfs for target')

        query = pd.read_pickle(self.query_input_path)
        query = query[['precursor', 'mode', 'spectrum', self.identity_column]]
        query['queryID'] = list(range(query.shape[0]))
        
        self.create_and_write_sub_dfs(all_identities,
                                      'query',
                                      query
                                      )
        
        self.log.info('wrote all sub dfs for query')

        self.create_match_datasets()

    def create_and_write_sub_dfs(self, all_identities, name, df):

        #build the different sets of identities
        identity_sets = list()
        assigned_inds = list()
        for i in range(len(self.dataset_names)):

            identity_sets.append(set(all_identities[int(i * len(all_identities) / len(self.dataset_names)): 
                                       int((i + 1) *len(all_identities) / len(self.dataset_names))]))
            
            assigned_inds.append(list())
            
        identities = df[self.identity_column].tolist()
        for i, identity in zip(list(range(len(identities))), identities):

            for j in range(len(identity_sets)):

                if identity in identity_sets[j]:

                    assigned_inds[j].append(i)
                    break

        for i in range(len(assigned_inds)):

            sub = df.iloc[assigned_inds[i]]
            sub.to_pickle(f'{self.outputs_directory}/raw/{name}/{self.dataset_names[i]}.pkl')

    def create_matches_df(self,
                          query_df, 
                        target_df, 
                        max_size,
                        outpath):

        """
        writes out to folder. No other nonspec info retained

        edited to also include inchiKey match as a field
        """

        start = time.perf_counter()

        target_df['spectrum'] = [np.array(i, dtype=np.float64) for i in target_df['spectrum']]
        query_df['spectrum'] = [np.array(i, dtype=np.float64) for i in query_df['spectrum']]

        #break into pos and neg, discard original
        target_pos = target_df[target_df['mode']=='+']
        target_neg = target_df[target_df['mode']=='-']

        #sort by precursor
        target_pos.sort_values(by='precursor', inplace=True)
        target_neg.sort_values(by='precursor', inplace=True)

        seen = 0
        seen_ = 0
        unmatched = 0
        pieces = list()
        query_identities_set = set()

        #precursors held here for first part of search
        precursors_pos = target_pos['precursor'].to_numpy()
        precursors_neg = target_neg['precursor'].to_numpy()

        #retain only necessary columns
        target_neg = target_neg[['queryID', 'spectrum', self.identity_column]]
        target_pos = target_pos[['queryID', 'spectrum', self.identity_column]]

        rows = zip([i for i in range(len(query_df))],
                   query_df['mode'],
                   query_df['queryID'],
                    query_df['precursor'], 
                    query_df['spectrum'],
                    query_df[self.identity_column])
        
        for i, precursor, mode, queryID, spectrum, identity in rows:

            seen_ += 1
            query_identities_set.add(identity)

            dif = ppm(precursor, self.ppm_match_window)
                
            #get precursor boundaries and their corresponding indices
            upper = precursor + dif
            lower = precursor - dif

            #search against pos precursors if pos mode
            if mode == '+':
                lower_ind = bisect.bisect_right(precursors_pos, lower)
                upper_ind = bisect.bisect_left(precursors_pos,upper)
                within_range = target_pos.iloc[lower_ind:upper_ind]

            else:
                lower_ind = bisect.bisect_right(precursors_neg, lower)
                upper_ind = bisect.bisect_left(precursors_neg,upper)
                within_range = target_neg.iloc[lower_ind:upper_ind]

            #always exclude self match
            within_range = within_range[within_range['queryID'] != queryID]

            #catch case where there are no precursor matches
            if within_range.shape[0] == 0:
                unmatched+=1
                continue
            
            seen += within_range.shape[0]

            #add new columns and rename old, flush to csv
            queries = list()
            targets = list()
            for target in within_range['spectrum']:

                matched = match_spectrum(spectrum,
                                        target, 
                                        ms2_da = self.ms2_da, 
                                        ms2_ppm = self.ms2_ppm)
                
                queries.append(matched[:,1] / np.sum(matched[:,1]))
                targets.append(matched[:,2] / np.sum(matched[:,2]))

            within_range['query'] = queries 
            within_range['target'] = targets

            within_range["score"] = identity == within_range[self.identity_column]
            within_range['queryID'] = queryID

            pieces.append(within_range[['queryID', self.identity_column, 'query', 'target', 'score']])

            if seen > max_size:
                break

        chunk_df = pd.concat(pieces)
        chunk_df.to_pickle(outpath)
        
        self.log.info(f'match info for {outpath}')
        self.log.info(f'time: {round((time.perf_counter() - start)/60, 4)} minutes to complete')
        self.log.info(f'total number of query spectra considered: {seen_}')
        self.log.info(f'total number of target spectra considered: {seen}')
        self.log.info(f'total query identities seen: {len(query_identities_set)}')
        self.log.info(f'{unmatched} queries went unmatched')

        return