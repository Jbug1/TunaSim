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
from dataclasses import dataclass
from logging import getLogger

def ppm(base, ppm):
    """
    convert ppm threshold to dalton based on precursor exact mass (base)
    """

    return base * (ppm / 1e6)


@dataclass
class trainSetBuilder:
    
    query_input_path: str
    target_input_path: str
    dataset_names: list
    dataset_max_sizes: list
    identity_column: str
    outputs_dir: str
    ppm_match_window: int

    log = getLogger(__name__)

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

            self.create_matches_df(query, 
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


    def create_matches_df(self,
                          query_df, 
                        target_df, 
                        precursor_thresh, 
                        max_len, 
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
        del(target_df)

        seen = 0
        seen_ = 0
        unmatched = 0
        pieces = list()
        identities_set = set()

        #precursors held here for first part of search
        precursors_pos = target_pos['precursor'].to_numpy()
        precursors_neg = target_neg['precursor'].to_numpy()
        for i in range(len(query_df)):

            row = query_df.iloc[i]

            seen_+=1
            identities_set.add(row[self.identity_column])

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
            if within_range.shape[0] == 0:
                unmatched+=1
                continue
            
            seen += within_range.shape[0]

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

            chunk_df = pd.concat(pieces)
            chunk_df.to_pickle(outpath)
            
            self.log.info(f'match info for {outpath}')
            self.log.info(f'matched prec thresh: {precursor_thresh}, max len:{max_len} in {time.perf_counter()-start} ')
            self.log.info(f'total number of query spectra considered: {seen_}')
            self.log.info(f'total number of target spectra considered: {seen}')
            self.log.info(f'total identities seen: {len(identities_set)}')
            self.log.info(f'{unmatched} queries went unmatched for')

        return