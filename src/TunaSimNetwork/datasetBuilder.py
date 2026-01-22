# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import os
import time
import bisect
from logging import getLogger
from numba import njit
import warnings
warnings.filterwarnings('ignore')

@njit
def compute_pair_scores(mz_a, mz_b, tolerance, units_ppm):
    """
    rank the associaton of matched peaks
    
    Args:
        mz_a, mz_b: Lists of mz for 2 spectra
        int_a, int_b: Lists of intensities for 2 spectra 
        tolerance: Maximum m/z difference ppm

    Returns:
        pairs: list of (i,j) tuples where mz_a[i] and mz_b[j] are within the tolerance window
        scores: list of real numbers from (0,1] where scores[i] is the score of the pair in pairs[i]
    """

    max_pairs = len(mz_a) * len(mz_b)
    pairs = np.zeros((max_pairs, 2), dtype=np.int32)
    scores = np.zeros(max_pairs, dtype=np.float64)

    count = 0
    b_start_ind = 0
    for i in range(mz_a.shape[0]):

        #convert ppm tolerance to da
        tolerance_ = tolerance * mz_a[i] / 1e6 if units_ppm else tolerance

        for j in range(b_start_ind, mz_b.shape[0]):

            #compute the difference in the mzs
            dif = np.abs(mz_a[i] - mz_b[j])

            #dif is within min tolerance
            if dif <= tolerance_:

                #add this pair score
                pairs[count] = [i, j]
                scores[count] = abs(dif)
                count += 1

            #otherwise we can increment the min index considered on b
            elif mz_a[i] > mz_b[j]:

                b_start_ind += 1

            else:
                break

    return pairs[:count], scores[:count]


def ppm(base, ppm):
    """
    convert ppm threshold to dalton based on precursor exact mass (base)
    """

    return base * (ppm / 1e6)

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
                 tolerance: float = None,
                 units_ppm: bool = False):
    
        self.query_input_path = query_input_path
        self.target_input_path = target_input_path
        self.dataset_names = dataset_names
        self.identity_column = identity_column
        self.dataset_max_sizes = dataset_max_sizes
        self.outputs_directory = outputs_directory
        self.ppm_match_window = ppm_match_window
        self.self_search = self_search
        self.tolerance = tolerance
        self.units_ppm = units_ppm

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

        except Exception as e:
            self.log.error(f'unable to create directory structure: {e}')
            raise 
        
    def create_match_datasets(self):

        for dataset in self.dataset_names:
         
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
                   query_df['precursor'], 
                   query_df['mode'],
                   query_df['queryID'],
                    query_df['spectrum'],
                    query_df[self.identity_column])
        
        for _, precursor, mode, queryID, spectrum, identity in rows:

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

                query_matched, target_matched = trainSetBuilder.match_spectra(spectrum.astype(np.float64),
                                                        target.astype(np.float64), 
                                                        tolerance = self.tolerance, 
                                                        units_ppm = self.units_ppm)
                
                queries.append(query_matched / np.sum(query_matched))
                targets.append(target_matched / np.sum(target_matched))

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
    
    @staticmethod
    def match_peaks_in_spectra(spec_a, spec_b, ms2_ppm=None, ms2_da=None):
        """
        Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
        :return: list. Each element in the list is a list contain three elements:
                                m/z, intensity from spec 1; intensity from spec 2.
        """

        a = 0
        b = 0

        spec_merged = []
        peak_b_int = 0.

        while a < spec_a.shape[0] and b < spec_b.shape[0]:
            if ms2_da is None:
                ms2_da = ms2_ppm * spec_a[a, 0] * 1e-6
            mass_delta = spec_a[a, 0] - spec_b[b, 0]

            if mass_delta < -ms2_da:
                # Peak only existed in spec a.
                spec_merged.append([spec_a[a, 0], spec_a[a, 1], peak_b_int])
                peak_b_int = 0.
                a += 1
            elif mass_delta > ms2_da:
                # Peak only existed in spec b.
                spec_merged.append([spec_b[b, 0], 0., spec_b[b, 1]])
                b += 1
            else:
                # Peak existed in both spec.
                peak_b_int += spec_b[b, 1]
                b += 1

        if peak_b_int > 0.:
            spec_merged.append([spec_a[a, 0], spec_a[a, 1], peak_b_int])
            peak_b_int = 0.
            a += 1

        if b < spec_b.shape[0]:
            spec_merged += [[x[0], 0., x[1]] for x in spec_b[b:]]

        if a < spec_a.shape[0]:
            spec_merged += [[x[0], x[1], 0.] for x in spec_a[a:]]

        if spec_merged:
            spec_merged = np.array(spec_merged, dtype=np.float64)
        else:
            spec_merged = np.array([[0., 0., 0.]], dtype=np.float64)
        return spec_merged

    
    @staticmethod
    def match_spectra(spec_a, spec_b, tolerance, units_ppm):
        """
        Perform greedy peak matching between two spectra using ppm tolerance
        
        Args:
            spec_a, spec_b: Lists of [mz, intensity] pairs 
            tolerance: Maximum m/z difference in parts per million 
        Returns:
            intensities_a, intensities_b: List of peak intensities where intensities_a[i] is matched to intensities_b[i]
        """

        #check that specs are not empty
        if spec_a.size == 0 or spec_b.size == 0:
            return np.array([]), np.array([])

        #ensure that all intensities are > 0 first
        spec_a = spec_a[spec_a[:,1] > 0]
        spec_b = spec_b[spec_b[:,1] > 0]

        mz_a, int_a = spec_a[:, 0], spec_a[:, 1]
        mz_b, int_b = spec_b[:, 0], spec_b[:, 1]

        combined_spec = np.zeros((mz_a.shape[0] + mz_b.shape[0], 2), dtype = np.float64)
        
        pairs, scores = compute_pair_scores(mz_a, mz_b, tolerance, units_ppm)
        
        # Initialize combined spectrum with all spec_a peaks (unmatched have int_b=0)
        combined_spec[:mz_a.shape[0], 0] += int_a
        matched_b = set()
        
        if len(pairs) > 0:
            sort_indices = np.argsort(scores)
            pairs = pairs[sort_indices]
            
            matched_a = set()
            for i, j in pairs:
                if i not in matched_a and j not in matched_b:
                    combined_spec[i][1] = int_b[j]
                    matched_a.add(i)
                    matched_b.add(j)
        
        # Add unmatched spec_b peaks w no spec_a match
        unmatched_count = mz_a.shape[0]
        for j in range(len(mz_b)):
            if j not in matched_b:

                combined_spec[unmatched_count][1] = int_b[j]
                unmatched_count +=1
        
        
        return combined_spec[:unmatched_count, 0], combined_spec[:unmatched_count, 1]


class specCleaner:

    def __init__(self,
                 noise_threshold: float = 0.01,
                 deisotoping_gaps: list[float] = [1.003355],
                 isotope_mz_tolerance: float = 0.005,
                 precursor_removal_window_mz: float = 1):
        
        self.noise_threshold = noise_threshold
        self.deisotoping_gaps = deisotoping_gaps
        self.isotope_mz_tolerance = isotope_mz_tolerance
        self.precursor_removal_window_mz = precursor_removal_window_mz

    def clean_spectra(self,
                      raw_spectra: list,
                      precursors: list) -> list:
        
        output_specs = list()
        for spec, prec in zip(raw_spectra, precursors):

            if len(spec) == 0:
                output_specs.append(spec)

            else:
                output_specs.append(self.clean_spectrum(spec, prec))

        return output_specs
    
    @staticmethod
    @njit
    def consolidate_isotopic_peaks(spec: NDArray, 
                                   gap: float, 
                                   mz_tolerance: float) -> NDArray:
        """ 
        consolidates intensities of isotopic peaks at the monoisotopic (most intense) mz
        this one is going to be a beast because it needs to remain jit-able (no dealing with other objects)
        """

        #handle empty input
        if spec.size == 0:
            return np.empty((0,0), dtype = np.float64)
        
        #create output placeholder
        output = np.zeros(spec.shape, dtype = np.float64)

        i = 0
        j = 1

        while j < spec.shape[0]:

            #jth value is below minimum
            if spec[i,0] + gap - mz_tolerance > spec[j,0]:
                j += 1

            #jth value is below maximum
            #we are within the window for consolidation
            elif spec[i,0] + gap + mz_tolerance > spec[j,0]:

                #assign combined intensity to monoisotopic peak
                #ith value is more prevalent isotope
                if spec[i,1] > spec[j,1]:

                    #check if we have consolidated i's intensity elsewhere
                    monoisotopic_index = i if output[i,0] == 0 else int(-output[i,0] - 1)

                    #update monoisotopic index of output with j index intensity
                    output[monoisotopic_index, 0] = spec[monoisotopic_index,0]
                    output[monoisotopic_index, 1] += spec[j,1]

                    #also add i's intensity if we have not previously consolidated it
                    if monoisotopic_index == i:

                        output[monoisotopic_index, 1] += spec[i,1]

                    #note in output that we have already consolidated the jth index to the ith position's intensity
                    #not that this is the negative index so that we can easily filter later
                    #add minus 1 to catch caswe of consolidation of first peak
                    output[j, 0] = -monoisotopic_index - 1

                #jth value is monoisotopic...be sure to set ith index to 0 for mz and int
                else:

                    #add index i's intensity in the j position
                    #do nothing at ith index (will be removed)
                    output[j, 1] = spec[i,1]

                #increment both pointers
                j += 1
                i += 1

            #jth value is above max value for consolidation
            else:

                #check to make sure that ith index has not already been consolidated to another
                if output[i,0] >= 0:
                    output[i] += spec[i]

                i += 1

        #once j hits the end of the spec, there may be remaining peaks on the i index to add
        while i < spec.shape[0]:

            #check to make sure that ith index has not already been consolidated to another
                if output[i,0] >= 0:
                    output[i] += spec[i]

                i += 1

        #return only the necessary indices (those which have >0 values), will exclude unused and consolidated
        return output[output[:,0] > 0]

    def clean_spectrum(self,
                       spec: NDArray, 
                       precursor_mz: float) -> NDArray:
        """
        runs through basic cleaning steps and calls deisotoping protocol
        """

        #ensure no negative values
        if np.any(spec[:,0] < 0):

            raise ValueError('Negative values on mz dimension')
        
        if np.any(spec[:,1] < 0):

            raise ValueError('Negative values on intensity dimension')

        #ensure spectrum is sorted in ascending order by mz
        spec = spec[spec[:,0].argsort()]

        #deisotoping
        for isotope_gap in self.deisotoping_gaps:
        
            spec = self.consolidate_isotopic_peaks(spec, isotope_gap, self.isotope_mz_tolerance)

        #precursor removal
        spec = spec[spec[:,0] < precursor_mz - self.precursor_removal_window_mz]

        if spec.shape[0] > 0:

            #noise peak clipping
            spec = spec[spec[:,1] > np.max(spec[:,1]) * self.noise_threshold]

            #renormalize
            spec[:,1] = spec[:,1] / np.sum(spec[:,1])

        return spec

