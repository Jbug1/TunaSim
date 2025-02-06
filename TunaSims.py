import numpy as np
import scipy
import tools_fast
from dataclasses import dataclass


class TunaSim:

    @staticmethod
    def tuna_weight_intensity(spectrum,
                          fixed_exp = 0,
                          mz_exp = 0,
                          entropy_exp = 0,
                          offset = -1
                          ):
    
        mz_power_array = np.power(spectrum[:,0], mz_exp)
        intensities = spectrum[:,1]
        
        return np.power(intensities, offset + mz_power_array + fixed_exp + scipy.stats.entropy(intensities) ** entropy_exp)

    @staticmethod
    def smooth_reweight(array,
                        intercept,
                        a,
                        b,
                        c):
        ''' 
        give new weightings based on array values and passed parameters
        '''
        
        return intercept + a * array + b * np.power(array,2) + c * np.power(array,3)
        
    @staticmethod
    def sigmoid(z):
    
        return 1/(1 + np.exp(-z))
    
@dataclass
class TunaSmoothSim(TunaSim):
                
    query_mz_int: float = 0
    query_mz_a: float = 0
    query_mz_b: float = 0
    query_mz_c: float = 0
    target_mz_int: float = 0
    target_mz_a: float = 0
    target_mz_b: float = 0
    target_mz_c: float = 0
    query_mz_offset_int: float = 0
    query_mz_offset_a: float = 0
    query_mz_offset_b: float = 0
    query_mz_offset_c: float = 0
    target_mz_offset_int: float = 0
    target_mz_offset_a: float = 0
    target_mz_offset_b: float = 0
    target_mz_offset_c: float = 0
    query_intensity_int: float = 0
    query_intensity_a: float = 0
    query_intensity_b: float = 0
    query_intensity_c: float = 0
    target_intensity_int: float = 0
    target_intensity_a: float = 0
    target_intensity_b: float = 0
    target_intensity_c: float = 0
    query_normalized_intensity_int: float = 0
    query_normalized_intensity_a: float = 0
    query_normalized_intensity_b: float = 0
    query_normalized_intensity_c: float = 0
    target_normalized_intensity_int: float = 0
    target_normalized_intensity_a: float = 0
    target_normalized_intensity_b: float = 0
    target_normalized_intensity_c: float = 0
    query_entropy_int: float = 0
    query_entropy_a: float = 0
    query_entropy_b: float = 0
    query_entropy_c: float = 0
    target_entropy_int: float = 0
    target_entropy_a: float = 0
    target_entropy_b: float = 0
    target_entropy_c: float = 0
    dif_a: float = 0
    dif_b: float= 1
    mult_a: float = 0
    mult_b: float = 1
    add_norm_a: float= 0
    add_norm_b: float = 1
    mult_norm_a: float = 0
    mult_norm_b: float = 1
    unnormed: float = 0
    collapsed: float = 0
    expanded: float = 0
    match_tolerance: float = 0.05
    sim_flip: float= False
    zero_clip: float = True
    restandardize: float = True
    sig_factor: float = 1

    def __post_init__(self):

        self.set_weight_triggers()

    def set_weight_triggers(self):
        ''' 
        look at which parameters are set to determine which calculations are necessary
        '''
        triggers = ['mz',
                    'mz_offset',
                    'intensity',
                    'normalized_intensity',
                    'entropy'
                    ]
        #set initial state to False, if some input is non-zero, then flip it to True
        for trigger in triggers:
            setattr(self, trigger, False)
            for side in ['query','target']:
                for variable in ['int','a','b','c']:

                    if getattr(self, f'{side}_{trigger}_{variable}') != 0:
                        setattr(self, trigger, True)

    def predict(self,
                query,
                target,
                prec_query,
                prec_target):
        ''' 
        this funciton will yield a [0,1] interval similarity prediction
        predict also sets the values of potentially relevant gradint calculation parameters,
        and is therefore analagous to forward pass before backprop
        '''

        query_ints_reweight, target_ints_reweight = self.smooth_reweight(query, target, prec_query, prec_target)

        if self.zero_clip:
            query_ints_reweight = np.clip(query_ints_reweight,0)
            target_ints_reweight = np.clip(target_ints_reweight,0)

        if self.restandardize:
            query[:,1] /= np.sum(query[:,1])
            target[:,1] /= np.sum(target[:,1])
        
        #match peaks
        matched = tools_fast.match_peaks_in_spectra(query,
                                            target,
                                            ms2_da = self.ms2_da,
                                            ms2_ppm = self.ms2_ppm)

        #refer to matched specs only from now on
        query = matched[:,1]
        target = matched[:,2]

        #generate uncollapsed intensity combining funcitons
        expanded_difs = self.dif_a * np.abs(query - target) ** self.dif_b
        expanded_mults = self.mult_a * query * target ** self.mult_b

        #generate normalizations
        mult_norm = self.mult_norm_a * (np.sum(np.power(query, self.mult_norm_b)) * np.sum(np.power(target, self.mult_norm_b)))
        add_norm = self.add_norm_a * (np.power(query, self.add_norm_b) + np.power(target, self.add_norm_b))

        #potential unnormed term
        unnormed_term = 0
        if self.unnormed != 0:
            unnormed_term = self.unnormed * (np.sum(expanded_difs) + np.sum(expanded_mults))

        #depending on collapse and expand terms, consolidate or don't to some degree
        collapsed_term = 0
        if self.collapsed != 0:
            collapsed_difs = self.dif_a * np.sum(np.abs(query - target)) ** self.dif_b
            collapsed_mults = self.mult_a * np.sum(query * target) ** self.mult_b
            collapsed_term = self.collapsed * (collapsed_difs + collapsed_mults) / (np.sum(mult_norm) + np.sum(add_norm)) 

        #mult norm doesn't really make sense for expanded normalization
        expanded_term = 0
        if self.expanded != 0:
            expanded_term = self.expanded * np.sum((expanded_difs + expanded_mults) / add_norm)

        #some metrics are expressed as sim measures
        if self.sim_flip:
            return self.sigmoid(self.sig_factor *(unnormed_term + collapsed_term + expanded_term))
        else:
            return 1 - self.sigmoid(self.sig_factor *(unnormed_term + collapsed_term + expanded_term))


    def smooth_reweight(self,
                query,
                target,
                prec_query,
                prec_target):
        ''' 
        This function performs reweighting only
        '''

        #get reweight based on raw mz values
        if self.mz:
            self.query_mz_weights = self.tuna_smooth_weight(query[:,0], 
                                                                self.query_mz_int,
                                                                self.query_mz_a,
                                                                self.query_mz_b,
                                                                self.query_mz_c,
                                                            )
                            
            self.target_mz_weights = self.tuna_smooth_weight(target[:,0], 
                                                                self.target_mz_int,
                                                                self.target_mz_a,
                                                                self.target_mz_b,
                                                                self.target_mz_c,
                                                            )
        
        #grab weights for spectra as precursor offset
        if self.mz_offset:
            self.query_mz_offset_weights = self.tuna_smooth_weight(query[:,0] - prec_query, 
                                                                    self.query_mz_offset_int,
                                                                    self.query_mz_offset_a,
                                                                    self.query_mz_offset_b,
                                                                    self.query_mz_offset_c,
                                                                )
            
            self.target_mz_offset_weights = self.tuna_smooth_weight(target[:,0] - prec_target, 
                                                                    self.target_mz_offset_int, 
                                                                    self.target_mz_offset_a,
                                                                    self.target_mz_offset_b,
                                                                    self.target_mz_offset_c,
                                                                )
        
        #as a function of intensities
        if self.intensity:
            self.query_int_weights = self.tuna_smooth_weight(query[:,1], 
                                                                self.query_intensity_int,
                                                                self.query_intensity_a,
                                                                self.query_intensity_b,
                                                                self.query_intensity_c,
                                                            )
            
            self.target_int_weights = self.tuna_smooth_weight(target[:,1],
                                                                self.target_intensity_int, 
                                                                self.target_intensity_a,
                                                                self.target_intensity_b,
                                                                self.target_intensity_c,
                                                            )
        
        #as a funciton of normalized intensities
        if self.normalized_intensity:
            self.query_norm_int_weights = self.tuna_smooth_weight(query[:,1] /np.sum(query[:,1]), 
                                                            self.query_normalized_intensity_int,
                                                            self.query_normalized_intensity_a,
                                                            self.query_normalized_intensity_b,
                                                            self.query_normalized_intensity_c,
                                                            )
            
            self.target_norm_int_weights = self.tuna_smooth_weight(target[:,1] / np.sum(target[:,1]), 
                                                    self.target_normalized_intensity_int,
                                                    self.target_normalized_intensity_a,
                                                    self.target_normalized_intensity_b,
                                                    self.target_normalized_intensity_c,
                                                    )
        

        if self.entropy:
            self.query_entropy_weights = self.tuna_smooth_weight(np.zeros(len(query)) + scipy.stats.entropy(query[:,1]), 
                                                    self.query_entropy_int,
                                                    self.query_entropy_a,
                                                    self.query_entropy_b,
                                                    self.query_entropy_c,
                                                    )
            
            self.target_entropy_weights = self.tuna_smooth_weight(np.zeros(len(target)) + scipy.stats.entropy(target[:,1]), 
                                                    self.target_entropy_int,
                                                    self.target_entropy_a,
                                                    self.target_entropy_b,
                                                    self.target_entropy_c,
                                                    )
            
        #combine components for intensities and return
        return (self.combine_intensity_weights('query'), self.combine_intensity_weights('target'),0)


        
