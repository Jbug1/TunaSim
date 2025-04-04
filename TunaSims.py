import numpy as np
import scipy
import tools_fast
from dataclasses import dataclass
from math import prod

@dataclass
class TunaSim:

    query = None
    target = None
    prec_query = None
    prec_target = None
    zero_clip: float = True
    standardize: float = True
    set_grad1: bool = True
    set_grad2: bool = True
    lazy_grads1_reweight = None

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
    def sigmoid(z):
    
        return 1/(1 + np.exp(-z))
    
    def restandardize(self):

        if self.lazy_grads1_reweight is not None:

            self.query[:,1] /= np.sum(self.query[:,1])
            self.target[:,1] /= np.sum(self.target[:,1])
    
    def clip(self):

        if self.set_grad1 and self.lazy_grads1_reweight is not None:



    
@dataclass
class TunaSmoothSim(TunaSim):
                
    query_mz_a: float = 0
    query_mz_b: float = 0
    query_mz_c: float = 0
    target_mz_a: float = 0
    target_mz_b: float = 0
    target_mz_c: float = 0
    query_mz_offset_a: float = 0
    query_mz_offset_b: float = 0
    query_mz_offset_c: float = 0
    target_mz_offset_a: float = 0
    target_mz_offset_b: float = 0
    target_mz_offset_c: float = 0
    query_intensity_a: float = 0
    query_intensity_b: float = 0
    query_intensity_c: float = 0
    target_intensity_a: float = 0
    target_intensity_b: float = 0
    target_intensity_c: float = 0
    query_normalized_intensity_a: float = 0
    query_normalized_intensity_b: float = 0
    query_normalized_intensity_c: float = 0
    target_normalized_intensity_a: float = 0
    target_normalized_intensity_b: float = 0
    target_normalized_intensity_c: float = 0
    query_entropy_a: float = 0
    query_entropy_b: float = 0
    query_entropy_c: float = 0
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
    sig_factor: float = 1
    ms2_da: float = None
    ms2_ppm: float = None
    weight_combine: str = 'add'

    lazy_grads1_reweight = dict()
    lazy_grads2_reweight = dict()


    def __post_init__(self):

        self.set_weight_triggers()

        self.query_mz_weights = None
        self.query_mz_offset_weights = None
        self.query_intensity_weights = None
        self.query_normalized_intensity_weights = None
        self.query_entropy_weights = None

        self.target_mz_weights = None
        self.target_mz_offset_weights = None
        self.target_intensity_weights = None
        self.target_normalized_intensity_weights = None
        self.target_entropy_weights = None

    def smooth_reweight(self,
                        name,
                        array,
                        intercept,
                        a,
                        b,
                        c):
        ''' 
        give new weightings based on array values and passed parameters
        '''

        if self.set_grad1:

            if intercept != 0:
                self.lazy_grads1_reweight[f'{name}_int'] = np.ones(len(array))
            
            if a != 0:
                self.lazy_grads1_reweight[f'{name}_a'] = array

            if b != 0:
                self.lazy_grads1_reweight[f'{name}_a'] = array**2

            if c != 0:
                self.lazy_grads1_reweight[f'{name}_a'] = array**3
        
        return intercept + a * array + b * np.power(array,2) + c * np.power(array,3)

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
        #begin under the assumption that we will do no reweighting
        self.unweighted = True
        for trigger in triggers:
            for side in ['query','target']:
                setattr(self, f'{trigger}_{side}', False)
                for variable in ['int','a','b','c']:

                    if getattr(self, f'{side}_{trigger}_{variable}') != 0:

                        #set the proper side to let us know to compute values and track derivatives
                        #set unweighted to false if at least one is true
                        setattr(self, f'{trigger}_{side}', True)
                        self.unweighted = False


    def predict(self):

        ''' 
        this function will yield a [0,1] interval similarity prediction
        predict also sets the values of potentially relevant gradint calculation parameters,
        and is therefore analagous to forward pass before backprop
        '''

        #set reweighted query and target
        self.set_reweighted_intensity()

        if self.zero_clip:
            query_ints_reweight = np.clip(query_ints_reweight,0)
            target_ints_reweight = np.clip(target_ints_reweight,0)

        self.consolidate()
        
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


    def set_reweighted_intensity(self):
        ''' 
        This function performs reweighting and deriv setting for reweighting
        '''

        #roll with original query if no reweighting is required
        if self.unweighted:
            return

        #get reweight based on raw mz values
        if self.query_mz:
            self.query_mz_weights = self.smooth_reweight(self.query[:,0],
                                                        'mz_query', 
                                                        self.query_mz_a,
                                                        self.query_mz_b,
                                                        self.query_mz_c,
                                                        )
            
        if self.target_mz:           
            self.target_mz_weights = self.smooth_reweight(self.target[:,0], 
                                                          'mz_target', 
                                                            self.target_mz_a,
                                                            self.target_mz_b,
                                                            self.target_mz_c,
                                                            )
        
        #grab weights for spectra as precursor offset
        if self.query_mz_offset:
            self.query_mz_offset_weights = self.smooth_reweight(self.query[:,0] - self.prec_query, 
                                                                'mz_offset_query',
                                                                self.query_mz_offset_a,
                                                                self.query_mz_offset_b,
                                                                self.query_mz_offset_c,
                                                                )
            
        if self.target_mz_offset:   
            self.target_mz_offset_weights = self.smooth_reweight(self.target[:,0] - self.prec_target, 
                                                                 'mz_offset_target',
                                                                self.target_mz_offset_a,
                                                                self.target_mz_offset_b,
                                                                self.target_mz_offset_c,
                                                                )
        
        #as a function of intensities
        if self.query_intensity:
            self.query_intensity_weights = self.smooth_reweight(self.query[:,1], 
                                                                'intensity_query',
                                                                self.query_intensity_a,
                                                                self.query_intensity_b,
                                                                self.query_intensity_c,
                                                            )
            
        if self.target_intensity:
            self.target_intensity_weights = self.smooth_reweight(self.target[:,1],
                                                                'intensity_target',
                                                                self.target_intensity_a,
                                                                self.target_intensity_b,
                                                                self.target_intensity_c,
                                                            )
        
        #as a function of normalized intensities
        if self.query_normalized_intensity:
            self.query_normalized_intensity_weights = self.smooth_reweight(self.query[:,1] /np.sum(self.query[:,1]), 
                                                                            'normalized_intensity_query',
                                                                            self.query_normalized_intensity_a,
                                                                            self.query_normalized_intensity_b,
                                                                            self.query_normalized_intensity_c,
                                                                            )
            
        if self.target_normalized_intensity:   
            self.target_normalized_intensity_weights = self.smooth_reweight(self.target[:,1] / np.sum(self.target[:,1]), 
                                                                            'normalized_intensity_target',
                                                                            self.target_normalized_intensity_a,
                                                                            self.target_normalized_intensity_b,
                                                                            self.target_normalized_intensity_c,
                                                                            )

        if self.query_entropy:
            self.query_entropy_weights = self.smooth_reweight(np.zeros(len(self.query)) + scipy.stats.entropy(self.query[:,1]), 
                                                              'entropy_query',
                                                                self.query_entropy_a,
                                                                self.query_entropy_b,
                                                                self.query_entropy_c,
                                                                )
            
        if self.target_entropy:
            self.target_entropy_weights = self.smooth_reweight(np.zeros(len(self.target)) + scipy.stats.entropy(self.target[:,1]),
                                                               'entropy_target', 
                                                                self.target_entropy_a,
                                                                self.target_entropy_b,
                                                                self.target_entropy_c,
                                                                )
            
        #combine components for intensities and return
        self.combine_intensity_weights()

    
    def combine_intensity_weights(self):

        query_components = [self.query_int,
                            self.query_mz_weights,
                            self.query_mz_offset_weights,
                            self.query_intensity_weights,
                            self.query_normalized_intensity_weights,
                            self.query_entropy_weights]
        
        target_components = [self.target_int,
                            self.target_mz_weights,
                            self.target_mz_offset_weights,
                            self.target_intensity_weights,
                            self.target_normalized_intensity_weights,
                            self.target_entropy_weights]
        
        query_components = [i for i in query_components if i is not None]
        target_components = [i for i in target_components if i is not None]

        if self.weight_combine == 'add':

            query_intensities = sum(query_components)
            target_intensities = sum(target_components)

        elif self.weight_combine == 'multiply':

            query_intensities = prod(query_components)
            target_intensities = prod(target_components)

            #change gradients
            for key, val in self.lazy_grads1_reweight.items():

                if 'query' in key:
                    aggregated = query_components
                else:
                    aggregated = target_components

                self.lazy_grads1_reweight[key] = val * (aggregated / getattr(self, '_'.join(key.split('_')[:-1] + '_weights')))

        self.query[:,1] = query_intensities
        self.target[:,1] = target_intensities




        
