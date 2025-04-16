import numpy as np
import scipy
import tools_fast
from dataclasses import dataclass
from math import prod

import warnings
warnings.filterwarnings("ignore")

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

    
@dataclass
class ExpandedTuna(TunaSim):
    ''' 
    covers harmonic mean, prob symetric chi square, lorentzian, bhattacharya, matusita
    '''

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
    dif_int: float = 0
    dif_a: float = 0
    dif_b: float= 1
    mult_int: float = 0
    mult_a: float = 0
    mult_b: float = 1
    add_norm_int: float= 0
    add_norm_a: float= 1
    add_norm_b: float = 0
    match_tolerance: float = 0.05
    sigmoid_score: bool = True
    ms2_da: float = 0.05
    ms2_ppm: float = None
    weight_combine: str = 'add'

    def __post_init__(self):

        self.grads1_int_param = dict()
        self.grads2_int_param = dict()
        self.grads1_agg_int = dict()
        self.grads2_agg_int = dict()
        self.grads1_score_agg = dict()
        self.grads2_score_agg = dict()

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
                self.grads1_int_param[f'{name}_int'] = np.ones(len(array))
            
            if a != 0:
                self.grads1_int_param[f'{name}_a'] = array

            if b != 0:
                self.grads1_int_param[f'{name}_b'] = array ** 2

            if c != 0:
                self.grads1_int_param[f'{name}_c'] = array ** 3
        
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


    def predict(self, query, target):

        ''' 
        this function will yield a [0,1] interval similarity prediction
        predict also sets the values of potentially relevant gradint calculation parameters,
        and is therefore analagous to forward pass before backprop
        '''

        #match peaks...will ensure same number for both specs
        #same number is easier for grad but could be worth changing in future
        matched = tools_fast.match_spectrum(query,
                                            target,
                                            ms2_da = self.ms2_da,
                                            ms2_ppm = self.ms2_ppm)
        
        query = matched[:,:2]
        target = matched[:,[0,2]]
         
        #set reweighted query and target and update reweight param gradients
        #intensities only from here on out
        query, target = self.set_reweighted_intensity(query, target)

        #generate uncollapsed intensity combining functions
        difs = query - target
        difs_abs = np.abs(difs)
        mults = query * target
        add = query + target

        #generate expanded terms
        expanded_difs = self.dif_a * np.power(difs_abs, self.dif_b)
        expanded_mults = self.mult_a * np.power(mults, self.mult_b)
        add_norm = self.add_norm_a * np.power(add, self.add_norm_b)

        #calcualte gradient for similarity score params of dif and mult a(R -> R)
        self.grads1_agg_int['dif_a'] = np.sum(expanded_difs / (self.dif_a * add_norm))
        self.grads1_agg_int['mult_a'] = np.sum(expanded_mults / (self.mult_a * add_norm))

        self.grads1_agg_int['dif_b'] = self.dif_a * np.sum((difs_abs ** self.dif_b) * np.log(difs_abs) / add_norm)
        self.grads1_agg_int['mult_b'] = self.mult_a * np.sum((mults ** self.mult_b) * np.log(mults) / add_norm)
        
        self.grads1_agg_int['add_a'] = np.sum(-(expanded_difs + expanded_mults)/ self.add_norm_a**2 * add)
        self.grads1_agg_int['add_b'] = np.sum(-((expanded_difs + expanded_mults) * np.power(add, -self.add_norm_b) * np.log(add)) / self.add_norm_a)

        #very messy cacluation of terms
        #going for efficiency with intermediate results here
        dif_grad_q = self.dif_a * self.dif_b * np.power(difs_abs, self.dif_b-2) * difs
        dif_grad_t = self.dif_a * self.dif_b * np.power(difs_abs, self.dif_b-2) * -difs
        mult_grad = self.mult_a * self.mult_b * np.power(mults, self.mult_b - 1)
        mult_grad_q = mult_grad * target
        mult_grad_t = mult_grad * query
        add_grad = self.add_norm_a * self.add_norm_b * np.power(add, self.add_norm_b - 1)
        second_term = (expanded_difs + expanded_mults) * add_grad
        add_norm_square = np.power(add_norm, 2)

        #gradients w.r.t. query and target...for passing down reweight param grads
        query_grad = ((dif_grad_q + mult_grad_q) * add_norm - second_term) / add_norm_square
        target_grad = ((dif_grad_t + mult_grad_t) * add_norm - second_term) / add_norm_square

        #final step is to calculate grads of score output w.r.t. all reweight params
        for key, value in self.grads1_int_param.items():

            if key.split('_')[0] == 'query':
                side = query_grad
            else:
                side = target_grad

            #chain rule f'(g(x)) is grad of query or target
            #g'(x) is grad w.r.t. whichever parameter
            if self.sigmoid_score:
                sig_grad = self.sigmoid(np.sum(value * side))
                self.grads1_score_agg[key] = sig_grad * (1 - sig_grad)
            else:
                self.grads1_score_agg[key] = np.sum(value * side)

        if self.sigmoid_score:
            for key, value in self.grads1_agg_int.items():

                    sig_grad = self.sigmoid(self.grads1_agg_int[key])
                    self.grads1_score_agg[key] =  sig_grad * (1 - sig_grad)
        else:
            self.grads1_score_agg.update(self.grads1_agg_int)

        if self.sigmoid_score:
            return self.sigmoid(np.sum((expanded_difs + expanded_mults) / add_norm))
        else:
            return np.sum((expanded_difs + expanded_mults) / add_norm)


    def set_reweighted_intensity(self, query, target):
        ''' 
        This function performs reweighting and deriv setting for reweighting
        '''

        #roll with original query if no reweighting is required
        if self.unweighted:
            return

        #get reweight based on raw mz values
        if self.mz_query:
            self.query_mz_weights = self.smooth_reweight('query_mz', 
                                                         query[:,0],
                                                        self.query_mz_int,
                                                        self.query_mz_a,
                                                        self.query_mz_b,
                                                        self.query_mz_c,
                                                        )
            
        if self.mz_target:           
            self.target_mz_weights = self.smooth_reweight('target_mz',
                                                          target[:,0], 
                                                            self.target_mz_int,
                                                            self.target_mz_a,
                                                            self.target_mz_b,
                                                            self.target_mz_c,
                                                            )
        
        #grab weights for spectra as precursor offset
        if self.mz_offset_query:
            self.query_mz_offset_weights = self.smooth_reweight('query_mz_offset',
                                                                query[:,0] - self.prec_query, 
                                                                self.query_mz_offset_int,
                                                                self.query_mz_offset_a,
                                                                self.query_mz_offset_b,
                                                                self.query_mz_offset_c,
                                                                )
            
        if self.mz_offset_target:   
            self.target_mz_offset_weights = self.smooth_reweight('target_mz_offset',
                                                                 target[:,0] - self.prec_target, 
                                                                 self.target_mz_offset_int,
                                                                self.target_mz_offset_a,
                                                                self.target_mz_offset_b,
                                                                self.target_mz_offset_c,
                                                                )
        
        #as a function of intensities
        if self.intensity_query:
            self.query_intensity_weights = self.smooth_reweight('query_intensity',
                                                                query[:,1], 
                                                                self.query_intensity_int,
                                                                self.query_intensity_a,
                                                                self.query_intensity_b,
                                                                self.query_intensity_c,
                                                            )
            
        if self.intensity_target:
            self.target_intensity_weights = self.smooth_reweight('target_intensity',
                                                                target[:,1],
                                                                self.target_intensity_int,
                                                                self.target_intensity_a,
                                                                self.target_intensity_b,
                                                                self.target_intensity_c,
                                                            )
        
        #as a function of normalized intensities
        if self.normalized_intensity_query:
            self.query_normalized_intensity_weights = self.smooth_reweight('query_normalized_intensity',
                                                                           query[:,1] /np.sum(query[:,1]), 
                                                                            self.query_normalized_intensity_int,
                                                                            self.query_normalized_intensity_a,
                                                                            self.query_normalized_intensity_b,
                                                                            self.query_normalized_intensity_c,
                                                                            )
            
        if self.normalized_intensity_target:  
            self.target_normalized_intensity_weights = self.smooth_reweight('target_normalized_intensity',
                                                                            target[:,1] / np.sum(target[:,1]), 
                                                                            self.target_normalized_intensity_int,
                                                                            self.target_normalized_intensity_a,
                                                                            self.target_normalized_intensity_b,
                                                                            self.target_normalized_intensity_c,
                                                                            )

        if self.entropy_query:
            self.query_entropy_weights = self.smooth_reweight('query_entropy',
                                                              np.zeros(len(query)) + scipy.stats.entropy(query[:,1]), 
                                                              self.query_entropy_int,
                                                              self.query_entropy_a,
                                                              self.query_entropy_b,
                                                              self.query_entropy_c,
                                                              )
            
        if self.entropy_target:
            self.target_entropy_weights = self.smooth_reweight('target_entropy',
                                                               np.zeros(len(target)) + scipy.stats.entropy(target[:,1]), 
                                                               self.target_entropy_int,
                                                               self.target_entropy_a,
                                                               self.target_entropy_b,
                                                               self.target_entropy_c,
                                                               )
            
        #combine components for intensities and return
        return self.combine_intensity_weights()

    
    def combine_intensity_weights(self):

        query_components = [self.query_mz_weights,
                            self.query_mz_offset_weights,
                            self.query_intensity_weights,
                            self.query_normalized_intensity_weights,
                            self.query_entropy_weights]
        
        target_components = [self.target_mz_weights,
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
            for key, val in self.grads1_int_param.items():

                if 'query' in key:
                    aggregated = query_intensities
                else:
                    aggregated = target_intensities

                self.grads1_int_param[key] = val * (aggregated / getattr(self, '_'.join(key.split('_')[:-1] + '_weights')))

        return query_intensities, target_intensities



def yool():
    #depending on collapse and expand terms, consolidate or don't to some degree
        if self.collapsed != 0:

            #collapse all terms an
            difs_abs = np.sum(difs_abs)
            mults = np.sum(mults)
            add_norm = np.sum(add)
            mult_norm = np.sum(query * target)

            collapsed_difs = self.dif_a * np.sum(np.abs(query - target)) ** self.dif_b
            collapsed_mults = self.mult_a * np.sum(query * target) ** self.mult_b
            collapsed_term = self.collapsed * (collapsed_difs + collapsed_mults) / (np.sum(mult_norm) + np.sum(add_norm)) 

        #some metrics are expressed as sim measures
        return self.sigmoid(self.sig_factor *(expanded_term))

        
