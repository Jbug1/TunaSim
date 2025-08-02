import numpy as np
import scipy
import tools_fast
from dataclasses import dataclass

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

    def smooth_reweight(self,
                          name,
                          array,
                          intercept,
                          a,
                          b,
                          grads = True):
        
        """ flexible exponenet simple reweight"""
        
        b_component = np.power(array, (b or 1))
        combined = a * b_component
        res = (intercept or 0) + combined

        #exclude both 0 inds from the original and 0 inds after reweight
        zero_inds = np.logical_or(res <= 0, array == 0)
        res[zero_inds] = 0

        if grads:

            #set intercept grad
            grad = np.ones(len(array))
            grad[zero_inds] = 0
            self.grads1[f'{name}_int'] = grad

            #set a grad
            grad = b_component
            grad[zero_inds] = 0
            self.grads1[f'{name}_a'] = grad

            #set b grad
            grad = combined * np.log(array)
            grad[zero_inds] = 0
            self.grads1[f'{name}_b'] = grad

        return res
    
    @staticmethod
    def sigmoid(z):
    
        return 1/(1 + np.exp(-z))
    

    def combine_intensity_weights(self, 
                                  query_components,
                                  target_components,
                                  query_names,
                                  target_names,
                                  grads):
        
        self.query_components = query_components
        self.target_components = target_components

        if len(query_components) > 1:

            if self.weight_combine == 'add':

                query_components = np.sum(query_components, axis = 0)
                target_components = np.sum(target_components, axis = 0)

            elif self.weight_combine == 'multiply':

                query_components = np.prod(query_components, axis = 0)
                target_components = np.prod(target_components, axis = 0)

                if grads:
                    
                    #update all query grads to account for multiplying combination
                    #for all query components
                    for name, value in zip(query_names, query_components):

                        chain_component = query_components / value
                        for key in [f'{name}_int', f'{name}_a', f'{name}_b']:

                            new_val = self.grads1[key] * chain_component
                            self.grads1[key] = np.nan_to_num(new_val, nan=0.0, posinf=0.0, neginf=0.0)

                    #then for all target components
                    for name, value in zip(target_names, target_components):

                        chain_component = target_components / value
                        for key in [f'{name}_int', f'{name}_a', f'{name}_b']:

                            new_val = self.grads1[key] * chain_component
                            self.grads1[key] = np.nan_to_num(new_val, nan=0.0, posinf=0.0, neginf=0.0)

                    #change gradients
                    # for key, val in self.grads1.items():

                    #     if 'query' in key:
                    #         aggregated = query_components
                    #     else:
                    #         aggregated = target_components

                    #     new_val = val * (aggregated / getattr(self, '_'.join(key.split('_')[:-1]) + '_weights'))
                    #     self.grads1[key] = np.nan_to_num(new_val, nan=0.0, posinf=0.0, neginf=0.0)

        else:
            query_components = query_components[0]
            target_components = target_components[0]

        #self.nonzero indices set here
        self.nonzero_indices = np.where((query_components != 0) | (target_components != 0))[0]

        return query_components[self.nonzero_indices], target_components[self.nonzero_indices]
    
    def smooth_reweight_old(self,
                        name,
                        array,
                        intercept,
                        a,
                        b,
                        c):
        ''' 
        give new weightings based on array values and passed parameters
        '''

        #be mindful of which inds have a gradient if we are clipping
        if self.zero_clip:
            res = (intercept or 0) + (a or 0) * array + (b or 0) * np.power(array,2) + (c or 0) * np.power(array,3)
            zero_inds = np.where(res <= 0)[0]
            res[zero_inds] = 0

        else:
            res = (intercept or 0) + (a or 0) * array + (b or 0) * np.power(array,2) + (c or 0) * np.power(array,3)
            zero_inds = []
        
        if self.set_grad1:

            if intercept is not None:
                grad = np.ones(len(array))
                grad[zero_inds] = 0
                self.grads1[f'{name}_int'] = grad
            
            if a is not None:
                grad = array
                grad[zero_inds] = 0
                self.grads1[f'{name}_a'] = grad

            if b is not None:
                grad = np.power(array, 2)
                grad[zero_inds] = 0
                self.grads1[f'{name}_b'] = grad

            if c is not None:
                grad = np.power(array, 0.5)
                grad[zero_inds] = 0
                self.grads1[f'{name}_c'] = grad
        
        return res
    
    def set_weight_triggers(self):
        ''' 
        look at which parameters are set to determine which calculations are necessary
        '''
        
        
        #set initial state to False, if some input is non-zero, then flip it to True
        #begin under the assumption that we will do no reweighting
        self.weight_triggers_activated = 0

        self.unweighted = True
        for trigger in self.triggers:
            for side in self.sides:

                if side is None:
                    var = trigger
                else:
                    var = f'{side}_{trigger}'

                setattr(self, var, False)
                for variable in ['int','a','b']:

                    if getattr(self, f'{var}_{variable}') is not None:

                        #set the proper side to let us know to compute values and track derivatives
                        #set unweighted to false if at least one is true
                        if side is not None:
                            setattr(self, var, True)

                        else:
                            setattr(self, trigger, True)
                       
                        self.unweighted = False
                        self.weight_triggers_activated +=1

    def predict_for_dataset(self, dataset):

        return dataset.apply(lambda x: self.predict(x['query'], 
                                            x['target'],
                                            x['precquery'], 
                                            x['prectarget'], 
                                            grads = False),
                                            axis=1,
                                            result_type="expand")
    
    def set_reweighted_intensity(self, query, target, prec_query, prec_target, grads = True):
        ''' 
        This function performs reweighting and deriv setting for reweighting
        '''

        #collectors to store all the reweighted arrays
        query_components = list()
        target_components = list()

        query_names = list()
        target_names = list()

        #get reweight based on raw mz values
        if self.query_mz:
            query_components.append(self.smooth_reweight('query_mz', 
                                                         query[:,0]/1000,
                                                        self.query_mz_int,
                                                        self.query_mz_a,
                                                        self.query_mz_b,
                                                        grads = grads))
            
            query_names.append('query_mz')
            
        if self.target_mz:           
            target_components.append(self.smooth_reweight('target_mz',
                                                          target[:,0]/1000, 
                                                            self.target_mz_int,
                                                            self.target_mz_a,
                                                            self.target_mz_b,
                                                            grads = grads))
            
            target_names.append('target_mz')
        
        #grab weights for spectra as precursor offset
        if self.query_mz_offset:
            query_components.append(self.smooth_reweight('query_mz_offset',
                                                                (query[:,0] - prec_query) / 1000, 
                                                                self.query_mz_offset_int,
                                                                self.query_mz_offset_a,
                                                                self.query_mz_offset_b,
                                                                grads = grads))
            
            query_names.append('query_mz_offset')
            
        if self.target_mz_offset:   
            target_components.append(self.smooth_reweight('target_mz_offset',
                                                                 (target[:,0] - prec_target) / 1000, 
                                                                 self.target_mz_offset_int,
                                                                self.target_mz_offset_a,
                                                                self.target_mz_offset_b,
                                                                grads = grads))
            
            target_names.append('target_mz_offset')
        
        #as a function of intensities
        if self.query_intensity:
            query_components.append(self.smooth_reweight('query_intensity',
                                                                query[:,1] / np.max(query[:,1]), 
                                                                self.query_intensity_int,
                                                                self.query_intensity_a,
                                                                self.query_intensity_b,
                                                                grads = grads))
            
            query_names.append('query_intensity')
            
        if self.target_intensity:
            target_components.append(self.smooth_reweight('target_intensity',
                                                                target[:,1] / np.max(target[:,1]),
                                                                self.target_intensity_int,
                                                                self.target_intensity_a,
                                                                self.target_intensity_b,
                                                                grads = grads))
            
            target_names.append('target_intensity')
        
        #as a function of normalized intensities
        if self.query_normalized_intensity:
            query_components.append(self.smooth_reweight('query_normalized_intensity',
                                                                           query[:,1] /np.sum(query[:,1]), 
                                                                            self.query_normalized_intensity_int,
                                                                            self.query_normalized_intensity_a,
                                                                            self.query_normalized_intensity_b,
                                                                            grads = grads))
            
            query_names.append('query_normalized_intensity')
            
        if self.target_normalized_intensity:  
            target_components.append(self.smooth_reweight('target_normalized_intensity',
                                                                            target[:,1] / np.sum(target[:,1]), 
                                                                            self.target_normalized_intensity_int,
                                                                            self.target_normalized_intensity_a,
                                                                            self.target_normalized_intensity_b,
                                                                            grads = grads))
            
            target_names.append('target_normalized_intensity')

       
        #combine components for intensities and return
        return self.combine_intensity_weights(query_components, target_components, query_names, target_names, grads)
    
@dataclass
class ExpandedTuna(TunaSim):
    ''' 
    covers harmonic mean, prob symetric chi square, lorentzian, bhattacharya, matusita
    '''

    query_mz_int: float = None         
    query_mz_a: float = None
    query_mz_b: float = None
    query_mz_c: float = None
    target_mz_int: float = None
    target_mz_a: float = None
    target_mz_b: float = None
    target_mz_c: float = None
    query_mz_offset_int: float = None
    query_mz_offset_a: float = None
    query_mz_offset_b: float = None
    query_mz_offset_c: float = None
    target_mz_offset_int: float = None
    target_mz_offset_a: float = None
    target_mz_offset_b: float = None
    target_mz_offset_c: float = None
    query_intensity_int: float = None
    query_intensity_a: float = None
    query_intensity_b: float = None
    query_intensity_c: float = None
    target_intensity_int: float = None
    target_intensity_a: float = None
    target_intensity_b: float = None
    target_intensity_c: float = None
    query_normalized_intensity_int: float = None
    query_normalized_intensity_a: float = None
    query_normalized_intensity_b: float = None
    query_normalized_intensity_c: float = None
    target_normalized_intensity_int: float = None
    target_normalized_intensity_a: float = None
    target_normalized_intensity_b: float = None
    target_normalized_intensity_c: float = None
    sim_int: float = 0
    dif_a: float = 0
    dif_b: float= 1
    mult_a: float = 0
    mult_b: float = 1
    add_norm_int: float= 0
    add_norm_a: float= 1
    add_norm_b: float = 0
    sigmoid_score: bool = True
    ms2_da: float = 0.05
    ms2_ppm: float = None
    weight_combine: str = 'add'
    zero_clip_reweight: bool = True,

    triggers = ['mz', 'mz_offset', 'intensity', 'normalized_intensity']
    sides = ['query', 'target']

    grads1 = dict()
    grads2 = dict()

    def __post_init__(self):

        self.set_weight_triggers()

    def predict(self, query, target, prec_query = None, prec_target = None, grads = True):
        ''' 
        this function will yield a [0,1] interval similarity prediction
        predict also sets the values of potentially relevant gradint calculation parameters,
        and is therefore analagous to forward pass before backprop
        '''

        #reset gradients if necessary
        if grads:
            self.grads1 = dict()
            self.grads2 = dict()

        #match peaks...will ensure same number for both specs
        #same number is easier for grad but could be worth changing in future
        matched = tools_fast.match_spectrum(query,
                                            target,
                                            ms2_da = self.ms2_da,
                                            ms2_ppm = self.ms2_ppm)
        
        self.matched = matched
        self.query = query
        self.target = target
        
        query = matched[:,:2]
        target = matched[:,[0,2]]
         
        #set reweighted query and target and update reweight param gradients
        #intensities only from here on out
        query, target = self.set_reweighted_intensity(query, 
                                                      target, 
                                                      prec_query, 
                                                      prec_target,
                                                      grads = grads)

        self.re_query = query.copy()
        self.re_target = target.copy()

        #generate uncollapsed intensity combining functions
        difs = query - target
        difs_abs = np.abs(difs)
        mults = query * target
        add = query + target

        self.difs = difs
        self.mults = mults
        self.add = add

        #generate expanded terms
        expanded_difs = self.dif_a * np.power(difs_abs, self.dif_b)
        expanded_mults = self.mult_a * np.power(mults, self.mult_b)
        add_norm = self.add_norm_a * np.power(add, self.add_norm_b)

        self.expanded_difs = expanded_difs
        self.expanded_mults = expanded_mults
        self.mults = mults
        self.add_norm = add_norm

        dif_b = (expanded_difs / add_norm) * np.log(difs_abs)
        dif_b = np.nan_to_num(dif_b, nan=0.0, posinf=0.0, neginf=0.0)

        mult_b = (expanded_mults / add_norm) * np.log(mults)
        mult_b = np.nan_to_num(mult_b, nan=0.0, posinf=0.0, neginf=0.0)
        
        add_b = np.sum(-((expanded_difs + expanded_mults) * np.power(add, -self.add_norm_b) * np.log(add)) / self.add_norm_a)
        add_b = np.nan_to_num(add_b, nan=0.0, posinf=0.0, neginf=0.0)

        if self.sigmoid_score:
            score = self.sigmoid(np.sum((expanded_difs + expanded_mults) / add_norm))
            
        else:
            score = np.sum((expanded_difs + expanded_mults) / add_norm)
        
        #very messy cacluation of terms, going for efficiency with intermediate results here
        #slight adjustment to take care of infinite grads...these result from no difference and therefore will be set to 0 anyways
        #calcualte gradient for similarity score params of dif and mult a(R -> R)
        if grads:

            #calculate sim parameter gradients
            dif_a_grad = np.sum(expanded_difs / (self.dif_a * add_norm))
            mult_a_grad = np.sum(expanded_mults / (self.mult_a * add_norm))

            dif_b_grad = np.sum(dif_b)
            mult_b_grad = np.sum(mult_b)

            add_norm_a_grad = np.sum(-(expanded_difs + expanded_mults)/ self.add_norm_a**2 * add)
            add_norm_b_grad = add_b

            #calculate gradients w.r.t. each side of input
            dif_grad_q = self.dif_a * self.dif_b * np.power(difs_abs, self.dif_b-2) * difs
            dif_grad_q = np.nan_to_num(dif_grad_q, nan=0.0, posinf=0.0, neginf=0.0)
            dif_grad_t = -dif_grad_q

            mult_grad = self.mult_a * self.mult_b * np.power(mults, self.mult_b - 1)
            mult_grad = np.nan_to_num(mult_grad, nan=0.0, posinf=0.0, neginf=0.0)
            mult_grad_q = mult_grad * target
            mult_grad_t = mult_grad * query

            self.mult_grad = mult_grad
            self.mult_grad_q = mult_grad_q

            add_grad = self.add_norm_a * self.add_norm_b * np.power(add, self.add_norm_b - 1)
            add_grad = np.nan_to_num(add_grad, nan=0.0, posinf=0.0, neginf=0.0)
            second_term = (expanded_difs + expanded_mults) * add_grad
            add_norm_square = np.power(add_norm, 2)

            self.second_term = second_term
            self.add_grad = add_grad

            #gradients of score w.r.t. query and target...for passing down reweight param grads
            query_grad = ((dif_grad_q + mult_grad_q) * add_norm - second_term) / add_norm_square
            query_grad = np.nan_to_num(query_grad, nan=0.0, posinf=0.0, neginf=0.0)
            target_grad = ((dif_grad_t + mult_grad_t) * add_norm - second_term) / add_norm_square
            target_grad = np.nan_to_num(target_grad, nan=0.0, posinf=0.0, neginf=0.0)

            self.query_grad = query_grad
            self.target_grad = target_grad

            if self.sigmoid_score:
                sig_grad = score * (1 - score)

            #final step is to calculate grads of score output w.r.t. all reweight params
            for key, value in self.grads1.items():

                if key.split('_')[0] == 'query':
                    side = query_grad
                else:
                    side = target_grad

                #remember to only apply to indices that were not clipped
                score_grad = np.sum(value[self.nonzero_indices] * side)

                if np.any(np.isnan(score_grad)) or np.any(np.isinf(score_grad)):
                    print(key, np.any(np.isnan(side)), np.any(np.isnan(value)))

                #chain rule f'(g(x)) is grad of query or target
                #g'(x) is grad w.r.t. whichever parameter
                if self.sigmoid_score:
                    self.grads1[key] = sig_grad * score_grad
                else:
                    self.grads1[key] = score_grad

            sim_grad_names = ['dif_a','dif_b', 'mult_a', 'mult_b', 'add_norm_a', 'add_norm_b']
            sim_grads = [dif_a_grad, dif_b_grad, mult_a_grad, mult_b_grad, add_norm_a_grad, add_norm_b_grad]

            for key, value in zip(sim_grad_names, sim_grads):
                
                if self.sigmoid_score:
                        
                    self.grads1[key] =  sig_grad * value
                
                else:
                    self.grads1[key] = value

        return score
    
class speedyTuna(TunaSim):

    def __init__(self,
                query_intensity_a: float = None,
                query_intensity_b: float = None,
                target_intensity_a: float = None,
                target_intensity_b: float = None,
                dif_a: float = 0,
                dif_b: float= 1,
                mult_a: float = 0,
                mult_b: float = 1,
                add_norm_int: float= 0,
                add_norm_a: float = 1,
                add_norm_b: float = 0,
                ms2_da: float = 0.05,
                ms2_ppm: float = None):
        
        super().__init__()

        self.query_intensity_a = query_intensity_a
        self.query_intensity_b = query_intensity_b
        self.target_intensity_a = target_intensity_a
        self.target_intensity_b = target_intensity_b
        self.dif_a = dif_a
        self.dif_b = dif_b
        self.mult_a = mult_a
        self.mult_b = mult_b
        self.add_norm_int = add_norm_int
        self.add_norm_a = add_norm_a
        self.add_norm_b = add_norm_b
        self.ms2_da = ms2_da
        self.ms2_ppm = ms2_ppm

        #reset gradients if necessary
        self.grads1 = dict()
        self.grads2 = dict()

        self.grad_names = ['dif_a',
                          'dif_b', 
                          'mult_a', 
                          'mult_b', 
                          'add_norm_int', 
                          'add_norm_a', 
                          'add_norm_b',
                          'query_intensity_a',
                          'query_intensity_b',
                          'target_intensity_a',
                          'target_intensity_b' ]
        
        self.grad_vals = np.zeros(11)
        
    def smooth_reweight(self,
                          array,
                          a,
                          b,
                          grads = True):
        
        """ flexible exponenet simple reweight"""
        
        b_component = np.power(array, b)
        res = a * b_component

        zero_inds = np.logical_or(res <= 0, array == 0)
        res[zero_inds] = 0

        if grads:

            #set a grad
            a_grad = b_component
            a_grad[zero_inds] = 0

            #set b grad
            b_grad = res * np.log(array)
            b_grad[zero_inds] = 0

            return res, a_grad, b_grad

        return res
        
    def predict(self, query, target, grads = True):
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
        
        #only need intensities
        query = matched[:,1] / np.sum(matched[:,1])
        target = matched[:,2] / np.sum(matched[:,2])
        
        #set reweighted query and target and update reweight param gradients
        #intensities only from here on out
        if grads:
            query, q_int_a_grad, q_int_b_grad  = self.smooth_reweight(query, 
                                                                      self.query_intensity_a, 
                                                                      self.query_intensity_b, 
                                                                      grads = True)
            
            target, t_int_a_grad, t_int_b_grad = self.smooth_reweight(target, 
                                                                      self.target_intensity_a, 
                                                                      self.target_intensity_b, 
                                                                      grads = True)
            
        else:
            query = self.smooth_reweight(query, 
                                         self.query_intensity_a, 
                                         self.query_intensity_b, 
                                         grads = False)
            
            target = self.smooth_reweight(target, 
                                          self.target_intensity_a, 
                                          self.target_intensity_b, 
                                          grads = False)
            

        #generate uncollapsed intensity combining functions
        difs = query - target
        difs_abs = np.abs(difs)
        mults = query * target
        add = query + target

        #generate expanded terms
        add_norm = self.add_norm_int + self.add_norm_a * np.power(add, self.add_norm_b)
        dif_abs_term = np.power(difs_abs, self.dif_b) / add_norm
        mult_term = np.power(mults, self.mult_b) / add_norm
        
        #very messy cacluation of terms, going for efficiency with intermediate results here
        #slight adjustment to take care of infinite grads...these result from no difference and therefore will be set to 0 anyways
        #calcualte gradient for similarity score params of dif and mult a(R -> R)
        if grads:

            #calculate sim parameter gradients
            self.grad_vals[0] = np.sum(dif_abs_term) #dif_a

            #update dif term
            dif_abs_term = self.dif_a * dif_abs_term

            self.grad_vals[1] = np.sum(np.nan_to_num(dif_abs_term * np.log(difs_abs), neginf = 0.0)) #dif_b
             
            self.grad_vals[2] = np.sum(mult_term) #mult_a

            mult_term = self.mult_a * mult_term

            self.grad_vals[3]= np.sum(np.nan_to_num(mult_term * np.log(mults), neginf = 0.0)) #mult_b

            #since all the add norm gradients build on each other, we can gain a speedup
            #chain rule
            numerator = dif_abs_term + mult_term

            add_grad = -numerator / add_norm
            self.grad_vals[4] = np.sum(add_grad) #add_norm_int

            add_grad = add_grad * np.power(add, self.add_norm_b)
            self.grad_vals[5] = np.sum(add_grad) #add_norm_a

            add_grad = add_grad * self.add_norm_a * np.log(add)
            self.grad_vals[6] = np.sum(add_grad) #add_norm_b

            #calculate component gradients w.r.t. each side of input
            #chain rule
            #exclude indices where dif is 0 b.c. no grad at these points
            dif_grad_q = self.dif_a * self.dif_b * np.power(difs_abs, self.dif_b-2) * difs
            dif_grad_q[np.isinf(dif_grad_q)] = 0
            dif_grad_t = -dif_grad_q

            mult_grad = self.mult_a * self.mult_b * np.power(mults, self.mult_b - 1)
            mult_grad[np.isinf(mult_grad)] = 0
            mult_grad_q = mult_grad * target
            mult_grad_t = mult_grad * query

            #add grad will be the same for query and target
            #chain rule
            #exclude indices where mult is 0 b.c. no grad at these points
            add_grad = self.add_norm_a * self.add_norm_b * np.power(add, self.add_norm_b - 1)
            
            #gradients of score w.r.t. query and target...for passing down reweight param grads
            #quotient rule and combining terms
            second_term = (mult_term + dif_abs_term) * add_grad
            query_grad = (mult_grad_q + dif_grad_q - second_term) / add_norm
            target_grad  = (mult_grad_t + dif_grad_t - second_term) / add_norm

            #get the gradient of score w.r.t reweight params
            #chain rule
            self.grad_vals[7] = np.nansum(q_int_a_grad * query_grad) #query intensity a
            self.grad_vals[8] = np.nansum(q_int_b_grad * query_grad) #query intensity b

            self.grad_vals[9] = np.nansum(t_int_a_grad * target_grad) #target intensity a
            self.grad_vals[10] = np.nansum(t_int_b_grad * target_grad) #target intensity b

            #finally calculate score
            score = self.sigmoid(np.sum(dif_abs_term + mult_term))

            #adjust gradients for final sigmoid layer
            #chain rule
            self.grad_vals = self.grad_vals * score * (1 - score)

        else:
            score = self.sigmoid(np.sum(self.dif_a * dif_abs_term + self.mult_a * mult_term))

        return score
    
    def predict_for_dataset(self, dataset):

        return dataset.apply(lambda x: self.predict(x['query'], 
                                            x['target'],
                                            grads = False),
                                            axis=1,
                                            result_type="expand")

 
@dataclass
class ScoreByQuery(TunaSim):

    ''' 
    reweights a set of core matches for a given query based on the scores of other potential matches
    '''

    raw_scores_int: float = None
    raw_scores_a: float = None
    raw_scores_b: float = None
    dif_from_next_int: float = None
    dif_from_next_a: float = None
    dif_from_next_b: float = None
    dif_from_top_int: float = None
    dif_from_top_a: float = None
    dif_from_top_b: float = None
    none_prob_int: float = 0
    none_prob_a: float = 1
    none_prob_b: float = 1
    weight_combine: str = 'add'

    triggers = ['raw_scores', 'dif_from_next', 'dif_from_top']
    
    sides = [None]

    none_grad_placeholder = {'none_prob_int': list(),
                             'none_prob_a': list(),
                             'none_prob_b': list()}

    def __post_init__(self):

        self.grads1 = dict()
        self.grads2 = dict()

        self.set_weight_triggers()

    def smooth_reweight(self,
                          name,
                          array,
                          intercept,
                          a,
                          b,
                          grads = True):
        
        """ flexible exponenet simple reweight"""
        
        b_component = np.power(array, b)
        combined = a * b_component
        res = intercept + combined

        if grads:

            #set intercept grad
            grad = np.ones(len(array))
            self.grads1[f'{name}_int'] = grad

            #set a grad
            grad = b_component
            self.grads1[f'{name}_a'] = grad

            #set b grad
            grad = combined * np.log(array)
            self.grads1[f'{name}_b'] = grad

        return res


    def combine_intensity_weights(self, components, grads):

        print(f'{components=}')
        
        if self.weight_combine == 'add':

            if len(components) > 1:

               components = [np.sum(components, axis = 0)]

            if grads:

                for term in ['none_prob_int', 'none_prob_a', 'none_prob_b']:

                    self.grads1[term] = sum(self.none_grad_placeholder[term])

        elif self.weight_combine == 'multiply':

            if len(components) > 1:

                components = np.prod(components, axis = 0)

                #change gradients
                if grads:
                    for key, val in self.grads1.items():

                        if 'none' not in key:

                            new_val = val * (components / getattr(self, '_'.join(key.split('_')[:-1]) + '_weights'))
                            self.grads1[key] = np.nan_to_num(new_val, nan=0.0, posinf=0.0, neginf=0.0)

                    #adjust here so that dimensions match
                    components = [components]
                    for term in ['none_prob_int', 'none_prob_a', 'none_prob_b']:

                        self.grads1[term] = np.prod(self.none_grad_placeholder[term], axis = 0)

        return components[0]

    def get_none_prob(self, max_prob, grads = True):
        """ 
        in a well calibrated model, the none of these probability should track 1 - the top score
        """

        max_exp = max_prob ** self.none_prob_b

        if grads:

            self.grads1['none_prob_int'] = -1
            self.grads1['none_prob_a'] = -max_exp
            self.grads1['none_prob_b'] = -self.none_prob_a * max_exp *np.log(max_prob)

        return 1 - self.none_prob_a * max_exp
    
    def update_none_grad_placeholder(self, a, b, array_val):
        """ 
        intermediate step to make sure none grads are properly chain-ruled
        array_val is all 0s except for the none_prob_index
        """

        for param in ['none_prob_int', 'none_prob_a', 'none_prob_b']:

            self.none_grad_placeholder[param].append(self.grads1[param] * a * b * array_val ** (b-1))
            

    def predict(self, scores, match_names, grads = True):
        """ 
        steps:
            1) get prob of not matching any of these labels
            2) reorder intensities by prob of match descending
            3) reweight probabilities according to which transformations  are triggered
            4) consolidate probabilities of match
            5) clip values below 0
            6) normalize so that all probs sum to 1
        """

        self.none_grad_placeholder = {'none_prob_int': list(),
                                'none_prob_a': list(),
                                'none_prob_b': list()}

        print([self.get_none_prob(np.max(scores), grads = grads)])
                           
        #this will optionally set grads based on whether we are in training or inference mode
        scores = np.concatenate(([self.get_none_prob(np.max(scores), grads = grads)], scores))
        match_names = np.concatenate(([None], match_names))

        print(scores)
        print(self.grads1)

        #negate to get in descending order
        sort_order = np.argsort(-scores)

        #track the index that contains that pesky score for (none of these)
        #this mask will be 1 at the index of the none_prob index and 0 elsewhere
        none_ind_mask = (sort_order == 0).astype(np.int64)

        scores = scores[sort_order]
        match_names = match_names[sort_order]

        print(scores)
        print(match_names)

        #collector to hold all component vectors
        components = list()
        if self.raw_scores:
            raw_score_weights = self.smooth_reweight('raw_scores', 
                                                         scores,
                                                        self.raw_scores_int,
                                                        self.raw_scores_a,
                                                        self.raw_scores_b,
                                                        grads = grads)
            components.append(raw_score_weights)
            
            print(self.grads1)
            
            #update none grads
            if grads:

                self.update_none_grad_placeholder(self.raw_scores_a, 
                                                  self.raw_scores_b, 
                                                  raw_score_weights * none_ind_mask)
            
        if self.dif_from_next: 
            dif_from_next_weights = self.smooth_reweight('dif_from_next',
                                                            np.zeros(len(scores)) + scores[0] - scores[1], 
                                                            self.dif_from_next_int,
                                                            self.dif_from_next_a,
                                                            self.dif_from_next_b,
                                                            grads = grads)          
            components.append(dif_from_next_weights)
            
            print(self.grads1)
            
            #update none grads
            if grads:
                self.update_none_grad_placeholder(self.dif_from_next_a, 
                                                  self.dif_from_next_b, 
                                                  dif_from_next_weights * none_ind_mask)

        if self.dif_from_top: 
            print(f'yerp: {np.max(scores) - scores}')
            dif_from_top_weights = self.smooth_reweight('dif_from_top',
                                                          np.max(scores) - scores, 
                                                            self.dif_from_top_int,
                                                            self.dif_from_top_a,
                                                            self.dif_from_top_b,
                                                            grads = grads) 

            print(f'{dif_from_top_weights=}')         
            components.append(dif_from_top_weights)

            print(self.grads1)
            
            #update none grads
            if grads:
                self.update_none_grad_placeholder(self.dif_from_top_a, 
                                                  self.dif_from_top_b,
                                                  dif_from_top_weights * none_ind_mask)
            
        print(self.none_grad_placeholder)
        reweighted_scores = self.combine_intensity_weights(components = components, grads = grads)
        print(f'{reweighted_scores=}')

        #need sigmoid layer to properly map to [0,1]
        #add the 0.5 offset so that we can actually have vals close to 0 for normalization
        # reweighted_scores = self.sigmoid(reweighted_scores - 0.5)
        # print(f'{reweighted_scores=}')

        # if grads:

        #     sig_grad = reweighted_scores * (1 - reweighted_scores)
        #     for key, value in self.grads1.items():

        #         self.grads1[key] = value * sig_grad

        #finally, normalize to 1 and optionally update gradients
        reweighted_scores = reweighted_scores / np.sum(reweighted_scores)
        print(f'{reweighted_scores=}')

        if grads:

            sum_square_term = np.sum(reweighted_scores) ** 2

            for key, value in self.grads1.items():

                #no need to worry about div 0 since we have sigmoided here
                self.grads1[key] = value * reweighted_scores / sum_square_term

        return reweighted_scores, match_names



@dataclass
class DisconnectedTuna(TunaSim):
    """  
    multiply base and absolute difference base has its own normalizing constant
    """

    query_mz_int: float = None         
    query_mz_a: float = None
    query_mz_b: float = None
    query_mz_c: float = None
    target_mz_int: float = None
    target_mz_a: float = None
    target_mz_b: float = None
    target_mz_c: float = None
    query_mz_offset_int: float = None
    query_mz_offset_a: float = None
    query_mz_offset_b: float = None
    query_mz_offset_c: float = None
    target_mz_offset_int: float = None
    target_mz_offset_a: float = None
    target_mz_offset_b: float = None
    target_mz_offset_c: float = None
    query_intensity_int: float = None
    query_intensity_a: float = None
    query_intensity_b: float = None
    query_intensity_c: float = None
    target_intensity_int: float = None
    target_intensity_a: float = None
    target_intensity_b: float = None
    target_intensity_c: float = None
    query_normalized_intensity_int: float = None
    query_normalized_intensity_a: float = None
    query_normalized_intensity_b: float = None
    query_normalized_intensity_c: float = None
    target_normalized_intensity_int: float = None
    target_normalized_intensity_a: float = None
    target_normalized_intensity_b: float = None
    target_normalized_intensity_c: float = None
    sim_int: float = 0
    dif_a: float = 0
    dif_b: float= 1
    mult_a: float = 0
    mult_b: float = 1
    dif_add_norm_int: float= 0
    dif_add_norm_a: float= 1
    dif_add_norm_b: float = 0
    mult_add_norm_int: float= 0
    mult_add_norm_a: float= 1
    mult_add_norm_b: float = 0
    sigmoid_score: bool = True
    ms2_da: float = 0.05
    ms2_ppm: float = None
    weight_combine: str = 'add'
    zero_clip_reweight: bool = True,


    def __post_init__(self):

        self.grads1 = dict()
        self.grads2 = dict()

        self.set_weight_triggers()

        self.query_mz_weights = None
        self.query_mz_offset_weights = None
        self.query_intensity_weights = None
        self.query_normalized_intensity_weights = None

        self.target_mz_weights = None
        self.target_mz_offset_weights = None
        self.target_intensity_weights = None
        self.target_normalized_intensity_weights = None
    
    def predict(self, query, target, prec_query = None, prec_target = None, grads = True):
        ''' 
        this function will yield a [0,1] interval similarity prediction
        predict also sets the values of potentially relevant gradint calculation parameters,
        and is therefore analagous to forward pass before backprop
        '''

        #reset gradients if necessary
        if grads:
            self.grads1 = dict()
            self.grads2 = dict()

        #match peaks...will ensure same number for both specs
        #same number is easier for grad but could be worth changing in future
        matched = tools_fast.match_spectrum(query,
                                            target,
                                            ms2_da = self.ms2_da,
                                            ms2_ppm = self.ms2_ppm)
        
        self.matched = matched
        self.query = query
        self.target = target
        
        query = matched[:,:2]
        target = matched[:,[0,2]]
         
        #set reweighted query and target and update reweight param gradients
        #intensities only from here on out
        query, target = self.set_reweighted_intensity(query, 
                                                      target, 
                                                      prec_query, 
                                                      prec_target,
                                                      grads = grads)

        self.re_query = query.copy()
        self.re_target = target.copy()

        #generate uncollapsed intensity combining functions
        difs = query - target
        difs_abs = np.abs(difs)
        mults = query * target
        add = query + target

        self.difs = difs
        self.mults = mults
        self.add = add

        #generate expanded terms
        expanded_difs = self.dif_a * np.power(difs_abs, self.dif_b)
        expanded_mults = self.mult_a * np.power(mults, self.mult_b)
        dif_add_norm = self.dif_add_norm_a * np.power(add, self.dif_add_norm_b)
        mult_add_norm = self.mult_add_norm_a * np.power(add, self.mult_add_norm_b)

        dif_b = (expanded_difs / dif_add_norm) * np.log(difs_abs)
        dif_b = np.nan_to_num(dif_b, nan=0.0, posinf=0.0, neginf=0.0)

        mult_b = (expanded_mults / mult_add_norm) * np.log(mults)
        mult_b = np.nan_to_num(mult_b, nan=0.0, posinf=0.0, neginf=0.0)
        
        dif_add_b = np.sum(-((expanded_difs) * np.power(add, -self.dif_add_norm_b) * np.log(add)) / self.dif_add_norm_a)
        dif_add_b = np.nan_to_num(dif_add_b, nan=0.0, posinf=0.0, neginf=0.0)

        mult_add_b = np.sum(-((expanded_mults) * np.power(add, -self.mult_add_norm_b) * np.log(add)) / self.mult_add_norm_a)
        mult_add_b = np.nan_to_num(mult_add_b, nan=0.0, posinf=0.0, neginf=0.0)

        if self.sigmoid_score:
            score = self.sigmoid(np.sum(expanded_difs / dif_add_norm + expanded_mults / mult_add_norm))
            
        else:
            score = np.sum(np.sum(expanded_difs / dif_add_norm + expanded_mults / mult_add_norm))
        
        #very messy cacluation of terms, going for efficiency with intermediate results here
        #slight adjustment to take care of infinite grads...these result from no difference and therefore will be set to 0 anyways
        #calcualte gradient for similarity score params of dif and mult a(R -> R)

        if grads:

            dif_a_grad = np.sum(expanded_difs / (self.dif_a * dif_add_norm))
            mult_a_grad = np.sum(expanded_mults / (self.mult_a * mult_add_norm))

            dif_b_grad = np.sum(dif_b)
            mult_b_grad = np.sum(mult_b)

            dif_add_norm_a_grad = np.sum(-(expanded_difs)/ self.dif_add_norm_a ** 2 * add)
            dif_add_norm_b_grad = dif_add_b

            mult_add_norm_a_grad = np.sum(-(expanded_mults)/ self.mult_add_norm_a ** 2 * add)
            mult_add_norm_b_grad = mult_add_b

            dif_grad_q = self.dif_a * self.dif_b * np.power(difs_abs, self.dif_b-2) * difs
            dif_grad_q = np.nan_to_num(dif_grad_q, nan=0.0, posinf=0.0, neginf=0.0)
            dif_grad_t = -dif_grad_q

            mult_grad = self.mult_a * self.mult_b * np.power(mults, self.mult_b - 1)
            mult_grad = np.nan_to_num(mult_grad, nan=0.0, posinf=0.0, neginf=0.0)
            mult_grad_q = mult_grad * target
            mult_grad_t = mult_grad * query

            dif_add_grad = self.dif_add_norm_a * self.dif_add_norm_b * np.power(add, self.dif_add_norm_b - 1)
            dif_add_grad = np.nan_to_num(dif_add_grad, nan=0.0, posinf=0.0, neginf=0.0)
            dif_second_term = (expanded_difs) * dif_add_grad
            dif_add_norm_square = np.power(dif_add_norm, 2)

            mult_add_grad = self.mult_add_norm_a * self.mult_add_norm_b * np.power(add, self.mult_add_norm_b - 1)
            mult_add_grad = np.nan_to_num(mult_add_grad, nan=0.0, posinf=0.0, neginf=0.0)
            mult_second_term = (expanded_difs) * dif_add_grad
            mult_add_norm_square = np.power(mult_add_norm, 2)

            #gradients of score w.r.t. query and target...for passing down reweight param grads
            query_grad = (dif_grad_q * dif_add_norm - dif_second_term) / dif_add_norm_square + (mult_grad_q * mult_add_norm - mult_second_term) / mult_add_norm_square
            query_grad = np.nan_to_num(query_grad, nan=0.0, posinf=0.0, neginf=0.0)
            target_grad = (dif_grad_t * dif_add_norm - dif_second_term) / dif_add_norm_square + (mult_grad_t * mult_add_norm - mult_second_term) / mult_add_norm_square
            target_grad = np.nan_to_num(target_grad, nan=0.0, posinf=0.0, neginf=0.0)

            if self.sigmoid_score:
                sig_grad = score * (1 - score)

            #final step is to calculate grads of score output w.r.t. all reweight params
            for key, value in self.grads1.items():

                if key.split('_')[0] == 'query':
                    side = query_grad
                else:
                    side = target_grad

                #remember to only apply to indices that were not clipped
                score_grad = np.sum(value[self.nonzero_indices] * side)

                if np.any(np.isnan(score_grad)) or np.any(np.isinf(score_grad)):
                    print(key, np.any(np.isnan(side)), np.any(np.isnan(value)))

                #chain rule f'(g(x)) is grad of query or target
                #g'(x) is grad w.r.t. whichever parameter
                if self.sigmoid_score:
                    self.grads1[key] = sig_grad * score_grad
                else:
                    self.grads1[key] = score_grad

            sim_grad_names = ['dif_a', 
                              'dif_b', 
                              'mult_a', 
                              'mult_b', 
                              'dif_add_norm_a', 
                              'dif_add_norm_b',
                              'mult_add_norm_a', 
                              'mult_add_norm_b']
            
            sim_grads = [dif_a_grad, 
                         dif_b_grad, 
                         mult_a_grad, 
                         mult_b_grad, 
                         dif_add_norm_a_grad, 
                         dif_add_norm_b_grad,
                         mult_add_norm_a_grad, 
                         mult_add_norm_b_grad]

            for key, value in zip(sim_grad_names, sim_grads):
                
                if self.sigmoid_score:
                        
                    self.grads1[key] =  sig_grad * value

                else:

                    self.grads1[key] =  value

        return score









    

    

    

        
