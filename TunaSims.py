import numpy as np
from numba import njit
import tools_fast
from dataclasses import dataclass
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

@njit
def sigmoid(z):
    
        return 1/(1 + np.exp(-z))

class speedyTuna:

    def __init__(self,
                query_intensity_a: float = None,
                query_intensity_b: float = None,
                target_intensity_a: float = None,
                target_intensity_b: float = None,
                dif_a: float = 0,
                dif_b: float= 1,
                mult_a: float = 0,
                mult_b: float = 1,
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
        self.add_norm_b = add_norm_b
        self.ms2_da = ms2_da
        self.ms2_ppm = ms2_ppm

        self.grad_names = ['dif_a',
                          'dif_b', 
                          'mult_a', 
                          'mult_b', 
                          'add_norm_b',
                          'query_intensity_a',
                          'query_intensity_b',
                          'target_intensity_a',
                          'target_intensity_b' ]
        
        self.grad_vals = np.zeros(9)

    @staticmethod 
    @njit
    def smooth_reweight_grads(array,
                        a,
                        b,
                        grads = True):
        
        """ flexible exponenet simple reweight"""
        
        b_component = np.power(array, b)
        res = a * b_component

        zero_inds = np.logical_or(res <= 0, array == 0)
        res[zero_inds] = 0

        #set a grad
        a_grad = b_component
        a_grad[zero_inds] = 0

        #set b grad
        b_grad = res * np.log(array)
        b_grad[zero_inds] = 0

        return res, a_grad, b_grad
    
    @staticmethod 
    @njit
    def smooth_reweight(array,
                        a,
                        b,
                        grads = True):
        
        """ flexible exponenet simple reweight"""
        
        b_component = np.power(array, b)
        res = a * b_component

        zero_inds = np.logical_or(res <= 0, array == 0)
        res[zero_inds] = 0

        return res
    
    
    @staticmethod
    @njit
    def sub_predict_grads(query,
                    target,
                    q_int_a_grad,
                    q_int_b_grad,
                    t_int_a_grad,
                    t_int_b_grad,
                    dif_a,
                    dif_b,
                    mult_a,
                    mult_b,
                    add_norm_b
                    ):
        
        grad_vals = np.zeros(9)
        
        #generate uncollapsed intensity combining functions
        difs = query - target
        difs_abs = np.abs(difs)
        mults = query * target
        add = query + target

        #generate expanded terms
        add_norm = np.power(add, add_norm_b)
        dif_abs_term = np.power(difs_abs, dif_b) / add_norm
        mult_term = np.power(mults, mult_b) / add_norm
        
        #very messy cacluation of terms, going for efficiency with intermediate results here
        #slight adjustment to take care of infinite grads...these result from no difference and therefore will be set to 0 anyways
        #calcualte gradient for similarity score params of dif and mult a(R -> R)

        #calculate sim parameter gradients
        grad_vals[0] = np.sum(dif_abs_term) #dif_a

        #update dif term
        dif_abs_term = dif_a * dif_abs_term
        b_grad = dif_abs_term * np.log(difs_abs)
        b_grad[np.isnan(b_grad)] = 0

        grad_vals[1] = np.sum(b_grad) #dif_b
            
        grad_vals[2] = np.sum(mult_term) #mult_a

        mult_term = mult_a * mult_term
        b_grad = mult_term * np.log(mults)
        b_grad[np.isnan(b_grad)] = 0

        grad_vals[3] = np.sum(b_grad) #mult_b

        #since all the add norm gradients build on each other, we can gain a speedup
        #chain rule, exponent rule
        raw_score = dif_abs_term + mult_term
        grad_vals[4] = -np.sum(raw_score * np.log(add)) #add norm b

        #calculate component gradients w.r.t. each side of input
        #chain rule
        #exclude indices where dif is 0 b.c. no grad at these points
        dif_grad_q = dif_a * dif_b * np.power(difs_abs, dif_b-2) * difs
        dif_grad_q[np.isinf(dif_grad_q)] = 0
        dif_grad_t = -dif_grad_q

        mult_grad = mult_a * mult_b * np.power(mults, mult_b - 1)
        mult_grad[np.isinf(mult_grad)] = 0
        mult_grad_q = mult_grad * target
        mult_grad_t = mult_grad * query

        #add grad will be the same for query and target
        #chain rule
        #exclude indices where mult is 0 b.c. no grad at these points
        add_grad = add_norm_b * np.power(add, add_norm_b - 1)
        
        #gradients of score w.r.t. query and target...for passing down reweight param grads
        #quotient rule and combining terms
        second_term = (mult_term + dif_abs_term) * add_grad
        query_grad = (mult_grad_q + dif_grad_q - second_term) / add_norm
        target_grad  = (mult_grad_t + dif_grad_t - second_term) / add_norm

        #get the gradient of score w.r.t reweight params
        #chain rule
        grad_vals[5] = np.nansum(q_int_a_grad * query_grad) #query intensity a
        grad_vals[6] = np.nansum(q_int_b_grad * query_grad) #query intensity b

        grad_vals[7] = np.nansum(t_int_a_grad * target_grad) #target intensity a
        grad_vals[8] = np.nansum(t_int_b_grad * target_grad) #target intensity b

        #finally calculate score
        score = sigmoid(np.sum(raw_score))

        #adjust gradients for final sigmoid layer
        #chain rule
        grad_vals = grad_vals * score * (1 - score)

        return score, grad_vals

    @staticmethod
    @njit
    def sub_predict(query,
                    target,
                    dif_a,
                    dif_b,
                    mult_a,
                    mult_b,
                    add_norm_b
                    ):    

        #generate uncollapsed intensity combining functions
        difs = query - target
        difs_abs = np.abs(difs)
        mults = query * target
        add = query + target

        return sigmoid(np.sum(((dif_a * np.power(difs_abs, dif_b)) + 
                                  (mult_a * np.power(mults, mult_b))) / 
                                  np.power(add, add_norm_b)))
    
    def predict(self, query, target, grads = False):
        ''' 
        this function will yield a [0,1] interval similarity prediction
        predict also sets the values of potentially relevant gradint calculation parameters,
        and is therefore analagous to forward pass before backprop
        '''
        
        #set reweighted query and target and update reweight param gradients
        #intensities only from here on out
        if grads:
            query, q_int_a_grad, q_int_b_grad  = speedyTuna.smooth_reweight_grads(query, 
                                                                      self.query_intensity_a, 
                                                                      self.query_intensity_b, 
                                                                      grads = True)
            
            target, t_int_a_grad, t_int_b_grad = speedyTuna.smooth_reweight_grads(target, 
                                                                      self.target_intensity_a, 
                                                                      self.target_intensity_b, 
                                                                      grads = True)
            
            score, grad_vals = speedyTuna.sub_predict_grads(query,
                                                            target,
                                                            q_int_a_grad,
                                                            q_int_b_grad,
                                                            t_int_a_grad,
                                                            t_int_b_grad,
                                                            self.dif_a,
                                                            self.dif_b,
                                                            self.mult_a,
                                                            self.mult_b,
                                                            self.add_norm_b)
            
            self.grad_vals = grad_vals
            return score

        else:
            query = speedyTuna.smooth_reweight(query, 
                                         self.query_intensity_a, 
                                         self.query_intensity_b, 
                                         grads = False)
            
            target = speedyTuna.smooth_reweight(target, 
                                          self.target_intensity_a, 
                                          self.target_intensity_b, 
                                          grads = False)
            
            return speedyTuna.sub_predict(query,
                                          target,
                                          self.dif_a,
                                          self.dif_b,
                                          self.mult_a,
                                          self.mult_b,
                                          self.add_norm_b)
            
        
    def predict_for_dataset(self, dataset):

        res = np.zeros(len(dataset))

        for index, query, target in zip([i for i in range(dataset.shape[0])], dataset['query'], dataset['target']):

            res[index] = self.predict(query, target)

        return res
    
@dataclass
class tunaQuery:

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
    none_prob_int: float = 1
    none_prob_a: float = -1
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

        return self.none_prob_int + self.none_prob_a * max_exp
    
    
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


 
class scoreByQuery2:

    ''' 
    reweights a set of core matches for a given query based on the scores of other potential matches
    '''

    def __init__(self,
                 raw_scores_int: float = 0,
                 raw_scores_a: float = None,
                 raw_scores_b: float = 0,
                 dif_from_next_int: float = 0,
                 dif_from_next_a: float = None,
                 dif_from_next_b: float = 0,
                 dif_from_top_int: float = 0,
                 dif_from_top_a: float = None,
                 dif_from_top_b: float = 0,
                 none_prob_int: float = 1,
                 none_prob_a: float = 1,
                 none_prob_b: float = 1,
                 weight_combine: str = 'add'):
        
        self.raw_scores_int = raw_scores_int
        self.raw_scores_a = raw_scores_a
        self.raw_scores_b = raw_scores_b
        self.dif_from_next_int = dif_from_next_int
        self.dif_from_next_a = dif_from_next_a
        self.dif_from_next_b = dif_from_next_b
        self.dif_from_top_int = dif_from_top_int
        self.dif_from_top_a = dif_from_top_a
        self.dif_from_top_b = dif_from_top_b
        self.none_prob_int = none_prob_int
        self.none_prob_a = none_prob_a
        self.none_prob_b = none_prob_b
        self.weight_combine = weight_combine

        if self.weight_combine == 'add':
            self.raw_scores_a = (self.raw_scores_a or 0)
            self.dif_from_next_a = (self.dif_from_next_a or 0)
            self.dif_from_top_a = (self.dif_from_top_a or 0)

        else:
            self.raw_scores_a = (self.raw_scores_a or 1)
            self.dif_from_next_a = (self.dif_from_next_a or 1)
            self.dif_from_top_a = (self.dif_from_top_a or 1)

        self.grad_names = ['raw_scores_int',
                            'raw_scores_a',
                            'raw_scores_b',
                            'dif_from_next_int',
                            'dif_from_next_a',
                            'dif_from_next_b',
                            'dif_from_top_int',
                            'dif_from_top_a',
                            'dif_from_top_b',
                            'none_prob_int',
                            'none_prob_a',
                            'none_prob_b']
            
        self.grad_vals = np.zeros(len(self.grad_names))

    @njit
    def smooth_reweight_grads(array,
                            intercept,
                            a,
                            b):
        
        """ flexible exponenet simple reweight"""
        
        b_component = np.power(array, b)
        combined = a * b_component
        res = intercept + combined

        return res, np.ones(len(array)), b_component, combined * np.log(array), a * b * np.power(array, b - 1)
    
    @njit
    def smooth_reweight(array,
                        intercept,
                        a,
                        b):
                          
        """ flexible exponenet simple reweight"""


        return intercept + a * np.power(array, b)


    def combine_intensity_weights(self, components, grads):

        
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

    
    def update_none_grad_placeholder(self, a, b, array_val):
        """ 
        intermediate step to make sure none grads are properly chain-ruled
        array_val is all 0s except for the none_prob_index
        """

        for param in ['none_prob_int', 'none_prob_a', 'none_prob_b']:

            self.none_grad_placeholder[param].append(self.grads1[param] * a * b * array_val ** (b-1))
            

    @staticmethod
    @njit
    def sub_predict_grads(scores,
                         match_names,
                         none_prob_int,
                         none_prob_a,
                         none_prob_b,
                         raw_scores_int,
                         raw_scores_a,
                         raw_scores_b,
                         top_from_next_int,
                         top_from_next_a,
                         top_from_next_b,
                         dif_from_top_int,
                         dif_from_top_a,
                         dif_from_top_b,
                         weight_combine):
        
        """ 
        steps:
            1) get prob of not matching any of these labels
            2) reorder intensities by prob of match descending
            3) reweight probabilities according to which transformations  are triggered
            4) consolidate probabilities of match
            5) softmax
        """

        components = np.zeros((3, len(scores) + 1))

        score_grads = np.zeros((3, len(scores) + 1))

        grads = np.zeros((12, len(scores) + 1))

        none_prob, none_grad_int, none_grad_a, none_grad_b, _ = scoreByQuery2.smooth_reweight_grads(np.max(scores),
                                                                                                none_prob_int,
                                                                                                none_prob_a,
                                                                                                none_prob_b)

        #this will optionally set grads based on whether we are in training or inference mode
        scores = np.concatenate(([none_prob], scores))
        match_names = np.concatenate(([None], match_names))

        #negate to get in descending order
        sort_order = np.argsort(-scores)

        #retain the index of the original None
        none_ind_mask = (sort_order == 0).astype(np.int64)

        #sort scores in descending order
        scores = scores[sort_order]
        match_names = match_names[sort_order]

        #set none prob gradients
        grads[0] = [none_ind_mask] * none_grad_int #none_grad_int
        grads[1] = [none_ind_mask] * none_grad_a #none_grad_a
        grads[2] = [none_ind_mask] * none_grad_b #none_grad_b
        
        #update components and their respective gradients
        components[0], grads[3], grads[4], grads[5], score_grads[0] = scoreByQuery2.smooth_reweight_grads(scores,
                                                                                                    raw_scores_int,
                                                                                                    raw_scores_a,
                                                                                                    raw_scores_b)
        
        components[1], grads[6], grads[7], grads[8], score_grads[1] = scoreByQuery2.smooth_reweight_grads(np.zeros(len(scores)) + scores[0] - scores[1], 
                                                                                                top_from_next_int,
                                                                                                top_from_next_a,
                                                                                                top_from_next_b)
        
        components[2], grads[9], grads[10], grads[11], score_grads[2] = scoreByQuery2.smooth_reweight_grads(np.max(scores) - scores, 
                                                                                                  dif_from_top_int,
                                                                                                  dif_from_top_a,
                                                                                                  dif_from_top_b)
        
        #combine wieghts according to specified protocol
        if weight_combine == 'add':

            components = np.sum(components, axis = 0)
            score_grads = np.sum(score_grads, axis = 0)

            grads[0] *= score_grads
            grads[1] *= score_grads
            grads[2] *= score_grads

        else:

            score_grads = np.prod(score_grads, axis = 0)

            grads[0] *= score_grads
            grads[1] *= score_grads
            grads[2] *= score_grads

            sub = components[1] * components[2]
            grads[3] *= sub
            grads[4] *= sub
            grads[5] *= sub

            sub = components[0] * components[2]
            grads[6] *= sub
            grads[7] *= sub
            grads[8] *= sub

            sub = components[0] * components[1]
            grads[9] *= sub
            grads[10] *= sub
            grads[11] *= sub

            components = np.prod(components, axis = 0)

        #softmax
        components = np.exp(components)
        components /= components

        #chain rule
        grads *= components * (1 - components)

        return components, match_names, grads

    @staticmethod
    @njit
    def sub_predict(scores,
                    match_names,
                    none_prob_int,
                    none_prob_a,
                    none_prob_b,
                    raw_scores_int,
                    raw_scores_a,
                    raw_scores_b,
                    top_from_next_int,
                    top_from_next_a,
                    top_from_next_b,
                    dif_from_top_int,
                    dif_from_top_a,
                    dif_from_top_b,
                    weight_combine):
        
        """ 
        steps:
            1) get prob of not matching any of these labels
            2) reorder intensities by prob of match descending
            3) reweight probabilities according to which transformations  are triggered
            4) consolidate probabilities of match
            5) softmax
        """

        components = np.zeros((3, len(scores) + 1))


        none_prob  = scoreByQuery2.smooth_reweight(np.max(scores),
                                                none_prob_int,
                                                none_prob_a,
                                                none_prob_b)

        #this will optionally set grads based on whether we are in training or inference mode
        scores = np.concatenate(([none_prob], scores))
        match_names = np.concatenate(([None], match_names))

        #negate to get in descending order
        sort_order = np.argsort(-scores)

        #retain the index of the original None
        none_ind_mask = (sort_order == 0).astype(np.int64)

        #sort scores in descending order
        scores = scores[sort_order]
        match_names = match_names[sort_order]

        
        #update components and their respective gradients
        components[0] = scoreByQuery2.smooth_reweight(scores,
                                                    raw_scores_int,
                                                    raw_scores_a,
                                                    raw_scores_b)
        
        components[1] = scoreByQuery2.smooth_reweight(np.zeros(len(scores)) + scores[0] - scores[1], 
                                                    top_from_next_int,
                                                    top_from_next_a,
                                                    top_from_next_b)
        
        components[2] = scoreByQuery2.smooth_reweight(np.max(scores) - scores, 
                                                    dif_from_top_int,
                                                    dif_from_top_a,
                                                    dif_from_top_b)
        
        #combine wieghts according to specified protocol
        if weight_combine == 'add':

            components = np.sum(components, axis = 0)
            
        else:

            components = np.prod(components, axis = 0)

        #softmax
        components = np.exp(components)
        components /= components

        return components, match_names

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

        if grads:

            pred_val, self.grads = scoreByQuery2.sub_predict_grads(scores,
                                           match_names,
                                           self.none_prob_int,
                                           self.none_prob_a,
                                           self.none_prob_b,
                                           self.raw_scores_int,
                                           self.raw_scores_a,
                                           self.raw_scores_b,
                                           self.dif_from_next_int,
                                           self.dif_from_next_a,
                                           self.dif_from_next_b,
                                           self.dif_from_next_int,
                                           self.dif_from_top_a,
                                           self.dif_from_top_b,
                                           self.weight_combine)
            
        
        else:

            pred_val  =  scoreByQuery2.sub_predict(scores,
                                           match_names,
                                           self.none_prob_int,
                                           self.none_prob_a,
                                           self.none_prob_b,
                                           self.raw_scores_int,
                                           self.raw_scores_a,
                                           self.raw_scores_b,
                                           self.dif_from_next_int,
                                           self.dif_from_next_a,
                                           self.dif_from_next_b,
                                           self.dif_from_next_int,
                                           self.dif_from_top_a,
                                           self.dif_from_top_b,
                                           self.weight_combine)
       
        return pred_val

