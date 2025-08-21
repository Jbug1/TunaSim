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

class tunaSim:

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
                        b):
        
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
                        b):
        
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

        grad_vals[1] = np.nansum(dif_abs_term * np.log(difs_abs)) #dif_b
            
        grad_vals[2] = np.sum(mult_term) #mult_a

        mult_term = mult_a * mult_term

        grad_vals[3] = np.nansum(mult_term * np.log(mults)) #mult_b

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
            query, q_int_a_grad, q_int_b_grad  = tunaSim.smooth_reweight_grads(query, 
                                                                      self.query_intensity_a, 
                                                                      self.query_intensity_b)
            
            target, t_int_a_grad, t_int_b_grad = tunaSim.smooth_reweight_grads(target, 
                                                                      self.target_intensity_a, 
                                                                      self.target_intensity_b)
            
            score, grad_vals = tunaSim.sub_predict_grads(query,
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
            query = tunaSim.smooth_reweight(query, 
                                         self.query_intensity_a, 
                                         self.query_intensity_b)
            
            target = tunaSim.smooth_reweight(target, 
                                          self.target_intensity_a, 
                                          self.target_intensity_b)
            
            return tunaSim.sub_predict(query,
                                          target,
                                          self.dif_a,
                                          self.dif_b,
                                          self.mult_a,
                                          self.mult_b,
                                          self.add_norm_b)
            
        
    def predict_for_dataset(self, dataset):

        res = np.zeros(dataset.shape[0])

        for index, query, target in zip([i for i in range(dataset.shape[0])], dataset['query'], dataset['target']):

            res[index] = self.predict(query, target)

        return res
    
#jit funcs for tunaQuery

@njit
def smooth_reweight_grads(array,
                        a,
                        b):
    
    """ flexible exponenet simple reweight"""
    
    b_component = np.power(array, b)
    res = a * b_component

    return res, b_component, res * np.log(array)

@njit
def smooth_reweight(array,
                    a,
                    b):
                        
    """ flexible exponenet simple reweight"""


    return a * np.power(array, b)


class tunaQuery:

    ''' 
    reweights a set of core matches for a given query based on the scores of other potential matches
    '''

    def __init__(self,
                 raw_scores_a: float = None,
                 raw_scores_b: float = 0,
                 dif_from_next_a: float = None,
                 dif_from_next_b: float = 0,
                 dif_from_top_a: float = None,
                 dif_from_top_b: float = 0,
                 weight_combine: str = 'add'):
        
        self.raw_scores_a = raw_scores_a
        self.raw_scores_b = raw_scores_b
        self.dif_from_next_a = dif_from_next_a
        self.dif_from_next_b = dif_from_next_b
        self.dif_from_top_a = dif_from_top_a
        self.dif_from_top_b = dif_from_top_b
        self.weight_combine = weight_combine

        if self.weight_combine == 'add':
            self.raw_scores_a = (self.raw_scores_a or 0)
            self.dif_from_next_a = (self.dif_from_next_a or 0)
            self.dif_from_top_a = (self.dif_from_top_a or 0)

        else:
            self.raw_scores_a = (self.raw_scores_a or 1)
            self.dif_from_next_a = (self.dif_from_next_a or 1)
            self.dif_from_top_a = (self.dif_from_top_a or 1)

        self.grad_names = np.array(['raw_scores_a',
                                    'raw_scores_b',
                                    'dif_from_next_a',
                                    'dif_from_next_b',
                                    'dif_from_top_a',
                                    'dif_from_top_b'])


    def predict(self, scores, grads = True):
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

            pred_val, self.grad_vals = tunaQuery.sub_predict_grads(scores,
                                                                    self.raw_scores_a,
                                                                    self.raw_scores_b,
                                                                    self.dif_from_next_a,
                                                                    self.dif_from_next_b,
                                                                    self.dif_from_top_a,
                                                                    self.dif_from_top_b,
                                                                    self.weight_combine)
            
        else:

            pred_val =  tunaQuery.sub_predict(scores,
                                            self.raw_scores_a,
                                            self.raw_scores_b,
                                            self.dif_from_next_a,
                                            self.dif_from_next_b,
                                            self.dif_from_top_a,
                                            self.dif_from_top_b,
                                            self.weight_combine)
       
        return pred_val
    
    @staticmethod
    @njit
    def sub_predict_grads(scores,
                            raw_scores_a,
                            raw_scores_b,
                            top_from_next_a,
                            top_from_next_b,
                            dif_from_top_a,
                            dif_from_top_b,
                            weight_combine):
        
        """ 
        steps:
            1) reweight probabilities according to which transformations  are triggered
            2) consolidate probabilities of match
            3) softmax
        """


        components = np.zeros((3, scores.shape[0]), dtype = np.float64)

        grads = np.zeros((6, scores.shape[0]), dtype = np.float64)

        #update components and their respective gradients
        components[0], grads[0], grads[1] = smooth_reweight_grads(scores,
                                                                 raw_scores_a,
                                                                 raw_scores_b)
        
        components[1], grads[2], grads[3] = smooth_reweight_grads(np.zeros(scores.shape[0]) + scores[0] - scores[1], 
                                                                            top_from_next_a,
                                                                            top_from_next_b)
        
        components[2], grads[4], grads[5] = smooth_reweight_grads(scores[0] - scores, 
                                                                  dif_from_top_a,
                                                                  dif_from_top_b)
        
        #avoid nan
        grads[5][0] = 0

        #combine wieghts according to specified protocol
        if weight_combine == 'add':

            components = np.sum(components, axis = 0)

        else:

            sub = components[1] * components[2]
            grads[0] *= sub
            grads[1] *= sub
            grads[2] *= sub

            sub = components[0] * components[2]
            grads[3] *= sub
            grads[4] *= sub
            grads[5] *= sub

            sub = components[0] * components[1]
            grads[6] *= sub
            grads[7] *= sub
            grads[8] *= sub

            components = components[0] * components[1] * components[2]

        #sigmoid
        components = sigmoid(components)
        sig_grad = components * (1 - components)
        grads *= sig_grad

        return components, grads

    @staticmethod
    @njit
    def sub_predict(scores,
                    raw_scores_a,
                    raw_scores_b,
                    top_from_next_a,
                    top_from_next_b,
                    dif_from_top_a,
                    dif_from_top_b,
                    weight_combine):
        
        """ 
        steps:
            1) reweight probabilities according to which transformations  are triggered
            2) consolidate probabilities of match
            3) softmax
        """

        components = np.zeros((3, scores.shape[0]), dtype = np.float64)

        #update components and their respective gradients
        components[0] = smooth_reweight(scores,
                                        raw_scores_a,
                                        raw_scores_b)
        
        components[1] = smooth_reweight(np.zeros(scores.shape[0]) + scores[0] - scores[1], 
                                        top_from_next_a,
                                        top_from_next_b)
        
        components[2] = smooth_reweight_grads(scores[0] + 1e-7 - scores, 
                                            dif_from_top_a,
                                            dif_from_top_b)
        
        #combine wieghts according to specified protocol
        if weight_combine == 'add':

            components = np.sum(components, axis = 0)

        else:

            components = components[0] * components[1] * components[2]

        return sigmoid(components)

