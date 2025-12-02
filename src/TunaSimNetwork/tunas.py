import numpy as np
from numba import njit
from scipy.stats import entropy

@njit
def sigmoid(z):
    
    return 1/(1 + np.exp(-z))

#jit funcs for reweighting intensities
@njit
def smooth_reweight_grads(array,
                        a,
                        b):
    
    """ flexible exponenet simple reweight"""
    
    b_component = np.power(array, b)
    res = a * b_component

    return res, b_component[0], res[0] * np.log(np.float64(array[0] + 1e-7))

@njit
def smooth_reweight(array,
                    a,
                    b):
                        
    """ flexible exponenet simple reweight"""


    return a * np.power(array, b)


class tunaSim:

    def __init__(self,
                query_intensity_a: float = None,
                query_intensity_b: float = None,
                target_intensity_a: float = None,
                target_intensity_b: float = None,
                sim_a: float = 0,
                sim_b: float= 1,
                add_norm_b: float = 0,
                ms2_da: float = 0.05,
                ms2_ppm: float = None):
        

        self.query_intensity_a = query_intensity_a
        self.query_intensity_b = query_intensity_b
        self.target_intensity_a = target_intensity_a
        self.target_intensity_b = target_intensity_b
        self.sim_a = sim_a
        self.sim_b = sim_b
        self.add_norm_b = add_norm_b
        self.ms2_da = ms2_da
        self.ms2_ppm = ms2_ppm

        self.grad_names = ['sim_a',
                           'sim_b',
                           'query_intensity_a',
                           'query_intensity_b',
                           'target_intensity_a',
                           'target_intensity_b' ]
        
        self.grad_vals = np.zeros(6)

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
    
    def predict(self, query, target, grads = False):
        ''' 
        this function will yield a [0,1] interval similarity prediction
        predict also sets the values of potentially relevant gradint calculation parameters,
        and is therefore analagous to forward pass before backprop
        '''
        
        #set reweighted query and target and update reweight param gradients
        if grads:
            query, q_int_a_grad, q_int_b_grad  = tunaSim.smooth_reweight_grads(query, 
                                                                      self.query_intensity_a, 
                                                                      self.query_intensity_b)
            
            target, t_int_a_grad, t_int_b_grad = tunaSim.smooth_reweight_grads(target, 
                                                                      self.target_intensity_a, 
                                                                      self.target_intensity_b)
            
            score, self.grad_vals = self.sub_predict_grads(query,
                                                            target,
                                                            q_int_a_grad,
                                                            q_int_b_grad,
                                                            t_int_a_grad,
                                                            t_int_b_grad,
                                                            self.sim_a,
                                                            self.sim_b)
            
            return score

        else:
            query = tunaSim.smooth_reweight(query, 
                                         self.query_intensity_a, 
                                         self.query_intensity_b)
            
            target = tunaSim.smooth_reweight(target, 
                                          self.target_intensity_a, 
                                          self.target_intensity_b)
            
            return self.sub_predict(query,
                                    target,
                                    self.sim_a,
                                    self.sim_b)
            
        
    def predict_for_dataset(self, dataset):

        res = np.zeros(dataset.shape[0])

        for index, query, target in zip([i for i in range(dataset.shape[0])], dataset['query'], dataset['target']):

            res[index] = self.predict(query, target)

        return res


class tunaDif(tunaSim):
    """
       tunaSim child class for similarities that rely on absolute dif in intensities
    """

    def __init__(self,
                query_intensity_a: float = None,
                query_intensity_b: float = None,
                target_intensity_a: float = None,
                target_intensity_b: float = None,
                sim_a: float = 0,
                sim_b: float= 1,
                ms2_da: float = 0.05,
                ms2_ppm: float = None):

        super().__init__(query_intensity_a = query_intensity_a,
                         query_intensity_b = query_intensity_b,
                         target_intensity_a = target_intensity_a,
                         target_intensity_b = target_intensity_b,
                         sim_a = sim_a,
                         sim_b = sim_b,
                         ms2_da = ms2_da,
                         ms2_ppm = ms2_ppm)
        
        self.grad_names = ['sim_a',
                          'sim_b',
                          'query_intensity_a',
                          'query_intensity_b',
                          'target_intensity_a',
                          'target_intensity_b' ]
        
        self.grad_vals = np.zeros(6)

    @staticmethod
    @njit
    def sub_predict_grads(query,
                    target,
                    q_int_a_grad,
                    q_int_b_grad,
                    t_int_a_grad,
                    t_int_b_grad,
                    dif_a,
                    dif_b):
        """
        Calculate the similarity given some terms and retain gradient values
        """
        
        grad_vals = np.zeros(6)
        
        #generate uncollapsed intensity combining functions
        difs = query - target
        difs_abs = np.abs(difs)

        #generate expanded term
        dif_abs_term = np.power(difs_abs, dif_b)
        
        #very messy cacluation of terms, going for efficiency with intermediate results here
        #slight adjustment to take care of infinite grads...these result from no difference and therefore will be set to 0 anyways
        #calcualte gradient for similarity score params of dif(R -> R)

        #calculate sim parameter gradients
        grad_vals[0] = np.sum(dif_abs_term) #dif_a

        #update dif term
        dif_abs_term = dif_a * dif_abs_term

        grad_vals[1] = np.nansum(dif_abs_term * np.log(difs_abs)) #dif_b

        #calculate component gradients w.r.t. each side of input
        #chain rule
        #exclude indices where dif is 0 b.c. no grad at these points
        dif_grad_q = dif_a * dif_b * np.power(difs_abs, dif_b - 2) * difs
        dif_grad_q[np.isinf(dif_grad_q)] = 0
        dif_grad_t = -dif_grad_q
        
        #gradients of score w.r.t. query and target...for passing down reweight param grads
        #quotient rule and combining terms
        second_term = (dif_abs_term)
        query_grad = (dif_grad_q - second_term)
        target_grad  = (dif_grad_t - second_term)

        #get the gradient of score w.r.t reweight params
        #chain rule
        grad_vals[2] = np.nansum(q_int_a_grad * query_grad) #query intensity a
        grad_vals[3] = np.nansum(q_int_b_grad * query_grad) #query intensity b

        grad_vals[4] = np.nansum(t_int_a_grad * target_grad) #target intensity a
        grad_vals[5] = np.nansum(t_int_b_grad * target_grad) #target intensity b

        #finally calculate score
        score = sigmoid(np.sum(dif_abs_term))

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
                    add_norm_b
                    ): 
        """ 
        When no gradients are necessary, simply implement the sim function given 
        some terms
        """   

        return sigmoid(np.sum(((dif_a * np.power(np.abs(query - target), dif_b)) / 
                               np.power(query + target, add_norm_b))))


class tunaMult(tunaSim):
    """
    tunaSim child class for similarities thatrely on multiplying intensities
    """

    def __init__(self,
                query_intensity_a: float = None,
                query_intensity_b: float = None,
                target_intensity_a: float = None,
                target_intensity_b: float = None,
                sim_a: float = 0,
                sim_b: float= 1,
                ms2_da: float = 0.05,
                ms2_ppm: float = None):

        super().__init__(query_intensity_a = query_intensity_a,
                            query_intensity_b = query_intensity_b,
                            target_intensity_a = target_intensity_a,
                            target_intensity_b = target_intensity_b,
                            sim_a = sim_a,
                            sim_b = sim_b,
                            ms2_da = ms2_da,
                            ms2_ppm = ms2_ppm)
        
        self.grad_names = ['sim_a',
                            'sim_b',
                            'query_intensity_a',
                            'query_intensity_b',
                            'target_intensity_a',
                            'target_intensity_b' ]
        
        self.grad_vals = np.zeros(6)

    @staticmethod
    @njit
    def sub_predict_grads(query,
                    target,
                    q_int_a_grad,
                    q_int_b_grad,
                    t_int_a_grad,
                    t_int_b_grad,
                    mult_a,
                    mult_b):
        
        grad_vals = np.zeros(6)
        
        #generate uncollapsed intensity combining functions
        mults = query * target

        #generate expanded terms
        mult_term = np.power(mults, mult_b)
        
        #very messy cacluation of terms, going for efficiency with intermediate results here
        #slight adjustment to take care of infinite grads...these result from no difference and therefore will be set to 0 anyways
        #calcualte gradient for similarity score params of dif and mult a(R -> R)
        grad_vals[0] = np.sum(mult_term) #mult_a

        mult_term = mult_a * mult_term

        grad_vals[1] = np.nsum(mult_term * np.log(mults)) #mult_b

        #calculate component gradients w.r.t. each side of input
        #chain rule
        #exclude indices where dif is 0 b.c. no grad at these points
        mult_grad = mult_a * mult_b * np.power(mults, mult_b - 1)
        mult_grad[np.isinf(mult_grad)] = 0
        mult_grad_q = mult_grad * target
        mult_grad_t = mult_grad * query
        
        #gradients of score w.r.t. query and target...for passing down reweight param grads
        #quotient rule and combining terms
        query_grad = (mult_grad_q - mult_term)
        target_grad  = (mult_grad_t - mult_term)

        #get the gradient of score w.r.t reweight params
        #chain rule
        grad_vals[2] = np.nansum(q_int_a_grad * query_grad) #query intensity a
        grad_vals[3] = np.nansum(q_int_b_grad * query_grad) #query intensity b

        grad_vals[4] = np.nansum(t_int_a_grad * target_grad) #target intensity a
        grad_vals[5] = np.nansum(t_int_b_grad * target_grad) #target intensity b

        #finally calculate score
        score = sigmoid(np.sum(mult_term))

        #adjust gradients for final sigmoid layer
        #chain rule
        grad_vals = grad_vals * score * (1 - score)

        return score, grad_vals

    @staticmethod
    @njit
    def sub_predict(query,
                    target,
                    sim_a,
                    sim_b,
                    add_norm_b
                    ):    

        return sigmoid(np.sum(((sim_a * np.power(query * target, sim_b))) / np.power(query + target, add_norm_b)))
    

class baseShellSim:

    def __init__(self,
                 sim_func):
        
        self.sim_func = sim_func

    def predict_for_dataset(self, dataset):

        res = np.zeros(dataset.shape[0])

        for index, query, target in zip([i for i in range(dataset.shape[0])], dataset['query'], dataset['target']):

            res[index] = self.predict(query, target)

        return res
    
    def predict(self,
                query,
                target):
        
        return self.sim_func(query, target)
    

        