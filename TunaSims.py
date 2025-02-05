import numpy as np
import scipy
import tools_fast

class TunaSim:

    def tuna_weight_intensity(spectrum,
                          fixed_exp = 0,
                          mz_exp = 0,
                          entropy_exp = 0,
                          offset = -1
                          ):
    
        mz_power_array = np.power(spectrum[:,0], mz_exp)
        intensities = spectrum[:,1]
        
        return np.power(intensities, offset + mz_power_array + fixed_exp + scipy.stats.entropy(intensities) ** entropy_exp)

    def smooth_reweight(array,
                        intercept,
                        a,
                        b,
                        c):
        ''' 
        give new weightings based on array values and passed parameters
        '''
        
        return intercept + a * array + b * np.power(array,2) + c * np.power(array,3)
        
    def sigmoid(z):
    
        return 1/(1 + np.exp(-z))
    
    
class TunaSmoothSim(TunaSim):

    def __init__(self,
                 query_raw_mz_int = 0,
                query_raw_mz_a = 0,
                query_raw_mz_b = 0,
                query_raw_mz_c = 0,
                target_raw_mz_int = 0,
                target_raw_mz_a = 0,
                target_raw_mz_b = 0,
                target_raw_mz_c = 0,
                query_prop_mz_int = 0,
                query_prop_mz_a = 0,
                query_prop_mz_b = 0,
                query_prop_mz_c = 0,
                target_prop_mz_int = 0,
                target_prop_mz_a = 0,
                target_prop_mz_b = 0,
                target_prop_mz_c = 0,
                query_intensity_mz_int = 0,
                query_intensity_mz_a = 0,
                query_intensity_mz_b = 0,
                query_intensity_mz_c = 0,
                target_intensity_int = 0,
                target_intensity_a = 0,
                target_intensity_b = 0,
                target_intensity_c = 0,
                query_norm_int_int = 0,
                query_norm_int_mz_a = 0,
                query_norm_int_mz_b = 0,
                query_norm_int_mz_c = 0,
                target_norm_int_int = 0,
                target_norm_int_a = 0,
                target_norm_int_b = 0,
                target_norm_int_c = 0,
                query_entropy_int = 0,
                query_entropy_a = 0,
                query_entropy_b = 0,
                query_entropy_c = 0,
                target_entropy_int = 0,
                target_entropy_a = 0,
                target_entropy_b = 0,
                target_entropy_c = 0,
                dif_a = 0,
                dif_b = 1,
                mult_a = 0,
                mult_b = 1,
                add_norm_a = 0,
                add_norm_b = 1,
                mult_norm_a = 0,
                mult_norm_b = 1,
                unnormed = 0,
                collapsed = 0,
                expanded = 0,
                match_tolerance = 0.05,
                sim_flip = False,
                zero_clip = True,
                restandardize = True
                ):
        
        super.__init__()
        
        self.query_raw_mz_int  = query_raw_mz_int 
        self.query_raw_mz_a  = query_raw_mz_a 
        self.query_raw_mz_b  = query_raw_mz_b 
        self.query_raw_mz_c  = query_raw_mz_c 
        self.target_raw_mz_int  = target_raw_mz_int 
        self.target_raw_mz_a  = target_raw_mz_a 
        self.target_raw_mz_b  = target_raw_mz_b 
        self.target_raw_mz_c  = target_raw_mz_c 
        self.query_prop_mz_int  = query_prop_mz_int 
        self.query_prop_mz_a  = query_prop_mz_a 
        self.query_prop_mz_b  = query_prop_mz_b 
        self.query_prop_mz_c  = query_prop_mz_c 
        self.target_prop_mz_int  = target_prop_mz_int 
        self.target_prop_mz_a  = target_prop_mz_a 
        self.target_prop_mz_b  = target_prop_mz_b 
        self.target_prop_mz_c  = target_prop_mz_c 
        self.query_intensity_mz_int  = query_intensity_mz_int 
        self.query_intensity_mz_a  = query_intensity_mz_a 
        self.query_intensity_mz_b  = query_intensity_mz_b 
        self.query_intensity_mz_c  = query_intensity_mz_c 
        self.target_intensity_int  = target_intensity_int 
        self.target_intensity_a  = target_intensity_a 
        self.target_intensity_b  = target_intensity_b 
        self.target_intensity_c  = target_intensity_c 
        self.query_norm_int_int  = query_norm_int_int 
        self.query_norm_int_mz_a  = query_norm_int_mz_a 
        self.query_norm_int_mz_b  = query_norm_int_mz_b 
        self.query_norm_int_mz_c  = query_norm_int_mz_c 
        self.target_norm_int_int  = target_norm_int_int 
        self.target_norm_int_a  = target_norm_int_a 
        self.target_norm_int_b  = target_norm_int_b 
        self.target_norm_int_c  = target_norm_int_c 
        self.query_entropy_int  = query_entropy_int 
        self.query_entropy_a  = query_entropy_a 
        self.query_entropy_b  = query_entropy_b 
        self.query_entropy_c  = query_entropy_c 
        self.target_entropy_int  = target_entropy_int 
        self.target_entropy_a  = target_entropy_a 
        self.target_entropy_b  = target_entropy_b 
        self.target_entropy_c  = target_entropy_c 
        self.dif_a  = dif_a 
        self.dif_b  = dif_b 
        self.mult_a  = mult_a 
        self.mult_b  = mult_b 
        self.add_norm_a  = add_norm_a 
        self.add_norm_b  = add_norm_b 
        self.mult_norm_a  = mult_norm_a 
        self.mult_norm_b  = mult_norm_b 
        self.unnormed  = unnormed 
        self.collapsed  = collapsed 
        self.expanded  = expanded 
        self.match_tolerance  = match_tolerance 
        self.sim_flip  = sim_flip 
        self.zero_clip = zero_clip
        self.restandardize = restandardize

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
            return self.sigmoid(unnormed_term + collapsed_term + expanded_term)
        else:
            return 1 - self.sigmoid(unnormed_term + collapsed_term + expanded_term)


    def smooth_reweight(self,
                query,
                target,
                prec_query,
                prec_target):
        ''' 
        This function performs reweighting only
        '''

        #get reweight based on raw mz values
        if self.mz_weights:
            self.query_mz_weights = tools.tuna_smooth_weight(query[:,0], 
                                                                self.query_raw_mz_int,
                                                                self.query_raw_mz_a,
                                                                self.query_raw_mz_b,
                                                                self.query_raw_mz_c,
                                                            )
                            
            self.target_mz_weights = tools.tuna_smooth_weight(target[:,0], 
                                                                self.target_raw_mz_int,
                                                                self.target_raw_mz_a,
                                                                self.target_raw_mz_b,
                                                                self.target_raw_mz_c,
                                                            )
        
        #grab weights for spectra as precursor offset
        if self.query_mz_offset_weights:
            self.query_mz_offset_weights = tools.tuna_smooth_weight(query[:,0] - prec_query, 
                                                                    self.query_prop_mz_int,
                                                                    self.query_prop_mz_a,
                                                                    self.query_prop_mz_b,
                                                                    self.query_prop_mz_c,
                                                                )
            
            self.target_mz_offset_weights = tools.tuna_smooth_weight(target[:,0] - prec_target, 
                                                                    self.target_prop_mz_int, 
                                                                    self.target_prop_mz_a,
                                                                    self.target_prop_mz_b,
                                                                    self.target_prop_mz_c,
                                                                )
        
        #as a function of intensities
        if self.intensity_weights:
            self.query_int_weights = tools.tuna_smooth_weight(query[:,1], 
                                                                self.query_intensity_mz_int,
                                                                self.query_intensity_mz_a,
                                                                self.query_intensity_mz_b,
                                                                self.query_intensity_mz_c,
                                                            )
            
            self.target_int_weights = tools.tuna_smooth_weight(target[:,1],
                                                                self.target_intensity_int, 
                                                                self.target_intensity_a,
                                                                self.target_intensity_b,
                                                                self.target_intensity_c,
                                                            )
        
        #as a funciton of normalized intensities
        if self.normalized_intensity_weights:
            self.query_norm_int_weights = tools.tuna_smooth_weight(query[:,1] /np.sum(query[:,1]), 
                                                            self.query_norm_int_int,
                                                            self.query_norm_int_mz_a,
                                                            self.query_norm_int_mz_b,
                                                            self.query_norm_int_mz_c,
                                                            )
            
            self.target_norm_int_weights = tools.tuna_smooth_weight(target[:,1] / np.sum(target[:,1]), 
                                                    self.target_norm_int_int,
                                                    self.target_norm_int_a,
                                                    self.target_norm_int_b,
                                                    self.target_norm_int_c,
                                                    )
        

        if self.entropy_weights:
            self.query_entropy_weights = tools.tuna_smooth_weight(np.zeros(len(query)) + scipy.stats.entropy(query[:,1]), 
                                                    self.query_entropy_int,
                                                    self.query_entropy_a,
                                                    self.query_entropy_b,
                                                    self.query_entropy_c,
                                                    )
            
            self.target_entropy_weights = tools.tuna_smooth_weight(np.zeros(len(target)) + scipy.stats.entropy(target[:,1]), 
                                                    self.target_entropy_int,
                                                    self.target_entropy_a,
                                                    self.target_entropy_b,
                                                    self.target_entropy_c,
                                                    )
            
        #combine components for intensities and return
        return (self.combine_intensity_weights('query'), self.combine_intensity_weights('target'),0)


        
