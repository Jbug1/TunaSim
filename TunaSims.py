import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import tools
from scipy import stats
from scipy.optimize import minimize as mini
from scipy.optimize import approx_fprime as approx

def tuna_distance(query,
                  target,
                  prec_query,
                  prec_target,
                  query_max_mz_fix = 0,
                  query_max_mz_var = 0,
                  query_fixed_noise = 0,
                  query_var_noise = 0,
                  query_da_thresh = 0.05,
                  target_max_mz_fix = 0,
                  target_max_mz_var = 0,
                  target_fixed_noise = 0,
                  target_var_noise = 0,
                  target_da_thresh = 0.05,
                  query_fixed_power = 0,
                  query_mz_power = 0,
                  query_ent_power = 0,
                  target_fixed_power = 0,
                  target_mz_power = 0,
                  target_ent_power = 0,
                  match_tolerance = 0.05,
                  dif_a = 0,
                  dif_b = 0,
                  mult_a = 0,
                  mult_b = 1,
                  add_norm_a = 0,
                  add_norm_b = 1,
                  mult_norm_a = 0,
                  mult_norm_b = 1,
                  collapsed = 0,
                  expanded = 0
                  ):
    
    #get precursor based params
    max_mz_query = tools.get_max_mz(prec_query, query_max_mz_fix, query_max_mz_var)
    max_mz_target = tools.get_max_mz(prec_target, target_max_mz_fix, target_max_mz_var)

    #clean spectra
    query = tools.tuna_clean_spectrum(query,
                                 max_mz = max_mz_query,
                                 ms2_da = query_da_thresh,
                                 noise_removal_fixed = query_fixed_noise,
                                 noise_removal_var = query_var_noise,
                                 )
    
    target = tools.tuna_clean_spectrum(target,
                                 max_mz = max_mz_target,
                                 ms2_da = target_da_thresh,
                                 noise_removal_fixed = target_fixed_noise,
                                 noise_removal_var = target_var_noise,
                                 )
    
    #reweight intensities
    query[:,1] = tools.tuna_weight_intensity(query,
                                             query_fixed_power,
                                             query_mz_power,
                                             query_ent_power
                                             )
    
    target[:,1] = tools.tuna_weight_intensity(target,
                                             target_fixed_power,
                                             target_mz_power,
                                             target_ent_power
                                             )
    
    #match peaks
    matched = tools.match_peaks_in_spectra(query,
                                           target,
                                           ms2_da = match_tolerance)
    

    #refer to matched specs only from now on
    query = matched[:,1]
    target = matched[:,2]

    #restandardize everything before distance
    query /= np.sum(query)
    target /= np.sum(query)

    #generate uncollapsed intensity combining funcitons
    difs = dif_a * np.abs(query - target) ** dif_b
    mults = mult_a * (query * target) ** mult_b

    #generate normalizations
    mult_norm = mult_norm_a * (np.power(query, mult_norm_b) * np.power(target, mult_norm_b))
    add_norm = add_norm_a * (np.power(query, add_norm_b) + np.power(target, add_norm_b))

    #depending on collapse and expand terms, consolidate or don't to some degree
    collapsed_term = 0
    if collapsed != 0:
        collapsed_term = collapsed * (np.sum(difs) + np.sum(mults)) / (np.sum(mult_norm) + np.sum(add_norm)) 

    #mult norm doesn't really make sense for expanded normalization
    expanded_term = 0
    if expanded != 0:
        expanded_term = expanded * np.sum((difs + mults) / add_norm)

    return collapsed_term + expanded_term


