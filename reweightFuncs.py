#reweight functions go here
import numpy as np
import scipy

def logent(x, intercept = 0, base = np.e):
    
    if np.sum(x) > 0:
        power = intercept + np.emath.logn(base,1+scipy.stats.entropy(x))
        x = np.power(x, power)
        x = x / np.sum(x)
    return x

def weight_intensity_by_entropy(x, WEIGHT_START = 0.25, ENTROPY_CUTOFF = 3, MAX_WEIGHT=1):
    
    
    weight_slope = (MAX_WEIGHT - WEIGHT_START) / ENTROPY_CUTOFF

    if np.sum(x) > 0:
        entropy_x = scipy.stats.entropy(x)
        if entropy_x < ENTROPY_CUTOFF:
            weight = WEIGHT_START + weight_slope * entropy_x
            x = np.power(x, weight)
            x_sum = np.sum(x)
            x = x / x_sum

    return x

def fixed_power(x, power):

    x = np.power(x, power)
    return x / np.sum(x)


def noise_clip(max_peak, perc_thresh = 0, fixed_thresh = 0):

    return perc_thresh * max_peak + fixed_thresh

def affine_quad_reweight(x, intercept, x1, x2):

    ent = scipy.stats.entropy(x)
    return max(0,intercept + x1 * ent + x2 * ent**2)

