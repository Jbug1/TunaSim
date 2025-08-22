# old similarity measures
import scipy
import numpy as np
from numba import njit
from joblib import Parallel, delayed


def _weight_intensity_by_entropy(x, entropy_x):
    WEIGHT_START = 0.25
    ENTROPY_CUTOFF = 3
    weight_slope = (1 - WEIGHT_START) / ENTROPY_CUTOFF

    if np.sum(x) > 0:
        entropy_x = scipy.stats.entropy(x)
        if entropy_x < ENTROPY_CUTOFF:
            weight = WEIGHT_START + weight_slope * entropy_x
            x = np.power(x, weight)
            x_sum = np.sum(x)
            x = x / x_sum
    return x

@njit
def harmonic_mean_distance(p, q):
    r"""
    Harmonic mean distance:

    .. math::

        1-2\sum(\frac{P_{i}Q_{i}}{P_{i}+Q_{i}})
    """

    return 2 * np.sum(p * q / (p + q))

@njit
def lorentzian_distance(p, q):
    r"""
    Lorentzian distance:

    .. math::

        \sum{\ln(1+|P_i-Q_i|)}
    """

    return 1 - np.sum(np.log(1 + np.abs(p - q)))

@njit
def matusita_distance(p, q):
    r"""
    Matusita distance:

    .. math::

        \sqrt{\sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2}
    """

    return 1- np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2))

@njit
def probabilistic_symmetric_chi_squared_distance(p, q):
    r"""
    Probabilistic symmetric χ2 distance:

    .. math::

        \frac{1}{2} \times \sum\frac{(P_{i}-Q_{i}\ )^2}{P_{i}+Q_{i}\ }
    """
 
    return 1- (1 / 2 * np.sum(np.power(p - q, 2) / (p + q)))


def entropy_distance(p, q):
    r"""
    Unweighted entropy distance:

    .. math::

        -\frac{2\times S_{PQ}-S_P-S_Q} {ln(4)}, S_I=\sum_{i} {I_i ln(I_i)}
    """
   
    merged = p + q
    entropy_increase = 2 * \
                       scipy.stats.entropy(merged) - scipy.stats.entropy(p) - \
                       scipy.stats.entropy(q)
    
    return 1 - entropy_increase

@njit
def dot_product_distance(p, q):
    r"""
    Dot product distance:

    .. math::

        1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
    """
    
    score = np.power(np.sum(q * p), 2) / (
        np.sum(np.power(q, 2)) * np.sum(np.power(p, 2))
    )
    return np.sqrt(score)

def evaluate_metrics(queries, targets):
    """ 
    Evaluates this set of old measures for a collection of queries and targets
    """

    metrics = [entropy_distance, ]
    #unweighted first
    for query, target in zip(queries, targets):

        


