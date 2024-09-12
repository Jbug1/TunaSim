# funcs to move over to new repo
import numpy as np
import scipy
from sklearn.metrics.pairwise import pairwise_kernels as pk
import scipy.spatial.distance as dist
import sklearn.metrics as met


def _select_common_peaks(p, q):
    """
    should we be renormalizing p?
    """
    select = q > 0
    p = p[select]
    p_sum = np.sum(p)
    if p_sum > 0:
        p = p / p_sum
    q = q[select]
    return p, q


def proportional_entropy_distance(p, q):
    r"""
    Unweighted entropy distance:

    .. math:

        -\frac{2\times S_{PQ}-S_P-S_Q} {ln(4)}, S_I=\sum_{i} {I_i ln(I_i)}
    """
    merged = p + q
    entropy_increase = (
        2 * scipy.stats.entropy(merged)
        - scipy.stats.entropy(p)
        - scipy.stats.entropy(q)
    )

    norm_distance = (
        2 * scipy.stats.entropy(np.concatenate([p, q]))
        - scipy.stats.entropy(p)
        - scipy.stats.entropy(q)
    )

    return entropy_increase / norm_distance

def proportional_manhattan_distance(p, q):
    r"""
    Manhattan distance:

    .. math::

        \sum|P_{i}-Q_{i}|
    """
    return np.sum(np.abs(p - q))/np.sum(np.concatenate([p,q]))


def proportional_lorentzian_distance(p, q):
    r"""
    Lorentzian distance:

    .. math::

        \sum{\ln(1+|P_i-Q_i|)}
    """
    return np.sum(np.log(1 + np.abs(p - q))) / np.sum(
        np.log(1 + np.concatenate((p, q)))
    )


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
    return entropy_increase


def common_mass_distance(p, q):

    # get indices of matched peaks and drop rest
    match_inds = np.where(p * q > 0)[0]

    # check that there is at least one match
    if len(match_inds) == 0:
        return 1

    p = p[match_inds]
    q = q[match_inds]
    matched = (p + q) / 2

    return 1 - sum(matched)


def cross_entropy(p, q):

    epsilon = 1e-50
    q = q + epsilon
    return -np.sum(p * np.log(q))


def binary_cross_entropy(y_true, y_pred):

    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return sum(ce)


def kl_distance(p, q):

    if len(p) == 0:
        return 1

    # add jitter for numerical stability
    p = p + 1e-10
    q = q + 1e-10

    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass
    lim_1 = lim_1 + 1e-10
    lim_2 = lim_2 + 1e-10

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def cross_ent_distance(p, q):

    return (cross_entropy(p, q) + cross_entropy(q, p))/2

def proportional_cross_ent_distance(p, q):

    return max(cross_entropy(p, q), cross_entropy(q, p))/cross_entropy(np.concatenate((p,q)),np.concatenate((q,p)))


def binary_cross_ent_distance(p, q):

    return max(binary_cross_entropy(p, q), binary_cross_entropy(q, p))


def euclidean_distance(p, q):
    r"""
    Euclidean distance:

    .. math::

        (\sum|P_{i}-Q_{i}|^2)^{1/2}
    """
    return np.sqrt(np.sum(np.power(p - q, 2)))


def manhattan_distance(p, q):
    r"""
    Manhattan distance:

    .. math::

        \sum|P_{i}-Q_{i}|
    """
    return np.sum(np.abs(p - q))


def chebyshev_distance(p, q):
    r"""
    Chebyshev distance:

    .. math::

        \underset{i}{\max}{(|P_{i}\ -\ Q_{i}|)}
    """
    return np.max(np.abs(p - q))


def squared_euclidean_distance(p, q):
    r"""
    Squared Euclidean distance:

    .. math::

        \sum(P_{i}-Q_{i})^2
    """
    return np.sum(np.power(p - q, 2))


def l2_distance(p, q):

    matched = (p + q) / 2
    return -1 * (2 * (matched @ matched) - (p @ p) - (q @ q))


def proportional_fidelity_distance(p, q):
    r"""
    Fidelity distance:

    .. math::

        1-\sum\sqrt{P_{i}Q_{i}}
    """
    return (1 - np.sum(np.sqrt(p * q)))/(1 - np.sum(np.sqrt(np.concatenate((p,q)))))

def fidelity_distance(p, q):
    r"""
    Fidelity distance:

    .. math::

        1-\sum\sqrt{P_{i}Q_{i}}
    """
    return (1 - np.sum(np.sqrt(p * q)))


def matusita_distance(p, q):
    r"""
    Matusita distance:

    .. math::

        \sqrt{\sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2}
    """
    return np.sqrt(np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2)))


def proportional_squared_chord_distance(p, q):
    r"""
    Squared-chord distance:

    .. math::

        \sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2
    """
    return np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2)) / np.sum(np.power(np.sqrt(np.concatenate((p,q))), 2))

def squared_chord_distance(p, q):
    r"""
    Squared-chord distance:

    .. math::

        \sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2
    """
    return np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2))

def mod_squared_chord_distance(p, q):
    r"""
    Squared-chord distance:

    .. math::

        \sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2
    """
    return np.sum(abs(np.sqrt(p) - np.sqrt(q)))

def mod2_squared_chord_distance(p, q):
    r"""
    Squared-chord distance:

    .. math::

        \sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2
    """
    return np.sum(np.power(np.sqrt(p) - np.sqrt(q), 4))


def bhattacharya_1_distance(p, q):
    r"""
    Bhattacharya 1 distance:

    .. math::

        (\arccos{(\sum\sqrt{P_{i}Q_{i}})})^2
    """
    s = np.sum(np.sqrt(p * q))
    if s > 1:
        s = 1
    return np.power(np.arccos(s), 2)


def bhattacharya_2_distance(p, q):
    r"""
    Bhattacharya 2 distance:

    .. math::

        -\ln{(\sum\sqrt{P_{i}Q_{i}})}
    """
    s = np.sum(np.sqrt(p * q))
    if s == 0:
        return np.inf
    else:
        return -np.log(s)


def harmonic_mean_distance(p, q):
    r"""
    Harmonic mean distance:

    .. math::

        1-2\sum(\frac{P_{i}Q_{i}}{P_{i}+Q_{i}})
    """
    return 1 - 2 * np.sum(p * q / (p + q))


def probabilistic_symmetric_chi_squared_distance(p, q):
    r"""
    Probabilistic symmetric χ2 distance:

    .. math::

        \frac{1}{2} \times \sum\frac{(P_{i}-Q_{i}\ )^2}{P_{i}+Q_{i}\ }
    """
    return 1 / 2 * np.sum(np.power(p - q, 2) / (p + q))


def ruzicka_distance(p, q):
    r"""
    Ruzicka distance:

    .. math::

        \frac{\sum{|P_{i}-Q_{i}|}}{\sum{\max(P_{i},Q_{i})}}
    """
    dist = np.sum(np.abs(p - q)) / np.sum(np.maximum(p, q))
    return dist


def roberts_distance(p, q):
    r"""
    Roberts distance:

    .. math::

        1-\sum\frac{(P_{i}+Q_{i})\frac{\min{(P_{i},Q_{i})}}{\max{(P_{i},Q_{i})}}}{\sum(P_{i}+Q_{i})}
    """
    return 1 - np.sum((p + q) / np.sum(p + q) * np.minimum(p, q) / np.maximum(p, q))


def intersection_distance(p, q):
    r"""
    Intersection distance:

    .. math::

        1-\frac{\sum\min{(P_{i},Q_{i})}}{\min(\sum{P_{i},\sum{Q_{i})}}}
    """
    return 1 - np.sum(np.minimum(p, q)) / min(np.sum(p), np.sum(q))


def motyka_distance(p, q):
    r"""
    Motyka distance:

    .. math::

        -\frac{\sum\min{(P_{i},Q_{i})}}{\sum(P_{i}+Q_{i})}
    """
    dist = np.sum(np.minimum(p, q)) / np.sum(p + q)
    return -dist


def canberra_distance(p, q):
    r"""
    Canberra distance:

    .. math::

        \sum\frac{|P_{i}-Q_{i}|}{|P_{i}|+|Q_{i}|}
    """
    return np.sum(np.abs(p - q) / (np.abs(p) + np.abs(q)))


def baroni_urbani_buser_distance(p, q):
    r"""
    Baroni-Urbani-Buser distance:

    .. math::

        1-\frac{\sum\min{(P_i,Q_i)}+\sqrt{\sum\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}{\sum{\max{(P_i,Q_i)}+\sqrt{\sum{\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}}}
    """
    if np.max(p) < np.max(q):
        p, q = q, p
    d1 = np.sqrt(np.sum(np.minimum(p, q) * np.sum(max(p) - np.maximum(p, q))))
    return 1 - (np.sum(np.minimum(p, q)) + d1) / (np.sum(np.maximum(p, q)) + d1)


def penrose_size_distance(p, q):
    r"""
    Penrose size distance:

    .. math::

        \sqrt N\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return np.sqrt(n) * np.sum(np.abs(p - q))


def mean_character_distance(p, q):
    r"""
    Mean character distance:

    .. math::

        \frac{1}{N}\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return 1 / n * np.sum(np.abs(p - q))


def lorentzian_distance(p, q):
    r"""
    Lorentzian distance:

    .. math::

        \sum{\ln(1+|P_i-Q_i|)}
    """
    return np.sum(np.log(1 + np.abs(p - q)))

def mod_lorentzian_distance(p, q):
    r"""
    Lorentzian distance:

    .. math::

        \sum{\ln(1+|P_i-Q_i|)}
    """
    return np.sum(np.log(1 + 0.5*np.abs(p - q)))


def penrose_shape_distance(p, q):
    r"""
    Penrose shape distance:

    .. math::

        \sqrt{\sum((P_i-\bar{P})-(Q_i-\bar{Q}))^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(np.sum(np.power((p - p_avg) - (q - q_avg), 2)))


def divergence_distance(p, q):
    r"""
    Divergence distance:

    .. math::

        2\sum\frac{(P_i-Q_i)^2}{(P_i+Q_i)^2}
    """
    return 2 * np.sum((np.power(p - q, 2)) / np.power(p + q, 2))


def avg_l_distance(p, q):
    r"""
    Avg (L1, L∞) distance:

    .. math::

        \frac{1}{2}(\sum|P_i-Q_i|+\underset{i}{\max}{|P_i-Q_i|})
    """
    return np.sum(np.abs(p - q)) + max(np.abs(p - q))


def clark_distance(p, q):
    r"""
    Clark distance:

    .. math::

        (\frac{1}{N}\sum(\frac{P_i-Q_i}{|P_i|+|Q_i|})^2)^\frac{1}{2}
    """
    n = np.sum(p > 0)
    return np.sqrt(1 / n * np.sum(np.power((p - q) / (np.abs(p) + np.abs(q)), 2)))


def hellinger_distance(p, q):
    r"""
    Hellinger distance:

    .. math::

        \sqrt{2\sum(\sqrt{\frac{P_i}{\bar{P}}}-\sqrt{\frac{Q_i}{\bar{Q}}})^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(2 * np.sum(np.power(np.sqrt(p / p_avg) - np.sqrt(q / q_avg), 2)))


def perc_peaks_in_common_distance(p, q):
    "max is 1"

    matched_peaks = len(np.where(p * q > 0)[0])
    peaks_p = len(np.where(p > 0)[0])
    peaks_q = len(np.where(q > 0)[0])

    if peaks_p == 0 or peaks_q == 0:
        return 1

    return 1 - min(matched_peaks / peaks_p, matched_peaks / peaks_q)


def whittaker_index_of_association_distance(p, q):
    r"""
    Whittaker index of association distance:

    .. math::

        \frac{1}{2}\sum|\frac{P_i}{\bar{P}}-\frac{Q_i}{\bar{Q}}|
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return 1 / 2 * np.sum(np.abs(p / p_avg - q / q_avg))


def symmetric_chi_squared_distance(p, q):
    r"""
    Symmetric χ2 distance:

    .. math::

        \sqrt{\sum{\frac{\bar{P}+\bar{Q}}{N(\bar{P}+\bar{Q})^2}\frac{(P_i\bar{Q}-Q_i\bar{P})^2}{P_i+Q_i}\ }}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    n = np.sum(p > 0)

    d1 = (p_avg + q_avg) / (n * np.power(p_avg + q_avg, 2))
    return np.sqrt(d1 * np.sum(np.power(p * q_avg - q * p_avg, 2) / (p + q)))


def pearson_correlation_distance(p, q):
    r"""
    Pearson/Spearman Correlation Coefficient:

    .. math::

        \frac{\sum[(Q_i-\bar{Q})(P_i-\bar{P})]}{\sqrt{\sum(Q_i-\bar{Q})^2\sum(P_i-\bar{P})^2}}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)

    x = np.sum((q - q_avg) * (p - p_avg))
    y = np.sqrt(np.sum(np.power(q - q_avg, 2)) * np.sum(np.power(p - p_avg, 2)))

    if x == 0 and y == 0:
        return 0.0
    else:
        return -x / y


def improved_similarity_distance(p, q):
    r"""
    Improved Similarity Index:

    .. math::

        \sqrt{\frac{1}{N}\sum\{\frac{P_i-Q_i}{P_i+Q_i}\}^2}
    """
    n = np.sum(p > 0)
    return np.sqrt(1 / n * np.sum(np.power((p - q) / (p + q), 2)))


def absolute_value_distance(p, q):
    r"""
    Absolute Value Distance:

    .. math::

        \frac { \sum(|Q_i-P_i|)}{\sum P_i}

    """
    dist = np.sum(np.abs(q - p)) / np.sum(p)
    return dist


def dot_product_distance(p, q):
    r"""
    Dot product distance:

    .. math::

        1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
    """
    score = np.power(np.sum(q * p), 2) / (
        np.sum(np.power(q, 2)) * np.sum(np.power(p, 2))
    )
    return 1 - np.sqrt(score)


def sigmoid_distance(p, q):
    """
    flexible norm constant implemented
    lim_1 and lim_2 will be passed thru kernel function to get max distance for this length
    In the case that the length in common is 0, return 1 for distance
    """

    if len(p) == 0:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass

    return (1 - pk([p], [q], metric="sigmoid")[0][0]) / (
        1 - pk([lim_1], [lim_2], metric="sigmoid")[0][0]
    )

def sigmoid_unnorm_distance(p, q):
    """
    flexible norm constant implemented
    lim_1 and lim_2 will be passed thru kernel function to get max distance for this length
    In the case that the length in common is 0, return 1 for distance
    """

    if len(p) == 0:
        return 1

    return (1 - pk([p], [q], metric="sigmoid")[0][0]) 

def cosine_kernel_distance(p, q):
    "max is always 1, no need for normalization constant"

    return 1 - pk([p], [q], metric="cosine")[0][0]


def laplacian_distance(p, q):
    """
    flexible norm constant implemented
    lim_1 and lim_2 will be passed thru kernel function to get max distance for this length
    In the case that the length in common is 0, return 1 for distance
    """
    if len(p) == 0:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass
    
    
    return (1 - pk([p], [q], metric="laplacian")[0][0]) / (
        1 - pk([lim_1], [lim_2], metric="laplacian")[0][0]
    )

def laplacian_unnorm_distance(p,q):
    
    if len(p) == 0:
        return 1
    
    return (1 - pk([p], [q], metric="laplacian")[0][0])



def cosine_kernel_distance(p, q):
    "max is always 1, no need for normalization constant"

    return 1 - pk([p], [q], metric="cosine")[0][0]


def cosine_distance(p, q):
    r"""
    Cosine distance, it gives the same result as the dot product.

    .. math::

        1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
    """
    return dot_product_distance(p, q)


def perc_peaks_common_distance(p, q):
    "max is 1"

    matched_peaks = len(np.where(p * q > 0)[0])
    peaks_p = len(np.where(p > 0)[0])
    peaks_q = len(np.where(q > 0)[0])

    return 1 - min(matched_peaks / peaks_p, matched_peaks / peaks_q)


def rbf_distance(p, q):
    """
    flexible norm constant implemented
    lim_1 and lim_2 will be passed thru kernel function to get max distance for this length
    In the case that the length in common is 0, return 1 for distance
    """

    if len(p) == 0:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass

    return (1 - pk([p], [q], metric="rbf")[0][0]) / (
        1 - pk([lim_1], [lim_2], metric="rbf")[0][0]
    )
  


def chi2_distance(p, q):
    "max is always 1, no need for normalization constant"
    try:
        return 1 - pk([p], [q], metric="chi2")[0][0]
    except:
        return 1


def additive_chi2_distance(p, q):
    "max dist is always -2, no need for normalization constant"

    try:
        return pk([p], [q], metric="additive_chi2")[0][0] / -2
    except:
        return 1


def linear_distance(p, q):
    "max is always 1, no need for normalization constant"

    return 1 - pk([p], [q], metric="linear")[0][0]


def reverse_distance(p, q, metric):

    p, q = _select_common_peaks(p, q)

    if np.sum(p) == 0:
        return 1
    else:
        return eval(f"{metric}_distance(p,q)")


def max_distance(p, q, metric):

    return max(
        reverse_distance(p, q, metric),
        reverse_distance(q, p, metric),
    )


def min_distance(p, q, metric):

    return min(
        reverse_distance(p, q, metric),
        reverse_distance(q, p, metric),
    )


def ave_distance(p, q, metric):

    return (reverse_distance(p, q, metric) + reverse_distance(q, p, metric)) / 2


def minkowski_distance(p_, q):
    """
    max is 1.18
    """

    if len(p_) == 1:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p_))
    lim_2 = np.zeros(len(p_))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass

    return dist.minkowski(p_, q, p=4) / dist.minkowski(lim_1, lim_2, p=4)


def correlation_distance(p, q):
    """
    max is 2
    """

    if len(p) < 2:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    lim_2[1] += 1

    return dist.correlation(p, q) / dist.correlation(lim_1, lim_2)


def jensenshannon_distance(p, q):
    """
    max is 0.82
    """

    if len(p) <2:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    lim_2[1] += 1

    dist_ = dist.jensenshannon(p, q) / dist.jensenshannon(lim_1, lim_2)

    if np.isnan(dist_):
        return 1
    else:
        return dist_


def sqeuclidean_distance(p, q):
    """
    max is 2
    """

    if len(p) == 0:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass

    return dist.sqeuclidean(p, q) / dist.sqeuclidean(lim_1, lim_2)


def braycurtis_distance(p, q):
    """
    max is 2
    """

    if len(p) == 0:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass

    return dist.braycurtis(p, q) / dist.braycurtis(lim_1, lim_2)


def gini_distance(p, q):

    if len(p) == 0:
        return 1

    matched = (p + q) / 2

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    try:
        lim_2[1] += 1
    except:
        pass

    lim_matched = (lim_1 + lim_2) / 2

    return (
        (matched @ (1 - matched))
        - (p @ (1 - p))
        - (q @ (1 - q)) / (lim_matched @ (1 - lim_matched))
        - (lim_1 @ (1 - lim_1))
        - (lim_2 @ (1 - lim_2))
    )
