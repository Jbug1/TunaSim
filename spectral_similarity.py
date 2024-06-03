import math_distance
import tools

import numpy as np
from typing import Union

methods_range = {
    "proportional_entropy": [0, 1],
    "proportional_manhattan": [0, 1],
    "proportional_lorentzian": [0,1],
    "entropy": [0, np.log(4)],
    "dot_product": [0,1],
    #"cosine": [0,1],
    "absolute_value": [0, 2],
    "bhattacharya_1": [0, np.arccos(0) ** 2],
    "bhattacharya_2": [0, np.inf],
    "canberra": [0, np.inf],
    #"clark": [0, np.inf],
    #"avg_l": [0, 1.5],
    "divergence": [0, np.inf],
    #"euclidean": [0, np.sqrt(2)],
    "hellinger": [0, np.inf],
    #"improved_similarity": [0, np.inf],
    "lorentzian": [0, np.inf],
    "mod_lorentzian": [0, np.inf],
    "manhattan": [0, 2],
    "matusita": [0, np.sqrt(2)],
    "mean_character": [0, 2],
    "motyka": [-0.5, 0],
    "pearson_correlation": [-1, 1],
    #"penrose_shape": [0, np.sqrt(2)],
    #"penrose_size": [0, np.inf],
    "probabilistic_symmetric_chi_squared": [0, 1],
    "squared_chord": [0, 2],
    "mod_squared_chord": [0, 2],
    "mod2_squared_chord": [0, 2],
    "proportional_squared_chord" : [0, 2],
    "squared_euclidean": [0, 2],
    "symmetric_chi_squared": [0, 0.5 * np.sqrt(2)],
    "whittaker_index_of_association": [0, np.inf],
    #"perc_peaks_in_common": [0, 1],
    #"rbf": [0, 1],
    "chi2": [0, 1],
    #"cosine_kernel": [0, 1],
    "laplacian": [0, 1],
    #"minkowski": [0, 1],
    #"correlation": [0, 1],
    "jensenshannon": [0, 1],
    #"sqeuclidean": [0, 1],
    #"gini": [0, 1],
    #"l2": [0, 1],
    "common_mass": [0, 1],
    "cross_ent": [0, np.inf],
    "proportional_cross_ent": [0, np.inf],
    "braycurtis": [0, 1],
    "binary_cross_ent": [0, np.inf],
    "kl": [0, 1],
    #"chebyshev": [0, 1],
    "fidelity": [0, 1],
    "harmonic_mean": [0, 1],
    "ruzicka": [0, 1],
    "roberts": [0, 1],
    "intersection": [0, 1],
}


def similarity(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    method: str,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_normalize_result: bool = True,
) -> float:
    """
    Calculate the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Similarity between two spectra
    """
    if need_normalize_result:
        return 1 - distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            method=method,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )
    else:
        return 0 - distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            method=method,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )


def distance(
    spec_matched,
    method: str,
    need_normalize_result: bool = True,
) -> float:
    """
    Calculate the distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """

    # Calculate similarity

    if "reverse" in method:
        dist = math_distance.reverse_distance(
            spec_matched[:, 1],
            spec_matched[:, 2],
            metric="_".join(method.split("_")[1:]),
        )

    elif "max" in method:
        dist = math_distance.max_distance(
            spec_matched[:, 1],
            spec_matched[:, 2],
            metric="_".join(method.split("_")[1:]),
        )

    elif "min" in method and method != "minkowski":

        dist = math_distance.min_distance(
            spec_matched[:, 1],
            spec_matched[:, 2],
            metric="_".join(method.split("_")[1:]),
        )

    elif "ave" in method:

        dist = math_distance.ave_distance(
            spec_matched[:, 1],
            spec_matched[:, 2],
            metric="_".join(method.split("_")[1:]),
        )

    else:
        function_name = method + "_distance"
        if hasattr(math_distance, function_name):
            f = getattr(math_distance, function_name)
            dist = f(spec_matched[:, 1], spec_matched[:, 2])

        else:
            raise RuntimeError("Method name: {} error!".format(method))

    # Normalize result
    if need_normalize_result:
        if method not in methods_range:
            try:
                dist_range = methods_range["_".join(method.split("_")[1:])]
            except:
                print(f'error on {method}')
        else:
            dist_range = methods_range[method]

        dist = normalize_distance(dist, dist_range)

    return dist


def distance_sep(
    query,
    target,
    method: str,
    need_normalize_result: bool = True,
) -> float:
    """
    Calculate the distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """

    # Calculate similarity

    if "reverse" in method:
        dist = math_distance.reverse_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    elif "max" in method:
        dist = math_distance.max_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    elif "min" in method and method != "minkowski":

        dist = math_distance.min_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    elif "ave" in method:

        dist = math_distance.ave_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    else:
        function_name = method + "_distance"
        if hasattr(math_distance, function_name):
            f = getattr(math_distance, function_name)
            dist = f(query, target)

        else:
            raise RuntimeError("Method name: {} error!".format(method))

    # Normalize result
    if need_normalize_result:
        if method not in methods_range:
            try:
                dist_range = methods_range["_".join(method.split("_")[1:])]
            except:
                print(f'error on {method}')
        else:
            dist_range = methods_range[method]

        dist = normalize_distance(dist, dist_range)

    if np.isnan(dist):
        dist=1
        
    return dist



def multiple_distance(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    methods: list = None,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_normalize_result: bool = True,
) -> dict:
    """
    Calculate multiple distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param methods: A list of method names.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """
    if methods is None:
        methods = (
            [i for i in methods_range]
            + [f"reverse_{i}" for i in methods_range]
            + [f"max_{i}" for i in methods_range]
        )

    result = {}
    if ms2_ppm is not None:
        spec_matched = tools.match_peaks_in_spectra(
            spectrum_query, spectrum_library, ms2_ppm=ms2_ppm
        )
    else:
        spec_matched = tools.match_peaks_in_spectra(
            spectrum_query, spectrum_library, ms2_da=ms2_da
        )

    for m in methods:
        dist = distance(
            spec_matched,
            method=m,
            need_normalize_result=need_normalize_result,
        )
        result[m] = float(dist)
    return result


def multiple_similarity(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    methods: list = None,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_normalize_result: bool = True,
    reweight_spectra = False,
    reweight_method=None
) -> dict:
    """
    Calculate multiple distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param methods: A list of method names.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """
    result = {}
    if len(spectrum_query)==0 or len(spectrum_library)==0:
        for m in methods:
            result[m]=0
        return result

    if methods is None:
        methods = (
            [i for i in methods_range]
            + [f"reverse_{i}" for i in methods_range]
            + [f"max_{i}" for i in methods_range]
        )

    if ms2_ppm is not None:
        spec_matched = tools.match_peaks_in_spectra(
            spectrum_query, spectrum_library, ms2_ppm=ms2_ppm
        )
    else:
        spec_matched = tools.match_peaks_in_spectra(
            spectrum_query, spectrum_library, ms2_da=ms2_da
        )

    if reweight_spectra:
        spec_matched[:,1]=tools.weight_intensity(spec_matched[:,1], reweight_method=reweight_method)
        spec_matched[:,2]=tools.weight_intensity(spec_matched[:,2], reweight_method=reweight_method)

    for m in methods:

        dist = distance(
            spec_matched, method=m, need_normalize_result=need_normalize_result
        )
        result[m] = float(1 - dist)
    return result


def normalize_distance(dist, dist_range):
    if dist_range[1] == np.inf:
        if dist_range[0] == 0:
            result = 1 - 1 / (1 + dist)
        elif dist_range[1] == 1:
            result = 1 - 1 / dist
        else:
            raise NotImplementedError()
    elif dist_range[0] == -np.inf:
        if dist_range[1] == 0:
            result = -1 / (-1 + dist)
        else:
            raise NotImplementedError()
    else:
        result = (dist - dist_range[0]) / (dist_range[1] - dist_range[0])

    if result < 0:
        result = 0.0
    elif result > 1:
        result = 1.0

    return result
