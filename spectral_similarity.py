import math_distance
import tools
import numpy as np


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
        print('g1')
        result = 1.0

    return result


def distance(
    spec_matched,
    method: str,
) -> float:
    """
    """
    function_name = method + "_distance"
    if hasattr(math_distance, function_name):
        f = getattr(math_distance, function_name)
        dist = f(spec_matched[:, 1], spec_matched[:, 2])

    else:
        raise RuntimeError("Method name: {} error!".format(method))

    # Normalize result
    return tools.sigmoid(dist)


def multiple_similarity(
    spectrum_query,
    spectrum_library,
    methods,
    ms2_ppm = None,
    ms2_da = None
):
    """

    """
    result = {}
    if len(spectrum_query)==0 or len(spectrum_library)==0:
        for m in methods:
            result[m]=-1
        return result

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
            spec_matched, method=m
        )
        result[m] = float(1 - dist)

    return result
