import math_distance
import tools
import numpy as np


def distance(
    spec_matched,
    method: str,
) -> float:
    """
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
    return tools.sigmoid(dist)

    return dist


def multiple_similarity(
    spectrum_query,
    spectrum_library,
    methods,
    ms2_ppm,
    ms2_da
):
    """

    """
    result = {}
    if len(spectrum_query)==0 or len(spectrum_library)==0:
        for m in methods:
            result[m]=0
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
