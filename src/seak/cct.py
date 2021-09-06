
"""

Functions related to STAAR

"""

import numpy as np
from scipy.stats import cauchy

c = cauchy()


def cct(pvals, weights=None):
    """
    Python port of the CCT function as defined in the STAAR R-package (https://github.com/xihaoli/STAAR/blob/2f67fafec591a45e81a54eca24564b09ce90e252/R/CCT.R)

    An analytical p-value combination method using the Cauchy distribution.

    takes in a numeric vector of p-values, a numeric vector of non-negative weights, and return the aggregated p-value using Cauchy method.

    :param np.ndarray pval: Numpy array containing the p-values
    :param np.ndarray weights: Numpy array containing the weights

    :return: The aggregated p-value
    :rtype: float

    Liu, Y., & Xie, J. (2020). Cauchy combination test: a powerful test with analytic p-value calculation under arbitrary dependency structures.
    Liu, Y., et al. (2019). Acat: A fast and powerful p value combination method for rare-variant analysis in sequencing studies.
    """

    # check for NA
    assert not np.isnan(pvals).any(), 'Error: Cannot have nan in the p-values!'
    # check range
    assert not (((pvals < 0).sum() + (pvals > 1).sum()) > 0), "Error: All p-values must be between 0 and 1"

    # check for p-values that are either exactly 0 or 1
    is_zero = (pvals == 0.).any()
    is_one = (pvals == 1.).any()

    assert not (is_zero & is_one), 'Error: Cannot have both 0 and 1 p-values'

    if is_zero:
        return 0
    if is_one:
        print('Warning: there are p-values that are exactly 1!')
        return 1.

    # check the validity of weights
    if weights is None:
        weights = np.ones_like(pvals) / len(pvals)
    else:
        assert len(weights) == len(pvals), 'Error: length of weights should be the same as that of the p-values!'
        assert not ((weights < 0).any()), 'Error: all weights must be positive!'
        weights /= weights.sum()

    # check if there are very small non-zero p-values
    is_small = pvals < 1e-16
    if not is_small.any():
        cct_stat = (weights * np.tan((0.5 - pvals) * np.pi)).sum()
    else:
        cct_stat = (weights[is_small] / pvals[is_small] / np.pi).sum()
        cct_stat += (weights[~is_small] * np.tan((0.5 - pvals[~is_small]) * np.pi)).sum()

    # check if the test statistic is very large
    if (cct_stat > 1e15):
        pval = (1. / cct_stat) / np.pi
    else:
        pval = c.sf(cct_stat)

    return pval
