"""
Differentially private histogram-related functions
Builds upon the histogram functionality of Numpy
"""
import warnings
from sys import maxsize

import numpy as np

from mechanisms.accountant import BudgetAccountant
from mechanisms.geometric import GeometricTruncated
from mechanisms.utils import PrivacyLeakWarning, warn_unused_args, check_random_state


# noinspection PyShadowingBuiltins
def histogram(sample, epsilon=1.0, bins=10, range=None, weights=None, density=None, random_state=None, accountant=None,
              **unused_args):
    r"""
    Compute the differentially private histogram of a set of data.
    The histogram is computed using :obj:`numpy.histogram`, and noise added using :class:`.GeometricTruncated` to
    satisfy differential privacy.  If the `range` parameter is not specified correctly, a :class:`.PrivacyLeakWarning`
    is thrown.  Users are referred to :obj:`numpy.histogram` for more usage notes.
    Parameters
    ----------
    sample : array_like
        Input data.  The histogram is computed over the flattened array.
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon` to be applied.
    bins : int or sequence of scalars or str, default: 10
        If `bins` is an int, it defines the number of equal-width bins in the given range (10, by default).  If `bins`
        is a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge, allowing
        for non-uniform bin widths.
        If `bins` is a string, it defines the method used to calculate the optimal bin width, as defined by
        `histogram_bin_edges`.
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range is simply ``(a.min(), a.max())``.  Values outside
        the range are ignored.  The first element of the range must be less than or equal to the second. `range` affects
        the automatic bin computation as well.  While bin width is computed to be optimal based on the actual data
        within `range`, the bin count will fill the entire range including portions containing no data.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in `a` only contributes its associated weight
        towards the bin count (instead of 1).  If `density` is True, the weights are normalized, so that the integral
        of the density over the range remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in each bin.  If ``True``, the result is the value
        of the probability *density* function at the bin, normalized such that the *integral* over the range is 1.
        Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen; it is
        not a probability *mass* function.
    random_state : int or RandomState, optional
        Controls the randomness of the algorithm.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.
    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.
    Returns
    -------
    hist : array
        The values of the histogram.  See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    See Also
    --------
    histogramdd, histogram2d
    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if `bins` is::
      [1, 2, 3, 4]
    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The last bin, however,
    is ``[3, 4]``, which *includes* 4.
    """
    warn_unused_args(unused_args)

    random_state = check_random_state(random_state)

    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, 0)

    if range is None:
        warnings.warn("Range parameter has not been specified. Falling back to taking range from the data.\n"
                      "To ensure differential privacy, and no additional privacy leakage, the range must be "
                      "specified independently of the data (i.e., using domain knowledge).", PrivacyLeakWarning)

    hist, bin_edges = np.histogram(sample, bins=bins, range=range, weights=weights, density=None)

    dp_mech = GeometricTruncated(epsilon=epsilon, sensitivity=1, lower=0, upper=maxsize, random_state=random_state)

    dp_hist = np.zeros_like(hist)

    for i in np.arange(dp_hist.shape[0]):
        dp_hist[i] = dp_mech.randomise(int(hist[i]))

    # dp_hist = dp_hist.astype(float, casting='safe')

    accountant.spend(epsilon, 0)

    if density:
        bin_sizes = np.array(np.diff(bin_edges), float)
        return dp_hist / bin_sizes / (dp_hist.sum() if dp_hist.sum() else 1), bin_edges

    return dp_hist, bin_edges
