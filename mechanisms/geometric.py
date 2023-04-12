"""
The classic geometric mechanism for differential privacy, and its derivatives.
"""
from numbers import Integral

import numpy as np

from mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from mechanisms.utils import copy_docstring


class Geometric(DPMechanism):
    r"""
    The classic geometric mechanism for differential privacy, as first proposed by Ghosh, Roughgarden and Sundararajan.
    Extended to allow for non-unity sensitivity.
    Paper link: https://arxiv.org/pdf/0811.2841.pdf
    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].
    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).
    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.
    """
    def __init__(self, *, epsilon, sensitivity=1, random_state=None):
        super().__init__(epsilon=epsilon, delta=0.0, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = - self.epsilon / self.sensitivity if self.sensitivity > 0 else - float("inf")

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Integral):
            raise TypeError("Sensitivity must be an integer")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return sensitivity

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        if not isinstance(value, Integral):
            raise TypeError("Value to be randomised must be an integer")

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super()._check_epsilon_delta(epsilon, delta)

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        self._check_all(value)

        leading_factor = (1 - np.exp(self._scale)) / (1 + np.exp(self._scale))
        geom_series = np.exp(self._scale) / (1 - np.exp(self._scale))

        return 2 * leading_factor * (geom_series + 3 * (geom_series ** 2) + 2 * (geom_series ** 3))

    def randomise(self, value):
        """Randomise `value` with the mechanism.
        Parameters
        ----------
        value : int
            The value to be randomised.
        Returns
        -------
        int
            The randomised value.
        """
        self._check_all(value)

        # Need to account for overlap of 0-value between distributions of different sign
        unif_rv = self._rng.random() - 0.5
        unif_rv *= 1 + np.exp(self._scale)
        sgn = -1 if unif_rv < 0 else 1

        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(np.round(value + sgn * np.floor(np.log(sgn * unif_rv) / self._scale)))


class GeometricTruncated(Geometric, TruncationAndFoldingMixin):
    r"""
    The truncated geometric mechanism, where values that fall outside a pre-described range are mapped back to the
    closest point within the range.
    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].
    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).
    lower : int
        The lower bound of the mechanism.
    upper : int
        The upper bound of the mechanism.
    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.
    """
    def __init__(self, *, epsilon, sensitivity=1, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @classmethod
    def _check_bounds(cls, lower, upper):
        if not isinstance(lower, Integral) and abs(lower) != float("inf"):
            raise TypeError(f"Lower bound must be integer-valued, got {lower}")
        if not isinstance(upper, Integral) and abs(upper) != float("inf"):
            raise TypeError(f"Upper bound must be integer-valued, got {upper}")

        return super()._check_bounds(lower, upper)

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.bias)
    def variance(self, value):
        raise NotImplementedError

    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return int(np.round(self._truncate(noisy_value)))
