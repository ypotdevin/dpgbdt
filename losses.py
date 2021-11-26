# ypo@informatik.uni-kiel.de

import logging
import math
from typing import Any, Optional, cast

import numpy as np
from sklearn.ensemble._gb_losses import (BinomialDeviance,
                                         ClassificationLossFunction,
                                         LeastSquaresError, LossFunction,
                                         MultinomialDeviance)

from dp_variance.dp_variance import smooth_sensitivity
from dp_rmse import dp_rMS_cauchy

__all__ = [
    BinomialDeviance, ClassificationLossFunction, LeastSquaresError,
    LossFunction, MultinomialDeviance, 'keep_each_tree_predicate',
    'useful_tree_predicate', 'ClippedLeastSquaresError',
    'ClippedLSEPredicate', 'ClippedBinomialDeviance',
    'ClippedMultinomialDeviance', 'RootExpQLeastSquaresError',
    'RootMedianLeastSquaresError', 'DP_quasi_rMSE',
]
logger = logging.getLogger(__name__)


def keep_each_tree_predicate(
        y: Any,
        raw_predictions: Any,
        previous_loss: float,
        current_loss: float) -> bool:
    """This predicate will always return True.

    This implies that every new tree will be used in the ensemble."""
    return True

def useful_tree_predicate(
        y: Any,
        raw_predictions: Any,
        previous_loss: float,
        current_loss: float) -> bool:
    """This predicated tells whether `current_loss < previous_loss`.

    This implies that only usefull trees (the ones that lower the
    overall loss) will be added to the ensemble. Trees that increase the
    loss will be discarded.
    """
    return current_loss < previous_loss

class ClippedLeastSquaresError(LeastSquaresError):# type: ignore
    """Loss function for clipped least squares (LS) estimation.

    This extension overrides the `LeastSquaresError` method __call__()
    by clipping the squared deviations before summing them.
    It extends `LeastSquaresError`'s constructor by adding the
    `clipping_bound` member.

    Parameters
    ----------
    clipping_bound : float
        The bound used to clip the squared deviations from above and
        below.
    """

    def __init__(self, clipping_bound: float) -> None:
        super().__init__()
        self.clipping_bound = clipping_bound

    def __call__(
            self,
            y: np.ndarray,
            raw_predictions: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
        ) -> float:
        """Compute the clipped least squares loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : ndarray of shape (n_samples,), optional
            Sample weights.
        """
        c = self.clipping_bound
        if sample_weight is None:
            return cast(
                float,
                np.mean(
                    np.clip((y - raw_predictions.ravel()) ** 2, -c, c)
                )
            )
        else:
            raise NotImplementedError(
                "Clipping is not implemented if argument `sample_weight`"
                "is not None."
            )

    def __repr__(self) -> str:
        return (
            "ClippedLeastSquaresError("
            "clipping_bound={})"
        ).format(self.clipping_bound)

class ClippedLSEPredicate():
    """This predicate realises the AboveThreshold mechanism.

    It compares the noisy current loss with the previous loss. Basically
    it evaluates `current_loss + noise < previous_loss`, where the noise
    is based on `privacy_budget`, `clipping_bound` and the length of its
    calling argument `y`.
    """
    def __init__(
            self,
            privacy_budget: float,
            clipping_bound: float,
            random_seed: Optional[int] = None):
        self.privacy_budget = privacy_budget
        self.clipping_bound = clipping_bound
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed = self.random_seed)

    def __call__(
            self,
            y: np.ndarray,
            raw_predictions: Any,
            previous_loss: float,
            current_loss: float) -> bool:
        lap_noise = cast(
            float,
            self.rng.laplace(
            scale = self.clipping_bound / (len(y) * self.privacy_budget)
            )
        )
        logger.debug(
            "Performing AboveThreshold mechanism using c = %f, len(y) = %d, "
            "eps = %f, resulting in lap_noise = %f",
            self.clipping_bound, len(y), self.privacy_budget, lap_noise
        )
        return current_loss + lap_noise < previous_loss

    def __repr__(self) -> str:
        _repr = (
            "ClippedLSEPredicate(privacy_budget={},clipping_bound={},"
            "random_seed={})"
        ).format(self.privacy_budget, self.clipping_bound, self.random_seed)
        return _repr


class ClippedBinomialDeviance(BinomialDeviance): # type: ignore
    def __init__(self, n_classes: int, clipping_bound: float) -> None:
        super().__init__(n_classes = n_classes)
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None): # type: ignore
        raise NotImplementedError()

class ClippedMultinomialDeviance(MultinomialDeviance): # type: ignore
    def __init__(self, n_classes: int, clipping_bound: float) -> None:
        super().__init__(n_classes = n_classes)
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None): # type: ignore
        raise NotImplementedError()

class RootExpQLeastSquaresError(LeastSquaresError): # type: ignore
    def __init__(
            self,
            lower_bound: float,
            upper_bound: float,
            privacy_budget: float,
            q: float = 0.5,
            random_seed: Optional[int] = None) -> None:
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.privacy_budget = privacy_budget
        self.q = q
        self.random_seed = random_seed

    def __call__(
            self,
            y: np.ndarray,
            raw_predictions: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> float:
        if sample_weight is None:
            # Note that sqrt(median(x_1 ** 2, x_2 ** 2, ..., x_n ** 2))
            #           = median(|x_1|, |x_2|, ..., |x_n|)
            # (the same holds for quantiles other than q = 0.5).
            abs_devs = np.abs(y - raw_predictions.ravel())
            abs_devs = np.sort(abs_devs)
            assert (
                self.lower_bound
                <= abs_devs[0]
                <= abs_devs[-1]
                <= self.upper_bound
            )
            abs_devs = np.insert(
                abs_devs,
                obj = (0, len(abs_devs)),
                values = [self.lower_bound, self.upper_bound]
            )
            quantile = self._exp_q(abs_devs, self.q)
            logger.debug(
                "True quantile: %f; DP quantile: %f",
                np.quantile(abs_devs, self.q),
                quantile
            )
            return quantile
        else:
            raise NotImplementedError(
                "DP-Median is not implemented if argument `sample_weight` is "
                "not None."
            )

    def _exp_q(self, x: np.ndarray, q: float = 0.5) -> float:
        """Assume that `x` = [x_min, x_1, x_2, ..., x_n, x_max] and that
        `x` is sorted. Return the median of `x` in a differentially
        private manner."""
        rng = np.random.default_rng(seed = self.random_seed)
        probabilities = self._probabilities(
            utilities = self._utilities(x, q),
            bin_sizes = self._bin_sizes(x),
            privacy_budget = self.privacy_budget
        )
        i = rng.choice(np.arange(len(x) - 1), p = probabilities)
        return cast(float, rng.uniform(low = x[i], high = x[i + 1]))

    def _utilities(self, x: np.ndarray, q: float) -> np.ndarray:
        """Assume that x has length n + 2 (i.e. it contains n elements
        of interest and the lower and upper bound. Return n + 1
        utilities."""
        _x = np.arange(len(x) - 1)
        median_position = np.floor(len(x) * q)
        utilities = cast(
            np.ndarray,
            np.where(
                _x < median_position,
                _x + 1 - median_position,
                median_position - _x
            )
        )
        return utilities

    def _bin_sizes(self, x: np.ndarray) -> np.ndarray:
        """Assume that `x` = [x_min, x_1, x_2, ..., x_n, x_max] and that
        `x` is sorted. Return n + 1 bin sizes"""
        bin_sizes = x - np.roll(x, 1)
        return  cast(np.ndarray, bin_sizes[1:])

    def _probabilities(self,
            utilities: np.ndarray,
            bin_sizes: np.ndarray,
            privacy_budget: float
        ) -> np.ndarray:
        utility_ = privacy_budget / 2 * utilities
        # This is for numerical stability:
        max_utility = utility_.max()
        ps = bin_sizes * np.exp(utility_ - max_utility)
        ps /= ps.sum()
        return cast(np.ndarray, ps)

    def __repr__(self) -> str:
        _repr = (
            "RootExpQLeastSquaresError(lower_bound={},upper_bound={},"
            "privacy_budget={},q={},random_seed={})"
        ).format(
            self.lower_bound,
            self.upper_bound,
            self.privacy_budget,
            self.q,
            self.random_seed,
        )
        return _repr

class RootMedianLeastSquaresError(LeastSquaresError): # type: ignore
    def __call__(
            self,
            y: np.ndarray,
            raw_predictions: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
        ) -> float:
        if sample_weight is None:
            # Note that sqrt(median(x_1 ** 2, x_2 ** 2, ..., x_n ** 2))
            #           = median(|x_1|, |x_2|, ..., |x_n|)
            return cast(
                float,
                np.median(np.abs(y - raw_predictions.ravel()))
            )
        else:
            raise NotImplementedError(
                "RMedLSE is not implemented if argument `sample_weight` is "
                "not None."
            )

    def __repr__(self) -> str:
        return "RootMedianLeastSquaresError"

class RootQuantileLeastSquaresError(LeastSquaresError): # type: ignore
    def __init__(self, q: float) -> None:
        super().__init__()
        self.q = q

    def __call__(self,
            y: np.ndarray,
            raw_predictions: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
        ) -> float:
        if sample_weight is None:
            # Note that sqrt(median(x_1 ** 2, x_2 ** 2, ..., x_n ** 2))
            #           = median(|x_1|, |x_2|, ..., |x_n|),
            # which holds analogously for other quantiles.
            return cast(
                float,
                np.quantile(np.abs(y - raw_predictions.ravel()), self.q)
            )
        else:
            raise NotImplementedError(
                "RQuantileLSE is not implemented if argument `sample_weight`"
                "is not None."
            )

    def __repr__(self) -> str:
        return "RootQuantileLeastSquaresError(q={})".format(self.q)

class DP_quasi_rMSE(LeastSquaresError): # type: ignore
    def __init__(self,
            privacy_budget: float,
            beta: float,
            L: float,
            U: float,
            m: int,
            seed: int
        ) -> None:
        super().__init__()
        self.privacy_budget = privacy_budget
        self.beta = beta
        self.L = L
        self.U = U
        self.m = m
        self.seed = seed

    def __call__(self,
            y: np.ndarray,
            raw_predictions: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
        ) -> float:
        if sample_weight is None:
            assert (
                self.privacy_budget > 0 and self.beta > 0
                and self.m >= 0 and self.L < self.U
            )
            sample = y - raw_predictions.ravel()
            sample.sort()
            trimmed_sample = sample[self.m : sample.size - self.m]
            s = smooth_sensitivity(
                trimmed_sample,
                0.0,
                self.beta,
                self.L,
                self.U,
                'std'
            )
            scale = math.sqrt(2) * s / self.privacy_budget
            rng = np.random.default_rng(seed = self.seed)
            noise = rng.standard_cauchy(size = 1)[0] * scale
            return cast(float, np.sqrt(np.mean(trimmed_sample ** 2)) + noise)
        else:
            raise NotImplementedError(
                "DP_quasi-rMSE is not implemented if argument `sample_weight`"
                "is not None."
            )

    def __repr__(self) -> str:
        template = "DP_quasi-rMSE(privacy_budget={},beta={},L={},U={},m={})"
        return template.format(
            self.privacy_budget,
            self.beta,
            self.L,
            self.U,
            self.m
        )

class DP_rMSE(LeastSquaresError):# type: ignore
    def __init__(self,
            privacy_budget: float,
            U: float,
            seed: Optional[int] = None,
        ) -> None:
        super().__init__()
        self.privacy_budget = privacy_budget
        self.U = U
        self.seed = seed

    def __call__(self,
            y: np.ndarray,
            raw_predictions: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
        ) -> float:
        if sample_weight is None:
            return dp_rMS_cauchy(
                y - raw_predictions.ravel(), self.privacy_budget,
                self.U, self.seed
            )
        else:
            raise NotImplementedError(
                "DP_rMSE is not implemented if argument `sample_weight`"
                "is not None."
            )

    def __repr__(self) -> str:
        return "DP-rMSE(privacy_budget={},U={})".format(
            self.privacy_budget, self.U
        )

LOSS_FUNCTIONS = {
    'ls'        : LeastSquaresError,
    'c_ls'      : ClippedLeastSquaresError,
    'expq_ls'   : RootExpQLeastSquaresError,
    'bin_dev'   : BinomialDeviance,
    'c_bin_dev' : ClippedBinomialDeviance,
    'mul_dev'   : MultinomialDeviance,
    'c_mul_dev' : ClippedMultinomialDeviance
}
