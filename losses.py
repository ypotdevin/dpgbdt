# -*- coding: utf-8 -*-
# ypo@informatik.uni-kiel.de

from typing import Optional
import numpy as np
from numpy.core.fromnumeric import clip
from sklearn.ensemble._gb_losses import (BinomialDeviance,
                                         ClassificationLossFunction,
                                         LeastSquaresError, LossFunction,
                                         MultinomialDeviance)

import logging

logger = logging.getLogger(__name__)


def keep_each_tree_predicate(
        y,
        raw_predictions,
        previous_loss: float,
        current_loss: float) -> bool:
    """This predicate will always return True.

    This implies that every new tree will be used in the ensemble."""
    return True

def useful_tree_predicate(
        y,
        raw_predictions,
        previous_loss: float,
        current_loss: float) -> bool:
    """This predicated tells whether `current_loss < previous_loss`.

    This implies that only usefull trees (the ones that lower the overall loss)
    will be added to the ensemble. Trees that increase the loss will be
    discarded.
    """
    return current_loss < previous_loss

class ClippedLeastSquaresError(LeastSquaresError):
    """Loss function for clipped least squares (LS) estimation.

    This extension overrides the `LeastSquaresError` method __call__() by
    clipping the squared deviations before summing them.
    It extends `LeastSquaresError`'s constructor by adding the `clipping_bound`
    member.

    Parameters
    ----------
    clipping_bound : float
        The bound used to clip the squared deviations from above and below.
    """

    def __init__(self, clipping_bound):
        super().__init__()
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None):
        """Compute the clipped least squares loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        c = self.clipping_bound
        if sample_weight is None:
            return np.mean(np.clip((y - raw_predictions.ravel()) ** 2, -c, c))
        else:
            raise NotImplementedError(
                "Clipping is not implemented if argument `sample_weight` is "
                "not None."
            )

    def __repr__(self) -> str:
        return (
            "ClippedLeastSquaresError("
            "clipping_bound={})"
        ).format(self.clipping_bound)

class ClippedLSEPredicate():
    """This predicate realises the AboveThreshold mechanism.

    It compares the noisy current loss with the previous loss. Basically it
    evaluates `current_loss + noise < previous_loss`, where the noise is based
    on `privacy_budget`, `clipping_bound` and the length of its calling argument
    `y`.
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
            y,
            raw_predictions,
            previous_loss: float,
            current_loss: float) -> bool:
        lap_noise = self.rng.laplace(
          scale = self.clipping_bound / (len(y) * self.privacy_budget)
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


class ClippedBinomialDeviance(BinomialDeviance):
    def __init__(self, n_classes, clipping_bound):
        super().__init__(n_classes = n_classes)
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None):
        raise NotImplementedError()

class ClippedMultinomialDeviance(MultinomialDeviance):
    def __init__(self, n_classes, clipping_bound):
        super().__init__(n_classes = n_classes)
        self.clipping_bound = clipping_bound

    def __call__(self, y, raw_predictions, sample_weight = None):
        raise NotImplementedError()

class RootExpQLeastSquaresError(LeastSquaresError):
    def __init__(
            self,
            lower_bound,
            upper_bound,
            privacy_budget,
            random_seed = None):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.privacy_budget = privacy_budget
        self.random_seed = random_seed

    def __call__(self, y, raw_predictions, sample_weight = None):
        if sample_weight is None:
            # Note that sqrt(median(x_1 ** 2, x_2 ** 2, ..., x_n ** 2))
            #           = median(|x_1|, |x_2|, ...,  |x_n|)
            abs_devs = np.abs(y - raw_predictions.ravel())
            abs_devs = np.sort(abs_devs)
            assert self.lower_bound <= abs_devs[0] <= abs_devs[-1] <= self.upper_bound
            abs_devs = np.insert(
                abs_devs,
                obj = (0, len(abs_devs)),
                values = [self.lower_bound, self.upper_bound]
            )
            med = self._dp_median(abs_devs)
            logger.debug(
                "True median: %f; DP median: %f", np.median(abs_devs), med
            )
            return med
        else:
            raise NotImplementedError(
                "DP-Median is not implemented if argument `sample_weight` is "
                "not None."
            )

    def _dp_median(self, x):
        """Assume that `x` = [x_min, x_1, x_2, ..., x_n, x_max] and that `x` is
        sorted. Return the median of `x` in a differentially private manner."""
        rng = np.random.default_rng(seed = self.random_seed)
        probabilities = self._probabilities(
            utilities = self._utilities(x),
            bin_sizes = self._bin_sizes(x),
            privacy_budget = self.privacy_budget
        )
        i = rng.choice(np.arange(len(x) - 1), p = probabilities)
        return rng.uniform(low = x[i], high = x[i + 1])

    def _utilities(self, x):
        """Assume that x has length n + 2 (i.e. it contains n elements of
        interest and the lower and upper bound. Return n + 1 utilities."""
        _x = np.arange(len(x) - 1)
        median_position = np.floor_divide(len(x), 2)
        utilities = np.where(
            _x < median_position,
            _x + 1 - median_position,
            median_position - _x
        )
        return utilities

    def _bin_sizes(self, x):
        """Assume that `x` = [x_min, x_1, x_2, ..., x_n, x_max] and that `x` is
        sorted. Return n + 1 bin sizes"""
        bin_sizes = x - np.roll(x, 1)
        return bin_sizes[1:]

    def _probabilities(self, utilities, bin_sizes, privacy_budget):
        ps = bin_sizes * np.exp(privacy_budget / 2 * utilities)
        ps /= ps.sum()
        return ps

    def __repr__(self) -> str:
        _repr = (
            "RootExpQLeastSquaresError(lower_bound={},upper_bound={},"
            "privacy_budget={},random_seed={})"
        ).format(
            self.lower_bound,
            self.upper_bound,
            self.privacy_budget,
            self.random_seed
        )
        return _repr

class RootMedianLeastSquaresError(LeastSquaresError):
    def __call__(self, y, raw_predictions, sample_weight = None):
        if sample_weight is None:
            # Note that sqrt(median(x_1 ** 2, x_2 ** 2, ..., x_n ** 2))
            #           = median(|x_1|, |x_2|, ..., |x_n|)
            return np.median(np.abs(y - raw_predictions.ravel()))
        else:
            raise NotImplementedError(
                "RMedLSE is not implemented if argument `sample_weight` is "
                "not None."
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
