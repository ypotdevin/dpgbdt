import math
from typing import Iterator, Optional, Tuple, cast

import numpy as np

def dp_rMS_cauchy(
        errors: np.ndarray, epsilon: float,
        U: float, seed: Optional[int] = None
    ) -> float:
    sorted_errors = np.sort(errors)
    rng = np.random.default_rng(seed)
    gamma = 2.0
    beta = epsilon / 2 * (gamma + 1)
    sens = rMS_smooth_sensitivity(sorted_errors, beta, U)
    rmse = np.sqrt((sorted_errors ** 2).sum() / len(sorted_errors))
    noise = rng.standard_cauchy()
    dp_rmse = cast(float, rmse + 2 * (gamma + 1) * sens * noise / epsilon)
    return dp_rmse

def rM_smooth_sensitivity(
        squared_errors: np.ndarray,
        beta: float,
        U: float
    ) -> float:
    """
    Parameters
    ----------
    squared_errors : np.ndarray
        A sorted(!) array of already squared errors (to avoid having two
        arguments and a more difficult analysis).
    U : float
        The (data independent) upper bound on squared errors.

    Returns
    -------
    smooth_sensitivity : float
        The beta-smooth sensitivity of the root mean (squared error)
        function, i.e.
            se_1, ..., se_n |-> sqrt((se_1 + ... + se_n) / n)
    """
    return _smooth_sensitivity(
        squared_errors, beta, U
    )

def rMS_smooth_sensitivity(
        errors: np.ndarray,
        beta: float,
        U: float
    ) -> float:
    """
    Parameters
    ----------
    errors : np.ndarray
        A sorted array of errors (to avoid having two arguments and a
        more difficult analysis).
    U : float
        The (data independent) upper bound on errors (not squared
        errors!).

    Returns
    -------
    smooth_sensitivity : float
        The beta-smooth sensitivity of the root mean squared (error)
        function, i.e.
            e_1, ..., e_n |-> sqrt((e_1 ** 2 + ... + e_n ** 2) / n)
    """
    return _smooth_sensitivity(
        errors ** 2, beta, U ** 2
    )

def _smooth_sensitivity(
        elements: np.ndarray,
        beta: float,
        U: float
    ) -> float:
    assert elements.max() <= U
    smooth_sens = -math.inf
    for (loc_sens, dist) in _local_sensitivities(elements, U):
        smooth_sens = max(
            loc_sens * math.exp(-beta * dist),
            smooth_sens
        )
    return smooth_sens

def _local_sensitivities(
        elements: np.ndarray, U: float
    ) -> Iterator[Tuple[float, int]]:
    """
    Yields
    ------
    (A^(k)(`elements`), k) : (float, int)
        A^(k)(`elements`) as defined in Definition 3.1 in 'Smooth
        Sensitivity and Sampling in Private Data Analysis', Nissim et
        al. 2011, for k = 0, ..., len(`elements`).
    """
    n = len(elements)
    for ((s1, k1), (s2, k2)) in zip(
            _prefix_sums(elements),
            _suffix_sums(elements, U)
        ):
        assert k1 == k2
        sens1 = _local_sensitivity(s1, n, U)
        sens2 = _local_sensitivity(s2, n, U)
        yield (max(sens1, sens2), k1)

def _local_sensitivity(s: float, n: int, U: float) -> float:
    """The term for calculating the local sensitivity of rM and rMS
    are identical. It must only be payed attention to the unit/scale of
    the inputs."""
    if s <= 0:
        return math.sqrt(U / n)
    else:
        return math.sqrt(s / n) * abs(math.sqrt(1 + U / s) - 1)

def _suffix_sums(elements: np.ndarray, U: float) -> Iterator[Tuple[float, int]]:
    """
    Yields
    ------
    (sum, k) : (float, int)
        Given `elements` = e1, ..., en,
            sum = U + ... + U + e_{k+1} + e_{k+2} + ... + e_{n}
        (the first/smallest k entries of `elements` have been replaced
        by `U`).
    """
    _sum = elements.sum()
    yield (_sum, 0)
    for (k, e) in enumerate(elements, 1):
        _sum = _sum - e + U
        yield (_sum, k)

def _prefix_sums(elements: np.ndarray) -> Iterator[Tuple[float, int]]:
    """
    Yields
    ------
    (sum, k) : (float, int)
        Given `elements` = e1, ..., en and k = 0, ..., n (increasing),
            sum = e_1 + ... + e_{n - k - 1} + e_{n - k} + 0 + ... + 0
        (the last k entries of `elements` have been replaced by 0).
    """
    _sum = elements.sum()
    yield (_sum, 0)
    for (k, e) in enumerate(reversed(elements), 1):# type: ignore
        _sum -= e
        yield (_sum, k)



def main() -> None:
    rng = np.random.default_rng(42)
    big_sample = rng.standard_normal(4500) + 5
    rMS = np.sqrt((big_sample ** 2).sum() / len(big_sample))
    dp_rMS = dp_rMS_cauchy(big_sample, 1.0, 10)
    print(
        "DP-rMS of {} element array: {}. Leaky rMS: {}".format(
            len(big_sample),
            dp_rMS,
            rMS
        )
    )

if __name__ == '__main__':
    main()