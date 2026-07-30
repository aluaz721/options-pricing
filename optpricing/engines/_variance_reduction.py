import numpy as np


def antithetic_aware_stderr(values: np.ndarray, antithetic: bool) -> float:
    """Standard error of the mean of `values`.

    Plain Monte Carlo: the values are (approximately) i.i.d., so the usual
    std(values)/sqrt(n) applies directly.

    Antithetic: by the pairing convention documented on
    StochasticProcess.simulate (values[:n/2] driven by Z, values[n/2:] by
    -Z), the two halves are *not* independent — they're negatively
    correlated by construction, that's the entire point. Computing
    std/sqrt(n) over the pooled array as if all n values were independent
    ignores that correlation: it estimates the marginal spread of a single
    path's outcome, not the (smaller) spread of the paired *average* that
    the antithetic estimator actually reports as its price. That would
    overstate the true standard error and hide the variance reduction the
    technique is there to provide.

    The correct estimator instead treats each pair average
    (values[i] + values[i + n/2]) / 2 as one i.i.d. observation, and takes
    the standard error over those n/2 pair averages.
    """
    if not antithetic:
        n = len(values)
        return float(values.std(ddof=1) / np.sqrt(n))

    half = len(values) // 2
    pair_averages = (values[:half] + values[half:]) / 2.0
    return float(pair_averages.std(ddof=1) / np.sqrt(half))
