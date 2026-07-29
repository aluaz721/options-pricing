from dataclasses import dataclass

import numpy as np

from optpricing.market import MarketData
from optpricing.processes.base import SimulatedPaths, StochasticProcess


@dataclass(frozen=True)
class Heston(StochasticProcess):
    v0: float  # initial variance
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    xi: float  # vol-of-vol
    rho: float  # correlation between spot and variance Brownian motions

    def simulate(
        self,
        market: MarketData,
        t: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> SimulatedPaths:
        # TODO: Andersen QE scheme (Euler under-resolves the variance process
        # near zero and can produce negative variance).
        raise NotImplementedError("Heston.simulate is not implemented yet")

    def characteristic_function(self, u: complex, t: float, market: MarketData) -> complex:
        # TODO: standard Heston closed-form CF, needed by the FFT/COS engine.
        raise NotImplementedError("Heston.characteristic_function is not implemented yet")
