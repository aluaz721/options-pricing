from dataclasses import dataclass

import numpy as np

from optpricing.market import MarketData
from optpricing.processes.base import SimulatedPaths, StochasticProcess


@dataclass(frozen=True)
class GBM(StochasticProcess):
    vol: float

    def simulate(
        self,
        market: MarketData,
        t: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> SimulatedPaths:
        # Simulating log(S) rather than S directly is exact for GBM (the SDE
        # has a closed-form solution in log space), so this has zero
        # discretization bias regardless of n_steps — unlike Heston or jump
        # processes, which need finer time steps or a scheme like Andersen QE
        # to control bias. n_steps here only controls path *resolution* for
        # path-dependent payoffs (Asian, barrier), not simulation accuracy.
        dt = t / n_steps
        drift = (market.rate - market.dividend_yield - 0.5 * self.vol**2) * dt  # risk-neutral drift
        diffusion = self.vol * np.sqrt(dt)

        z = rng.standard_normal((n_paths, n_steps))
        log_increments = drift + diffusion * z
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
        )
        spot_paths = market.spot * np.exp(log_paths)
        return SimulatedPaths(spot=spot_paths)

    def characteristic_function(self, u: complex, t: float, market: MarketData) -> complex:
        # Not used by any engine yet (BlackScholesEngine is closed-form and
        # MonteCarloEngine only calls simulate()) — implemented here anyway
        # since it's simple for GBM, ready for when an FFT/COS engine lands
        # and needs a uniform way to price under any process that has one.
        drift = market.rate - market.dividend_yield - 0.5 * self.vol**2
        return np.exp(1j * u * (np.log(market.spot) + drift * t) - 0.5 * self.vol**2 * t * u**2)
