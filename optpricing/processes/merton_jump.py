from dataclasses import dataclass

import numpy as np

from optpricing.market import MarketData
from optpricing.processes.base import SimulatedPaths, StochasticProcess


@dataclass(frozen=True)
class MertonJump(StochasticProcess):
    vol: float  # diffusive vol
    jump_intensity: float  # lambda, jumps per year
    jump_mean: float  # mean of log-jump size
    jump_std: float  # std of log-jump size

    def simulate(
        self,
        market: MarketData,
        t: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> SimulatedPaths:
        # TODO: Euler on the diffusive part + compound Poisson jumps per step,
        # with the compensator folded into the drift so E[S_T] matches the forward.
        raise NotImplementedError("MertonJump.simulate is not implemented yet")

    def characteristic_function(self, u: complex, t: float, market: MarketData) -> complex:
        # TODO: closed-form Merton CF, needed by the FFT/COS engine.
        raise NotImplementedError("MertonJump.characteristic_function is not implemented yet")
