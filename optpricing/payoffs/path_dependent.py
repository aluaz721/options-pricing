from dataclasses import dataclass
from typing import Literal

import numpy as np

from optpricing.payoffs.base import Payoff

# Two representative path-dependent payoffs to prove out the interface.
# Lookback, ladder/cliquet, and the down/in barrier variants follow the same
# pattern and are left for when an engine actually needs to price them.


@dataclass(frozen=True)
class AsianCall(Payoff):
    strike: float
    average_type: Literal["arithmetic", "geometric"] = "arithmetic"
    is_path_dependent: bool = True

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        # paths[:, 0] is today's known spot, not a random observation — the
        # average is taken over the simulated (post-t0) path only.
        if self.average_type == "arithmetic":
            average = paths[:, 1:].mean(axis=1)
        else:
            average = np.exp(np.log(paths[:, 1:]).mean(axis=1))
        return np.maximum(average - self.strike, 0.0)


@dataclass(frozen=True)
class UpAndOutCall(Payoff):
    strike: float
    barrier: float
    is_path_dependent: bool = True

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        knocked_out = (paths >= self.barrier).any(axis=1)
        vanilla = np.maximum(paths[:, -1] - self.strike, 0.0)
        return np.where(knocked_out, 0.0, vanilla)
