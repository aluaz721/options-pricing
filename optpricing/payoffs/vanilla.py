from dataclasses import dataclass

import numpy as np

from optpricing.payoffs.base import Payoff

# frozen dataclasses: cheap value-equality and hashing for free, and they can
# be handed straight to dataclasses.replace() by anything that wants a bumped
# copy (not needed for strike today, but keeps every Payoff subclass uniform).


@dataclass(frozen=True)
class Call(Payoff):
    strike: float

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(paths[:, -1] - self.strike, 0.0)


@dataclass(frozen=True)
class Put(Payoff):
    strike: float

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - paths[:, -1], 0.0)
