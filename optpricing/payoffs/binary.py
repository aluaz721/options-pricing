from dataclasses import dataclass

import numpy as np

from optpricing.payoffs.base import Payoff

# Cash-or-nothing digitals: pay a fixed amount if the terminal price clears
# the strike, nothing otherwise. Included mainly because, unlike most exotics
# here, they have a simple closed form (see BlackScholesEngine) as well as an
# MC and FD price — useful for a three-way engine comparison without having
# to reach for a barrier or Asian payoff.


@dataclass(frozen=True)
class CashOrNothingCall(Payoff):
    strike: float
    cash: float = 1.0

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        return np.where(paths[:, -1] > self.strike, self.cash, 0.0)


@dataclass(frozen=True)
class CashOrNothingPut(Payoff):
    strike: float
    cash: float = 1.0

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        return np.where(paths[:, -1] < self.strike, self.cash, 0.0)
