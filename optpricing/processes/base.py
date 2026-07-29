from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from optpricing.market import MarketData


@dataclass
class SimulatedPaths:
    """Underlying price paths, plus any auxiliary state factors a multi-factor
    process needs to carry (e.g. Heston's variance path)."""

    spot: np.ndarray  # (n_paths, n_steps + 1)
    # A free-form dict rather than named fields (e.g. `variance: np.ndarray`)
    # because different processes carry different extra state — GBM needs
    # none, Heston needs variance, a future local-vol model might need
    # something else again. Naming it here would force every process to
    # carry fields it doesn't use.
    extra: dict[str, np.ndarray] = field(default_factory=dict)


class StochasticProcess(ABC):
    """The only thing every process must be able to do is simulate itself.

    Everything else (characteristic function for FFT/COS, tree/PDE coefficients
    for lattice and finite-difference methods) is opt-in: a given engine declares
    which process types it supports via its own `supports()` check, rather than
    every process being forced to implement machinery only some engines need.
    """

    @abstractmethod
    def simulate(
        self,
        market: MarketData,
        t: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> SimulatedPaths:
        raise NotImplementedError

    def characteristic_function(self, u: complex, t: float, market: MarketData) -> complex:
        """log-price characteristic function, for FFT/COS engines. Not every process has one in closed form."""
        raise NotImplementedError(f"{type(self).__name__} has no characteristic function yet")
