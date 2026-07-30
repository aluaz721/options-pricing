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
    # something else again.
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
        antithetic: bool = False,
    ) -> SimulatedPaths:
        """antithetic=True asks the process to draw its own random numbers in
        mirrored +Z/-Z pairs (a variance-reduction technique — see
        optpricing.engines._variance_reduction) instead of independently.
        This lives on the process, not the engine, because the process is
        what actually owns the random-number draws; an engine can't mirror
        randomness it never sees. It's opt-in per process (default False,
        and a process may raise NotImplementedError if it can't support it —
        see Heston, where the QE scheme's variance step is a nonlinear
        function of its driving randomness, so naive mirroring isn't
        guaranteed to reduce variance the way it provably does for GBM's
        linear log-price map).

        Convention when antithetic=True and supported: the returned
        SimulatedPaths.spot has n_paths rows ordered as
        [paths driven by Z (first n_paths//2), paths driven by -Z (second
        n_paths//2)] — i.e. row i and row i + n_paths//2 are a mirrored
        pair. Any caller exploiting the pairing (e.g. to compute a correctly
        reduced standard error) relies on this ordering.
        """
        raise NotImplementedError

    def characteristic_function(self, u: complex, t: float, market: MarketData) -> complex:
        """log-price characteristic function, for FFT/COS engines. Not every process has one in closed form."""
        raise NotImplementedError(f"{type(self).__name__} has no characteristic function yet")
