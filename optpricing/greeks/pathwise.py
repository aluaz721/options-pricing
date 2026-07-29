from optpricing.greeks.base import Greeks, GreeksEngine
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.processes.base import StochasticProcess


class PathwiseGreeks(GreeksEngine):
    """Differentiate the discounted payoff path-by-path instead of bumping
    prices: dV/dS0 = E[e^{-rT} * dPayoff/dS_T * dS_T/dS_0]. No bump bias, no
    extra simulations, but only valid where the payoff is a.e. differentiable
    in the state it depends on (fine for calls/puts, breaks for barrier and
    digital payoffs, where Malliavin weights are needed instead).
    """

    def __init__(self, n_paths: int = 100_000, n_steps: int = 252, seed: int | None = None):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def compute(self, option: Option, process: StochasticProcess, market: MarketData) -> Greeks:
        # TODO: requires each process to expose dS_T/dS_0 (and dS_T/d(vol) for
        # vega) along the simulated path — not yet part of the StochasticProcess
        # interface.
        raise NotImplementedError("PathwiseGreeks.compute is not implemented yet")
