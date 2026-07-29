from optpricing.greeks.base import Greeks, GreeksEngine
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.processes.base import StochasticProcess


class MalliavinGreeks(GreeksEngine):
    """Malliavin-weighted MC: dV/dtheta = E[e^{-rT} * Payoff(S_T) * pi_theta],
    where pi_theta is a process-specific weight built from the Malliavin
    derivative of S_T. Unlike pathwise, this differentiates the *density*
    rather than the payoff, so it handles discontinuous payoffs (barrier,
    digital) that break pathwise differentiation.
    """

    def __init__(self, n_paths: int = 100_000, n_steps: int = 252, seed: int | None = None):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def compute(self, option: Option, process: StochasticProcess, market: MarketData) -> Greeks:
        # TODO: needs each process to expose its Malliavin weight for delta/
        # vega (closed-form for GBM: weight_delta = Z / (S0 * sigma * sqrt(T)));
        # not yet part of the StochasticProcess interface.
        raise NotImplementedError("MalliavinGreeks.compute is not implemented yet")
