import numpy as np

from optpricing.engines._variance_reduction import antithetic_aware_stderr
from optpricing.engines.base import PricingEngine, PricingResult
from optpricing.instruments.exercise import European
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.processes.base import StochasticProcess


class MonteCarloEngine(PricingEngine):
    """Simulate paths, discount the terminal payoff, average.

    Handles any process/payoff pair the process can simulate — vanilla or
    path-dependent, single-factor or multi-factor. Early exercise (American,
    Bermudan) needs a regression-based continuation-value estimate (Longstaff-
    Schwartz), which is a separate engine, not this one.
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: int | None = None,
        antithetic: bool = False,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        # Whether antithetic=True actually works depends on the process —
        # see StochasticProcess.simulate's docstring — so this engine just
        # passes the flag through and lets the process accept or reject it,
        # rather than trying to know in advance which processes support it.
        self.antithetic = antithetic

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        # No process/payoff check here, deliberately: this engine only calls
        # process.simulate() and option.payoff(paths), so it works for any
        # process that implements simulate() and any payoff — the sole
        # constraint is European exercise, since there's no continuation
        # value to compare against without a regression step (that's
        # Longstaff-Schwartz, a separate engine).
        return isinstance(option.exercise, European)

    def price(self, option: Option, process: StochasticProcess, market: MarketData) -> PricingResult:
        self._check_supported(option, process)

        rng = np.random.default_rng(self.seed)
        paths = process.simulate(
            market, option.expiry, self.n_steps, self.n_paths, rng, antithetic=self.antithetic
        )

        # Constant-rate discounting on the whole payoff vector at once —
        # fine here since MarketData.rate is a flat rate, not a curve.
        discounted = np.exp(-market.rate * option.expiry) * option.payoff(paths.spot)
        price = discounted.mean()
        std_error = antithetic_aware_stderr(discounted, self.antithetic)

        return PricingResult(price=float(price), std_error=std_error)
