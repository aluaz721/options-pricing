from optpricing.engines.base import PricingEngine, PricingResult
from optpricing.instruments.exercise import American, European
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.payoffs.vanilla import Call, Put
from optpricing.processes.base import StochasticProcess
from optpricing.processes.gbm import GBM


class BinomialTreeEngine(PricingEngine):
    """Cox-Ross-Rubinstein tree. European/American vanilla under GBM only —
    early exercise is the whole point of reaching for a tree here.
    """

    def __init__(self, n_steps: int = 500):
        self.n_steps = n_steps

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        return (
            isinstance(process, GBM)
            and isinstance(option.exercise, (European, American))
            and isinstance(option.payoff, (Call, Put))
        )

    def price(self, option: Option, process: StochasticProcess, market: MarketData) -> PricingResult:
        self._check_supported(option, process)
        # TODO: build the CRR lattice, backward-induct, apply max(intrinsic,
        # continuation) at each node when option.exercise is American.
        raise NotImplementedError("BinomialTreeEngine.price is not implemented yet")
