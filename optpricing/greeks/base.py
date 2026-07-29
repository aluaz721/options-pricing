from abc import ABC, abstractmethod
from dataclasses import dataclass

from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.processes.base import StochasticProcess


@dataclass(frozen=True)
class Greeks:
    # All optional: not every (GreeksEngine, process) pair can produce every
    # sensitivity — e.g. FiniteDifferenceGreeks leaves vega as None for a
    # process with no single `.vol` field to bump (Heston). None means "not
    # computed here," not "zero."
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None


class GreeksEngine(ABC):
    @abstractmethod
    def compute(self, option: Option, process: StochasticProcess, market: MarketData) -> Greeks:
        raise NotImplementedError
