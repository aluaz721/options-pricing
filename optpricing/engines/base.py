from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.processes.base import StochasticProcess


class UnsupportedCombination(ValueError):
    """Raised when an engine is asked to price an (option, process) pair it doesn't handle.

    Not every method is meaningful for every combination — e.g. there's no CRR
    tree for Heston, and no simple closed form for an American Asian option.
    Engines are expected to be explicit about what they support rather than
    silently producing a wrong number.
    """


@dataclass(frozen=True)
class PricingResult:
    price: float
    std_error: float | None = None  # populated by Monte Carlo engines
    # Free-form rather than named fields: BlackScholesEngine wants to expose
    # d1/d2, a tree engine might want the lattice, an FD engine the grid —
    # a shared schema would force every engine to populate fields it has no
    # use for. Callers that need a specific diagnostic already know which
    # engine they used and what it puts here.
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PricingEngine(ABC):
    @abstractmethod
    def supports(self, option: Option, process: StochasticProcess) -> bool:
        raise NotImplementedError

    @abstractmethod
    def price(self, option: Option, process: StochasticProcess, market: MarketData) -> PricingResult:
        raise NotImplementedError

    # Not called automatically by the ABC — each price() implementation calls
    # this itself as its first line. Enforcing the check here (e.g. via a
    # template-method pattern) would mean every engine can no longer just
    # write a straight-line price(); the one-line convention is simpler and
    # the tests catch an engine that forgets it.
    def _check_supported(self, option: Option, process: StochasticProcess) -> None:
        if not self.supports(option, process):
            raise UnsupportedCombination(
                f"{type(self).__name__} cannot price a {type(option.payoff).__name__} "
                f"with {type(option.exercise).__name__} exercise under {type(process).__name__}"
            )
