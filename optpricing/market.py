from dataclasses import dataclass


# frozen so FiniteDifferenceGreeks can bump copies via dataclasses.replace()
# instead of mutating shared state between the up/down repricings.
@dataclass(frozen=True)
class MarketData:
    """Everything a pricing engine needs that isn't part of the instrument or the process.

    Notably absent: volatility. It's a model parameter, not observed market
    data, so it lives on the StochasticProcess instead — GBM has one vol,
    Heston has five vol-related parameters, and this class shouldn't need to
    change shape depending on which one is in use.
    """

    spot: float
    rate: float
    dividend_yield: float = 0.0  # continuous yield; fine for equity/FX, not a full curve
