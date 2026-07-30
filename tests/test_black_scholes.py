import math

import pytest

from optpricing.engines import BlackScholesEngine, UnsupportedCombination
from optpricing.instruments import American, Option
from optpricing.payoffs import Call, Put


def test_black_scholes_rejects_american_exercise(market, process):
    """BlackScholesEngine.supports() is the contract under test here, not the
    pricing formula — American has no closed form because the early-exercise
    boundary isn't known analytically, so this must raise rather than
    silently price it as European.
    """
    option = Option(payoff=Call(strike=100.0), expiry=1.0, exercise=American())

    with pytest.raises(UnsupportedCombination):
        BlackScholesEngine().price(option, process, market)


def test_call_put_parity_holds(market, process):
    """C - P = S*e^{-qT} - K*e^{-rT} for European options on the same
    strike/expiry — a model-independent identity (it follows from static
    replication, not from GBM specifically), so it's a sanity check on the
    closed-form formula's r/q/T handling that doesn't depend on any external
    reference value.
    """
    call = Option(payoff=Call(strike=100.0), expiry=1.0)
    put = Option(payoff=Put(strike=100.0), expiry=1.0)

    engine = BlackScholesEngine()
    call_price = engine.price(call, process, market).price
    put_price = engine.price(put, process, market).price

    expected = market.spot * math.exp(-market.dividend_yield * call.expiry) - 100.0 * math.exp(
        -market.rate * call.expiry
    )
    assert call_price - put_price == pytest.approx(expected, abs=1e-8)
