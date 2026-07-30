import math

import pytest
from scipy.stats import norm

from optpricing.engines import BlackScholesEngine
from optpricing.greeks import FiniteDifferenceGreeks
from optpricing.instruments import Option
from optpricing.payoffs import Call


def test_finite_difference_delta_matches_bs_closed_form(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    engine = BlackScholesEngine()
    fd_greeks = FiniteDifferenceGreeks(engine, bump=1e-3).compute(option, process, market)

    # Closed-form BS delta for a call: e^{-qT} * N(d1)
    bs_result = engine.price(option, process, market)
    d1 = bs_result.diagnostics["d1"]
    expected_delta = math.exp(-market.dividend_yield * option.expiry) * norm.cdf(d1)
    assert fd_greeks.delta == pytest.approx(expected_delta, abs=1e-3)
