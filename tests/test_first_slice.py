import math

import pytest
from scipy.stats import norm

from optpricing.engines import BlackScholesEngine, MonteCarloEngine, UnsupportedCombination
from optpricing.greeks import FiniteDifferenceGreeks
from optpricing.instruments import American, Option
from optpricing.market import MarketData
from optpricing.payoffs import AsianCall, Call, Put
from optpricing.processes import GBM


@pytest.fixture
def market():
    return MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)


@pytest.fixture
def process():
    return GBM(vol=0.2)


def test_mc_matches_black_scholes_for_call(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    bs_price = BlackScholesEngine().price(option, process, market).price
    mc_result = MonteCarloEngine(n_paths=200_000, n_steps=1, seed=7).price(option, process, market)

    assert mc_result.price == pytest.approx(bs_price, abs=4 * mc_result.std_error)


def test_mc_matches_black_scholes_for_put(market, process):
    option = Option(payoff=Put(strike=100.0), expiry=1.0)

    bs_price = BlackScholesEngine().price(option, process, market).price
    mc_result = MonteCarloEngine(n_paths=200_000, n_steps=1, seed=7).price(option, process, market)

    assert mc_result.price == pytest.approx(bs_price, abs=4 * mc_result.std_error)


def test_black_scholes_rejects_american_exercise(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0, exercise=American())

    with pytest.raises(UnsupportedCombination):
        BlackScholesEngine().price(option, process, market)


def test_monte_carlo_handles_path_dependent_payoff(market, process):
    option = Option(payoff=AsianCall(strike=100.0), expiry=1.0)

    result = MonteCarloEngine(n_paths=50_000, n_steps=50, seed=7).price(option, process, market)

    assert result.price > 0.0


def test_finite_difference_delta_matches_bs_closed_form(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    engine = BlackScholesEngine()
    fd_greeks = FiniteDifferenceGreeks(engine, bump=1e-3).compute(option, process, market)

    # Closed-form BS delta for a call: e^{-qT} * N(d1)
    bs_result = engine.price(option, process, market)
    d1 = bs_result.diagnostics["d1"]
    expected_delta = math.exp(-market.dividend_yield * option.expiry) * norm.cdf(d1)
    assert fd_greeks.delta == pytest.approx(expected_delta, abs=1e-3)
