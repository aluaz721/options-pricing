import pytest

from optpricing.engines import BlackScholesEngine, MonteCarloEngine
from optpricing.instruments import Option
from optpricing.payoffs import AsianCall, Call, Put


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


def test_monte_carlo_handles_path_dependent_payoff(market, process):
    """MonteCarloEngine never special-cases payoff type — it just calls
    option.payoff(paths.spot) — so an Asian payoff should work with zero
    engine-side changes. This is really a check on that architectural claim,
    not just "the price is positive."
    """
    option = Option(payoff=AsianCall(strike=100.0), expiry=1.0)

    result = MonteCarloEngine(n_paths=50_000, n_steps=50, seed=7).price(option, process, market)

    assert result.price > 0.0
    # An arithmetic-average Asian call is worth less than a vanilla call on
    # the same strike: averaging reduces the effective volatility the payoff
    # is exposed to, so its optionality is strictly cheaper.
    vanilla_price = BlackScholesEngine().price(Option(payoff=Call(strike=100.0), expiry=1.0), process, market).price
    assert result.price < vanilla_price
