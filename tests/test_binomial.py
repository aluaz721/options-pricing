import pytest

from optpricing.engines import BinomialTreeEngine, BlackScholesEngine, CrankNicolsonEngine
from optpricing.instruments import American, Option
from optpricing.market import MarketData
from optpricing.payoffs import Call, Put
from optpricing.processes import GBM


def test_binomial_converges_to_black_scholes_for_european_call(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    bs_price = BlackScholesEngine().price(option, process, market).price

    coarse = BinomialTreeEngine(n_steps=50).price(option, process, market).price
    fine = BinomialTreeEngine(n_steps=2000).price(option, process, market).price

    # CRR converges to BS at O(1/N) (with characteristic sawtooth
    # oscillation from the tree/BS grids aligning differently as N changes),
    # so a much finer tree should land noticeably closer than a coarse one.
    assert abs(fine - bs_price) < abs(coarse - bs_price)
    assert fine == pytest.approx(bs_price, abs=0.01)


def test_binomial_converges_to_black_scholes_for_european_put(market, process):
    option = Option(payoff=Put(strike=100.0), expiry=1.0)
    bs_price = BlackScholesEngine().price(option, process, market).price
    fine = BinomialTreeEngine(n_steps=2000).price(option, process, market).price
    assert fine == pytest.approx(bs_price, abs=0.01)


def test_risk_neutral_probability_is_a_valid_probability(market, process):
    """p = (e^{(r-q)dt} - d)/(u - d) is only a coherent pricing measure if
    0 < p < 1 — outside that range the tree would imply arbitrage between
    the up- and down-branches. Not something a user should ever hit with
    reasonable inputs, but worth pinning down given it's a genuine
    correctness precondition, not just a numerical nicety.
    """
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    result = BinomialTreeEngine(n_steps=200).price(option, process, market)
    assert 0.0 < result.diagnostics["p"] < 1.0


def test_american_put_has_early_exercise_premium(market, process):
    european = Option(payoff=Put(strike=100.0), expiry=1.0)
    american = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())

    engine = BinomialTreeEngine(n_steps=500)
    european_price = engine.price(european, process, market).price
    american_price = engine.price(american, process, market).price

    assert american_price >= european_price
    assert american_price > european_price


def test_american_call_has_no_early_exercise_premium_without_dividends():
    """Textbook result: it's never optimal to exercise an American call
    early on a non-dividend-paying stock (the option is always worth more
    alive than exercised, since exercising forfeits remaining time value for
    no offsetting benefit). With q=0 the binomial American call should
    therefore price identically to its European counterpart.
    """
    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
    process = GBM(vol=0.2)
    european = Option(payoff=Call(strike=100.0), expiry=1.0)
    american = Option(payoff=Call(strike=100.0), expiry=1.0, exercise=American())

    engine = BinomialTreeEngine(n_steps=500)
    european_price = engine.price(european, process, market).price
    american_price = engine.price(american, process, market).price

    assert american_price == pytest.approx(european_price, abs=1e-8)


def test_binomial_agrees_with_finite_difference_for_american_put(market, process):
    """Two entirely different numerical methods (a recombining lattice vs.
    Crank-Nicolson finite differences with Brennan-Schwartz projection)
    solving the same free-boundary problem should converge to the same
    price — cross-validation that doesn't depend on either being "known
    correct" in isolation.
    """
    option = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())

    tree_price = BinomialTreeEngine(n_steps=2000).price(option, process, market).price
    fd_price = CrankNicolsonEngine(n_space_steps=400, n_time_steps=400).price(option, process, market).price

    assert tree_price == pytest.approx(fd_price, abs=0.02)
