import pytest

from optpricing.engines import BinomialTreeEngine, BlackScholesEngine, LongstaffSchwartzEngine
from optpricing.instruments import American, European, Option
from optpricing.payoffs import Call, Put


def test_lsm_agrees_with_binomial_tree_for_american_put(market, process):
    """The primary correctness check: an entirely different method (forward
    simulation + regression) against a converged lattice, on the case LSM is
    actually for (early exercise). Loose tolerance because LSM adds genuine
    Monte Carlo noise and regression approximation error on top of time
    discretization, unlike comparing two deterministic grid methods.
    """
    option = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())

    lsm_result = LongstaffSchwartzEngine(n_paths=100_000, n_steps=50, seed=7).price(option, process, market)
    tree_price = BinomialTreeEngine(n_steps=2000).price(option, process, market).price

    assert lsm_result.price == pytest.approx(tree_price, abs=max(0.1, 4 * lsm_result.std_error))


def test_lsm_degenerates_to_european_at_one_step(market, process):
    """With n_steps=1 there's no intermediate date to exercise at (the
    backward loop over interior steps is empty by construction), so LSM
    should collapse to plain discounted-terminal-payoff Monte Carlo —
    exactly what MonteCarloEngine computes for the European version. This
    pins down the discounting bookkeeping independently of whether the
    regression/exercise logic is correct.
    """
    from optpricing.engines import MonteCarloEngine

    american = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())
    european = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=European())

    lsm_result = LongstaffSchwartzEngine(n_paths=100_000, n_steps=1, seed=7).price(american, process, market)
    mc_result = MonteCarloEngine(n_paths=100_000, n_steps=1, seed=7).price(european, process, market)

    assert lsm_result.price == pytest.approx(mc_result.price, abs=1e-8)


def test_american_put_from_lsm_is_at_least_intrinsic_and_european_value(market, process):
    option_american = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())
    lsm_price = LongstaffSchwartzEngine(n_paths=100_000, n_steps=50, seed=7).price(
        option_american, process, market
    ).price

    european_price = BlackScholesEngine().price(Option(payoff=Put(strike=100.0), expiry=1.0), process, market).price
    intrinsic = max(100.0 - market.spot, 0.0)

    assert lsm_price >= european_price - 1e-6
    assert lsm_price >= intrinsic - 1e-6


def test_american_call_lsm_matches_european_without_dividends():
    """Same no-early-exercise-for-calls argument as the binomial test —
    LSM should discover this empirically through the regression (it should
    essentially never choose to exercise early), landing close to the
    European closed form.
    """
    from optpricing.market import MarketData
    from optpricing.processes import GBM

    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
    process = GBM(vol=0.2)

    european_price = BlackScholesEngine().price(Option(payoff=Call(strike=100.0), expiry=1.0), process, market).price
    lsm_result = LongstaffSchwartzEngine(n_paths=100_000, n_steps=50, seed=7).price(
        Option(payoff=Call(strike=100.0), expiry=1.0, exercise=American()), process, market
    )

    assert lsm_result.price == pytest.approx(european_price, abs=max(0.1, 4 * lsm_result.std_error))
