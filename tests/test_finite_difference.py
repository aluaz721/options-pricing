import pytest

from optpricing.engines import BlackScholesEngine, CrankNicolsonEngine, MonteCarloEngine
from optpricing.instruments import American, Option
from optpricing.payoffs import CashOrNothingCall, Call, Put, UpAndOutCall


@pytest.fixture
def fd_engine():
    return CrankNicolsonEngine(n_space_steps=200, n_time_steps=200)


def test_fd_matches_bs_for_european_call(market, process, fd_engine):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    bs_price = BlackScholesEngine().price(option, process, market).price
    fd_price = fd_engine.price(option, process, market).price
    assert fd_price == pytest.approx(bs_price, abs=0.02)


def test_fd_matches_bs_for_european_put(market, process, fd_engine):
    option = Option(payoff=Put(strike=100.0), expiry=1.0)
    bs_price = BlackScholesEngine().price(option, process, market).price
    fd_price = fd_engine.price(option, process, market).price
    assert fd_price == pytest.approx(bs_price, abs=0.02)


def test_fd_matches_bs_for_digital_call(market, process, fd_engine):
    option = Option(payoff=CashOrNothingCall(strike=100.0, cash=1.0), expiry=1.0)
    bs_price = BlackScholesEngine().price(option, process, market).price
    fd_price = fd_engine.price(option, process, market).price
    assert fd_price == pytest.approx(bs_price, abs=0.01)


def test_american_put_is_worth_at_least_european_put(market, process, fd_engine):
    european = Option(payoff=Put(strike=100.0), expiry=1.0)
    american = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())

    european_price = fd_engine.price(european, process, market).price
    american_price = fd_engine.price(american, process, market).price

    assert american_price >= european_price - 1e-6
    assert american_price > european_price  # early exercise premium should be strictly positive here


def test_american_put_never_below_intrinsic(market, process, fd_engine):
    option = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())
    result = fd_engine.price(option, process, market)
    assert result.price >= max(100.0 - market.spot, 0.0) - 1e-6


def test_fd_barrier_converges_to_discretely_monitored_monte_carlo(market, process, fd_engine):
    """FD enforces a *continuously* monitored barrier (any crossing knocks
    out); Monte Carlo only checks the barrier at n_steps discrete dates, so
    it misses paths that spike through and back between checks. Discretely
    monitored barrier options are therefore always priced *above* the
    continuous-barrier value (Broadie-Glasserman-Kou) — the two only agree
    as n_steps -> infinity, so this checks convergence in that direction
    rather than statistical agreement at a fixed n_steps.
    """
    option = Option(payoff=UpAndOutCall(strike=100.0, barrier=130.0), expiry=1.0)
    fd_price = fd_engine.price(option, process, market).price

    coarse = MonteCarloEngine(n_paths=100_000, n_steps=50, seed=11).price(option, process, market).price
    fine = MonteCarloEngine(n_paths=100_000, n_steps=1000, seed=11).price(option, process, market).price

    assert coarse > fine > fd_price  # monotonically approaching the continuous-barrier price from above
    assert fine == pytest.approx(fd_price, rel=0.03)


def test_fd_diagnostics_expose_the_price_surface(market, process, fd_engine):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    result = fd_engine.price(option, process, market)

    S_grid = result.diagnostics["S_grid"]
    t_grid = result.diagnostics["t_grid"]
    surface = result.diagnostics["V_surface"]

    assert surface.shape == (len(t_grid), len(S_grid))
    assert t_grid[0] == pytest.approx(0.0)
    assert t_grid[-1] == pytest.approx(option.expiry)
    # terminal slice (t = expiry) should be the payoff itself
    assert surface[-1] == pytest.approx(option.payoff(S_grid.reshape(-1, 1)))
