import numpy as np
import pytest

from optpricing.engines import BlackScholesEngine, LongstaffSchwartzEngine, MonteCarloEngine
from optpricing.instruments import American, Option
from optpricing.market import MarketData
from optpricing.payoffs import Call, Put
from optpricing.processes import GBM, MertonJump


def test_merton_compensator_preserves_the_risk_neutral_forward():
    """The whole point of subtracting lambda*kappa_bar from the drift is
    that jumps shouldn't change the risk-neutral forward — E[S_T] must still
    equal S_0*e^{(r-q)T} exactly, jumps or not. If the compensator formula
    were wrong, this would drift off silently rather than error loudly,
    so it's worth checking directly against simulated paths.
    """
    market = MarketData(spot=100.0, rate=0.04, dividend_yield=0.01)
    process = MertonJump(vol=0.15, jump_intensity=1.5, jump_mean=-0.1, jump_std=0.2)
    T = 1.0
    n_paths = 500_000

    rng = np.random.default_rng(0)
    paths = process.simulate(market, T, n_steps=50, n_paths=n_paths, rng=rng)
    S_T = paths.spot[:, -1]

    expected_mean = market.spot * np.exp((market.rate - market.dividend_yield) * T)
    stderr = S_T.std(ddof=1) / np.sqrt(n_paths)
    assert S_T.mean() == pytest.approx(expected_mean, abs=6 * stderr)


def test_merton_collapses_to_black_scholes_without_jumps():
    """jump_intensity=0 should make MertonJump statistically indistinguishable
    from GBM(vol=sigma) — the compound Poisson component contributes exactly
    zero jumps almost surely, and the compensator term is exactly zero too
    (lambda=0), so nothing but the pure diffusion survives.
    """
    sigma = 0.2
    market = MarketData(spot=100.0, rate=0.04, dividend_yield=0.01)
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    bs_price = BlackScholesEngine().price(option, GBM(vol=sigma), market).price

    merton = MertonJump(vol=sigma, jump_intensity=0.0, jump_mean=-0.1, jump_std=0.2)
    result = MonteCarloEngine(n_paths=300_000, n_steps=50, seed=3).price(option, merton, market)

    assert result.price == pytest.approx(bs_price, abs=6 * result.std_error)


def test_jumps_increase_out_of_the_money_call_value():
    """Holding the diffusive vol fixed and only turning on jumps adds pure
    variance on top of the diffusive part (the compensator keeps the mean/
    forward unchanged — see the martingale test above), so a deep
    out-of-the-money call, whose value is driven by tail probability, should
    be strictly more valuable with jumps than without. This isolates "jumps
    add tail risk" from "jumps change the mean," which the compensator
    already rules out.
    """
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.0)
    option = Option(payoff=Call(strike=140.0), expiry=1.0)

    no_jumps = MertonJump(vol=0.15, jump_intensity=0.0, jump_mean=0.0, jump_std=0.2)
    with_jumps = MertonJump(vol=0.15, jump_intensity=1.0, jump_mean=0.0, jump_std=0.3)

    engine = MonteCarloEngine(n_paths=300_000, n_steps=50, seed=7)
    price_no_jumps = engine.price(option, no_jumps, market).price
    price_with_jumps = engine.price(option, with_jumps, market).price

    assert price_with_jumps > price_no_jumps


def test_merton_put_call_parity():
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
    process = MertonJump(vol=0.15, jump_intensity=1.0, jump_mean=-0.1, jump_std=0.25)
    call = Option(payoff=Call(strike=100.0), expiry=1.0)
    put = Option(payoff=Put(strike=100.0), expiry=1.0)

    engine = MonteCarloEngine(n_paths=300_000, n_steps=50, seed=5)
    call_result = engine.price(call, process, market)
    put_result = engine.price(put, process, market)

    expected = market.spot * np.exp(-market.dividend_yield) - 100.0 * np.exp(-market.rate)
    combined_stderr = (call_result.std_error**2 + put_result.std_error**2) ** 0.5
    assert (call_result.price - put_result.price) == pytest.approx(expected, abs=6 * combined_stderr)


def test_longstaff_schwartz_prices_american_option_under_merton():
    """Same architectural point as the Heston version of this test: LSM only
    calls process.simulate(), so American exercise under jump-diffusion
    needs zero LSM changes, just a process that can simulate itself.
    """
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.0)
    process = MertonJump(vol=0.15, jump_intensity=1.0, jump_mean=-0.1, jump_std=0.25)
    option = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())

    result = LongstaffSchwartzEngine(n_paths=100_000, n_steps=50, seed=11).price(option, process, market)

    intrinsic = max(100.0 - market.spot, 0.0)
    assert result.price >= intrinsic - 1e-6
    assert result.price > 0.0
