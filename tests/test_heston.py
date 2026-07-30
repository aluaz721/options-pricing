import numpy as np
import pytest

from optpricing.engines import BlackScholesEngine, LongstaffSchwartzEngine, MonteCarloEngine
from optpricing.instruments import American, Option
from optpricing.market import MarketData
from optpricing.payoffs import Call, Put
from optpricing.processes import GBM, Heston


def test_qe_scheme_matches_known_cir_moments():
    """The QE scheme is a *moment-matched* sampler: v_{t+dt} is drawn from a
    distribution engineered to have exactly the CIR process's true
    conditional mean/variance (a known closed form), even though its shape
    is only an approximation of the true (non-central chi-squared)
    distribution. So the mean and variance of a large sample of one-step
    draws should match the closed-form CIR moments tightly — this checks
    _step_variance in isolation, independent of the log-price discretization.

    Feller condition (2*kappa*theta > xi^2) is deliberately violated here
    (2*2*0.04 = 0.16 < xi^2 = 0.36): this is exactly the regime a naive
    Euler discretization breaks in (frequent negative variance), and where
    the QE scheme's exponential branch actually gets exercised.
    """
    kappa, theta, xi, v0 = 2.0, 0.04, 0.6, 0.09
    process = Heston(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=-0.5)
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.0)
    T = 1.0
    n_paths = 300_000

    rng = np.random.default_rng(0)
    paths = process.simulate(market, T, n_steps=1, n_paths=n_paths, rng=rng)
    v_T = paths.extra["variance"][:, -1]

    assert np.all(v_T >= 0.0)  # the entire point of QE over naive Euler

    exp_kT = np.exp(-kappa * T)
    expected_mean = theta + (v0 - theta) * exp_kT
    expected_var = (
        v0 * xi**2 * exp_kT / kappa * (1 - exp_kT) + theta * xi**2 / (2 * kappa) * (1 - exp_kT) ** 2
    )

    sample_mean = v_T.mean()
    sample_stderr = v_T.std(ddof=1) / np.sqrt(n_paths)
    assert sample_mean == pytest.approx(expected_mean, abs=6 * sample_stderr)

    # Variance of a variance estimate is noisier than a mean estimate;
    # a looser relative tolerance is appropriate here.
    assert v_T.var(ddof=1) == pytest.approx(expected_var, rel=0.05)


def test_heston_collapses_to_black_scholes_as_vol_of_vol_vanishes():
    """With xi -> 0 and v0 = theta = sigma^2, the variance process barely
    moves away from sigma^2 over the option's life, so Heston should reduce
    to GBM(vol=sigma) — a full-pipeline check (QE variance step *and* the
    correlated log-price step together), unlike the moment test above which
    only checks the variance step in isolation.
    """
    sigma = 0.2
    market = MarketData(spot=100.0, rate=0.04, dividend_yield=0.01)
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    bs_price = BlackScholesEngine().price(option, GBM(vol=sigma), market).price

    heston = Heston(v0=sigma**2, kappa=2.0, theta=sigma**2, xi=1e-4, rho=-0.3)
    heston_result = MonteCarloEngine(n_paths=200_000, n_steps=100, seed=3).price(option, heston, market)

    assert heston_result.price == pytest.approx(bs_price, abs=6 * heston_result.std_error)


def test_heston_put_call_parity():
    """C - P = S*e^{-qT} - K*e^{-rT} is a model-independent identity (static
    replication, not dependent on GBM or any particular vol dynamics) — a
    good check that the simulation's discounting and drift are consistent,
    independent of whether the smile/skew this model produces is "correct."
    """
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
    heston = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.4, rho=-0.6)
    call = Option(payoff=Call(strike=100.0), expiry=1.0)
    put = Option(payoff=Put(strike=100.0), expiry=1.0)

    engine = MonteCarloEngine(n_paths=200_000, n_steps=100, seed=5)
    call_result = engine.price(call, heston, market)
    put_result = engine.price(put, heston, market)

    expected = market.spot * np.exp(-market.dividend_yield) - 100.0 * np.exp(-market.rate)
    combined_stderr = (call_result.std_error**2 + put_result.std_error**2) ** 0.5
    assert (call_result.price - put_result.price) == pytest.approx(expected, abs=6 * combined_stderr)


def test_negative_rho_produces_equity_style_skew():
    """Put-call parity and the GBM-collapse check above are both
    insensitive to the *sign* of rho (parity holds regardless, and xi -> 0
    makes correlation irrelevant), so neither actually tests that the
    correlation is wired up correctly rather than, say, flipped. This test
    targets that directly: the well-known "leverage effect" stylized fact is
    that rho < 0 (falling spot coincides with rising vol) fattens the left
    tail of the risk-neutral terminal distribution — so as rho decreases, an
    OTM put should get strictly more expensive and an OTM call strictly
    cheaper, holding the variance dynamics (v0, kappa, theta, xi) fixed.
    Forward/ATM value is unaffected by rho (parity is model-independent, and
    rho doesn't enter the variance SDE), so this isolates skew specifically.
    """
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.0)
    otm_put = Option(payoff=Put(strike=85.0), expiry=1.0)
    otm_call = Option(payoff=Call(strike=115.0), expiry=1.0)

    def price_both(rho: float, seed: int) -> tuple[float, float]:
        heston = Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=rho)
        put_price = MonteCarloEngine(n_paths=300_000, n_steps=100, seed=seed).price(otm_put, heston, market).price
        call_price = MonteCarloEngine(n_paths=300_000, n_steps=100, seed=seed).price(otm_call, heston, market).price
        return put_price, call_price

    put_neg, call_neg = price_both(-0.7, seed=42)
    put_zero, call_zero = price_both(0.0, seed=42)
    put_pos, call_pos = price_both(0.7, seed=42)

    assert put_neg > put_zero > put_pos
    assert call_neg < call_zero < call_pos


def test_longstaff_schwartz_prices_american_option_under_heston():
    """LongstaffSchwartzEngine never checks what process it's given — it
    only calls process.simulate() — so American exercise under Heston
    should work the moment Heston.simulate() exists, with no changes to the
    LSM engine itself. This is really a test of that architectural claim.
    """
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.0)
    heston = Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    option = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())

    result = LongstaffSchwartzEngine(n_paths=100_000, n_steps=50, seed=11).price(option, heston, market)

    intrinsic = max(100.0 - market.spot, 0.0)
    assert result.price >= intrinsic - 1e-6
    assert result.price > 0.0
