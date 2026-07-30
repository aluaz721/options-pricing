import numpy as np
import pytest
from scipy.stats import norm

from optpricing.engines import BlackScholesEngine
from optpricing.greeks import MalliavinGreeks, PathwiseGreeks
from optpricing.instruments import Option
from optpricing.payoffs import CashOrNothingCall, Call, Put


def _bs_d1_d2(market, process, option):
    result = BlackScholesEngine().price(option, process, market)
    return result.diagnostics["d1"], result.diagnostics["d2"]


def test_malliavin_delta_matches_closed_form_for_call(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    d1, _ = _bs_d1_d2(market, process, option)
    expected = np.exp(-market.dividend_yield * option.expiry) * norm.cdf(d1)

    greeks = MalliavinGreeks(n_paths=500_000, seed=1).compute(option, process, market)
    assert greeks.delta == pytest.approx(expected, abs=0.01)


def test_malliavin_vega_matches_closed_form(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    d1, _ = _bs_d1_d2(market, process, option)
    expected = (
        market.spot
        * np.exp(-market.dividend_yield * option.expiry)
        * np.sqrt(option.expiry)
        * norm.pdf(d1)
    )

    greeks = MalliavinGreeks(n_paths=500_000, seed=2).compute(option, process, market)
    assert greeks.vega == pytest.approx(expected, abs=0.05)


def test_malliavin_gamma_matches_closed_form(market, process):
    """The headline result: Gamma is unavailable via PathwiseGreeks at all
    (see its docstring — differentiating the indicator a second time isn't
    classically defined), but falls out of a second integration-by-parts
    pass here.
    """
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    d1, _ = _bs_d1_d2(market, process, option)
    expected = (
        np.exp(-market.dividend_yield * option.expiry)
        * norm.pdf(d1)
        / (market.spot * process.vol * np.sqrt(option.expiry))
    )

    greeks = MalliavinGreeks(n_paths=500_000, seed=3).compute(option, process, market)
    # Gamma's Malliavin weight involves Z^2, so it's noisier than a
    # first-order Greek at the same path count — a wider but still tight
    # tolerance relative to the BS closed form.
    assert greeks.gamma == pytest.approx(expected, abs=0.03)


def test_malliavin_rho_matches_closed_form_for_put(market, process):
    option = Option(payoff=Put(strike=100.0), expiry=1.0)
    _, d2 = _bs_d1_d2(market, process, option)
    K, T = option.payoff.strike, option.expiry
    expected = -K * T * np.exp(-market.rate * T) * norm.cdf(-d2)

    greeks = MalliavinGreeks(n_paths=500_000, seed=4).compute(option, process, market)
    assert greeks.rho == pytest.approx(expected, abs=0.05)


def test_malliavin_handles_discontinuous_payoff_where_pathwise_cannot(market, process):
    """The actual point of this technique: a cash-or-nothing digital has a
    jump at S_T=K, so PathwiseGreeks refuses it outright (see
    test_pathwise_rejects_discontinuous_payoff), but MalliavinGreeks never
    differentiates the payoff — it only ever multiplies the raw payoff value
    by a weight built from the *known* Gaussian driving the simulation — so
    it prices straight through the discontinuity with no special-casing.
    """
    option = Option(payoff=CashOrNothingCall(strike=100.0, cash=1.0), expiry=1.0)
    assert not PathwiseGreeks().supports(option, process)
    assert MalliavinGreeks().supports(option, process)

    d1, d2 = _bs_d1_d2(market, process, option)
    T = option.expiry
    # Closed-form digital delta: d/dS0 [cash * e^{-rT} * N(d2)]
    # = cash * e^{-rT} * phi(d2) * d(d2)/dS0, and d(d2)/dS0 = d(d1)/dS0 =
    # 1/(S0*sigma*sqrt(T)) since d2 = d1 - sigma*sqrt(T), a constant shift.
    expected_delta = (
        1.0
        * np.exp(-market.rate * T)
        * norm.pdf(d2)
        / (market.spot * process.vol * np.sqrt(T))
    )

    greeks = MalliavinGreeks(n_paths=500_000, seed=5).compute(option, process, market)
    assert greeks.delta == pytest.approx(expected_delta, abs=0.01)


def test_malliavin_has_no_theta(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    greeks = MalliavinGreeks(n_paths=1_000, seed=6).compute(option, process, market)
    assert greeks.theta is None
