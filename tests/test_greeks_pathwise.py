import numpy as np
import pytest
from scipy.stats import norm

from optpricing.engines import BlackScholesEngine
from optpricing.greeks import PathwiseGreeks
from optpricing.instruments import Option
from optpricing.payoffs import CashOrNothingCall, Call, Put


def _bs_d1_d2(market, process, option):
    result = BlackScholesEngine().price(option, process, market)
    return result.diagnostics["d1"], result.diagnostics["d2"]


def test_pathwise_delta_matches_closed_form_for_call(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    d1, _ = _bs_d1_d2(market, process, option)
    expected = np.exp(-market.dividend_yield * option.expiry) * norm.cdf(d1)

    greeks = PathwiseGreeks(n_paths=500_000, seed=1).compute(option, process, market)
    assert greeks.delta == pytest.approx(expected, abs=0.01)


def test_pathwise_delta_matches_closed_form_for_put(market, process):
    option = Option(payoff=Put(strike=100.0), expiry=1.0)
    d1, _ = _bs_d1_d2(market, process, option)
    expected = -np.exp(-market.dividend_yield * option.expiry) * norm.cdf(-d1)

    greeks = PathwiseGreeks(n_paths=500_000, seed=1).compute(option, process, market)
    assert greeks.delta == pytest.approx(expected, abs=0.01)


def test_pathwise_vega_matches_closed_form(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    d1, _ = _bs_d1_d2(market, process, option)
    expected = (
        market.spot
        * np.exp(-market.dividend_yield * option.expiry)
        * np.sqrt(option.expiry)
        * norm.pdf(d1)
    )

    greeks = PathwiseGreeks(n_paths=500_000, seed=2).compute(option, process, market)
    assert greeks.vega == pytest.approx(expected, abs=0.05)


def test_pathwise_rho_matches_closed_form_for_call(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    _, d2 = _bs_d1_d2(market, process, option)
    K, T = option.payoff.strike, option.expiry
    expected = K * T * np.exp(-market.rate * T) * norm.cdf(d2)

    greeks = PathwiseGreeks(n_paths=500_000, seed=3).compute(option, process, market)
    assert greeks.rho == pytest.approx(expected, abs=0.05)


def test_pathwise_has_no_gamma_or_theta(market, process):
    """Documented, not just an oversight: Gamma would need to differentiate
    the already-discontinuous indicator f' a second time, which pathwise
    can't do (that's exactly the gap MalliavinGreeks fills), and Theta
    doesn't fit the "smooth explicit parameter" mold the other three do.
    """
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    greeks = PathwiseGreeks(n_paths=1_000, seed=4).compute(option, process, market)
    assert greeks.gamma is None
    assert greeks.theta is None


def test_pathwise_rejects_discontinuous_payoff(market, process):
    """A digital's jump has no classical derivative — supports() must reject
    it rather than silently return a biased pathwise estimate.
    """
    option = Option(payoff=CashOrNothingCall(strike=100.0, cash=1.0), expiry=1.0)
    assert not PathwiseGreeks().supports(option, process)
    with pytest.raises(ValueError):
        PathwiseGreeks(n_paths=1_000).compute(option, process, market)
