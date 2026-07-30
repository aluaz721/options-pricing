import pytest

from optpricing.engines import BlackScholesEngine, LongstaffSchwartzEngine, MonteCarloEngine
from optpricing.instruments import American, Option
from optpricing.market import MarketData
from optpricing.payoffs import Call, Put
from optpricing.processes import GBM, Heston, MertonJump


def test_antithetic_price_agrees_with_plain_monte_carlo(market, process):
    """Antithetic variates change *which* paths get simulated, not what
    they're simulating — both estimators are unbiased for the same price,
    so they should agree (well within their respective, larger-than-usual
    Monte Carlo noise) even though one is materially tighter than the other.
    """
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    bs_price = BlackScholesEngine().price(option, process, market).price

    plain = MonteCarloEngine(n_paths=200_000, n_steps=1, seed=1, antithetic=False).price(option, process, market)
    anti = MonteCarloEngine(n_paths=200_000, n_steps=1, seed=1, antithetic=True).price(option, process, market)

    assert plain.price == pytest.approx(bs_price, abs=4 * plain.std_error)
    assert anti.price == pytest.approx(bs_price, abs=4 * anti.std_error)


def test_antithetic_reduces_standard_error_for_gbm(market, process):
    """The actual point of the technique: at the *same* n_paths (i.e. the
    same simulation cost), antithetic pairing should give a visibly tighter
    standard error than independent draws, because payoff(Z) and
    payoff(-Z) are negatively correlated for a monotone payoff like a call.
    """
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    plain = MonteCarloEngine(n_paths=200_000, n_steps=1, seed=1, antithetic=False).price(option, process, market)
    anti = MonteCarloEngine(n_paths=200_000, n_steps=1, seed=1, antithetic=True).price(option, process, market)

    assert anti.std_error < plain.std_error


def test_antithetic_requires_even_n_paths(market, process):
    option = Option(payoff=Call(strike=100.0), expiry=1.0)
    with pytest.raises(ValueError):
        MonteCarloEngine(n_paths=100_001, n_steps=1, antithetic=True).price(option, process, market)


def test_heston_rejects_antithetic():
    """Heston explicitly refuses rather than silently ignoring the flag or
    applying a mirroring with no proven variance-reduction benefit — see
    Heston.simulate's docstring for why the QE scheme doesn't get this for
    free the way GBM's linear log-price map does.
    """
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.0)
    heston = Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5)
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    with pytest.raises(NotImplementedError):
        MonteCarloEngine(n_paths=10_000, n_steps=10, antithetic=True).price(option, heston, market)


def test_antithetic_reduces_standard_error_for_merton():
    """Merton only mirrors the diffusive Brownian increment (jump counts
    can't be mirrored the same way — see MertonJump.simulate's docstring),
    so the reduction should be more modest than GBM's, but still present
    given the diffusive component is a meaningful share of the variance here.
    """
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.0)
    process = MertonJump(vol=0.2, jump_intensity=0.5, jump_mean=-0.05, jump_std=0.1)
    option = Option(payoff=Call(strike=100.0), expiry=1.0)

    plain = MonteCarloEngine(n_paths=200_000, n_steps=20, seed=2, antithetic=False).price(option, process, market)
    anti = MonteCarloEngine(n_paths=200_000, n_steps=20, seed=2, antithetic=True).price(option, process, market)

    assert anti.std_error < plain.std_error


def test_antithetic_works_with_longstaff_schwartz(market, process):
    """LSM's exercise-decision loop touches every path individually but
    never reorders them, so the [Z-paths, -Z-paths] pairing convention
    survives to the final cash-flow array intact, and the same reduced
    standard error should show up there too.
    """
    option = Option(payoff=Put(strike=100.0), expiry=1.0, exercise=American())

    plain = LongstaffSchwartzEngine(n_paths=100_000, n_steps=50, seed=9, antithetic=False).price(
        option, process, market
    )
    anti = LongstaffSchwartzEngine(n_paths=100_000, n_steps=50, seed=9, antithetic=True).price(
        option, process, market
    )

    assert anti.price == pytest.approx(plain.price, abs=6 * max(plain.std_error, anti.std_error))
    assert anti.std_error < plain.std_error
