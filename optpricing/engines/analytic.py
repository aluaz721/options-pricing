import numpy as np
from scipy.stats import norm

from optpricing.engines.base import PricingEngine, PricingResult
from optpricing.instruments.exercise import European
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.payoffs.binary import CashOrNothingCall, CashOrNothingPut
from optpricing.payoffs.vanilla import Call, Put
from optpricing.processes.base import StochasticProcess
from optpricing.processes.gbm import GBM

# Payoffs priced here all share the same d1/d2 — they're the payoffs whose
# terminal value is a function of S_T alone and happens to integrate against
# the lognormal density in closed form under GBM. Barrier and Asian payoffs
# have closed forms too (Reiner-Rubinstein, Kemna-Vorst) but aren't wired up
# yet — this engine only covers the d1/d2 family so far.
_SUPPORTED_PAYOFFS = (Call, Put, CashOrNothingCall, CashOrNothingPut)


class BlackScholesEngine(PricingEngine):
    """Closed-form BSM (and the digital variants sharing its d1/d2). European
    exercise under GBM only — American has no closed form because the early-
    exercise boundary isn't known in advance.
    """

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        return (
            isinstance(process, GBM)
            and isinstance(option.exercise, European)
            and isinstance(option.payoff, _SUPPORTED_PAYOFFS)
        )

    def price(self, option: Option, process: StochasticProcess, market: MarketData) -> PricingResult:
        self._check_supported(option, process)
        # _check_supported already guarantees this at runtime; the assert
        # just narrows the type for readability/type-checkers below (process
        # is typed as the general StochasticProcess in the signature).
        assert isinstance(process, GBM)

        payoff = option.payoff
        S, K, T = market.spot, payoff.strike, option.expiry
        r, q, sigma = market.rate, market.dividend_yield, process.vol

        # Standard Black-Scholes-Merton formula with a continuous dividend
        # yield q (Merton's 1973 extension) — this is the only formula in the
        # library that assumes constant volatility in closed form.
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if isinstance(payoff, Call):
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif isinstance(payoff, Put):
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        elif isinstance(payoff, CashOrNothingCall):
            # P(S_T > K) under the risk-neutral measure, discounted — no S_T
            # term because the payout is fixed, so only N(d2) survives.
            price = payoff.cash * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = payoff.cash * np.exp(-r * T) * norm.cdf(-d2)

        return PricingResult(price=float(price), diagnostics={"d1": d1, "d2": d2})
