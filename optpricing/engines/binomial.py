import numpy as np

from optpricing.engines.base import PricingEngine, PricingResult
from optpricing.instruments.exercise import American, European
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.payoffs.vanilla import Call, Put
from optpricing.processes.base import StochasticProcess
from optpricing.processes.gbm import GBM


class BinomialTreeEngine(PricingEngine):
    """Cox-Ross-Rubinstein (1979) binomial tree. European/American vanilla
    under GBM only — early exercise is the whole point of reaching for a
    tree here, and it gives an independent cross-check on CrankNicolsonEngine's
    Brennan-Schwartz American pricing (different numerical method entirely,
    same PDE limit as n_steps -> infinity).
    """

    def __init__(self, n_steps: int = 500):
        self.n_steps = n_steps

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        return (
            isinstance(process, GBM)
            and isinstance(option.exercise, (European, American))
            and isinstance(option.payoff, (Call, Put))
        )

    def price(self, option: Option, process: StochasticProcess, market: MarketData) -> PricingResult:
        self._check_supported(option, process)
        assert isinstance(process, GBM)

        N = self.n_steps
        dt = option.expiry / N
        sigma, r, q = process.vol, market.rate, market.dividend_yield

        # CRR's specific choice of up/down factors: u = e^{sigma*sqrt(dt)},
        # d = 1/u. This isn't the only valid choice (any u,d with the right
        # local mean/variance works), but it's the one that makes the tree
        # recombine (u*d = 1, so "up then down" lands on the same node as
        # "down then up") and converges to the GBM diffusion limit as
        # dt -> 0, by construction: matching the first two moments of the
        # binomial step to sigma^2*dt is exactly what pins down this u.
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u

        # Risk-neutral probability: the unique p making the discounted
        # expected return match the risk-neutral drift, e^{(r-q)dt} = p*u + (1-p)*d.
        # This is *not* a real-world probability of an up-move — it's a
        # pricing device, same role as the risk-neutral measure in the
        # continuous-time model. No-arbitrage between the two branches
        # requires d < e^{(r-q)dt} < u, which holds here for dt small enough.
        p = (np.exp((r - q) * dt) - d) / (u - d)
        discount = np.exp(-r * dt)

        is_call = isinstance(option.payoff, Call)
        is_american = isinstance(option.exercise, American)

        # Terminal spot at node i (0..N up-moves out of N steps, i up and
        # N-i down): S_i = S_0 * u^i * d^(N-i). Evaluated via the Payoff
        # interface itself (paths of length 1, i.e. "already at expiry") so
        # the payoff formula lives in exactly one place, not duplicated here.
        i = np.arange(N + 1)
        S = market.spot * u**i * d ** (N - i)
        V = option.payoff(S.reshape(-1, 1))

        # Backward induction: at each step, node i (0..step) represents i
        # up-moves out of `step` so far. V[i] is currently valued at step+1;
        # discount the risk-neutral expectation of its two children back one
        # step. For American, compare against immediate exercise at that
        # node — this is the tree's version of the same free-boundary
        # problem CrankNicolsonEngine solves via Brennan-Schwartz, and
        # doesn't need any special-casing because it's just an elementwise
        # max applied at every node on the way back.
        for step in range(N - 1, -1, -1):
            V = discount * (p * V[1:] + (1 - p) * V[:-1])
            if is_american:
                i = np.arange(step + 1)
                S = market.spot * u**i * d ** (step - i)
                intrinsic = np.maximum(S - option.payoff.strike, 0.0) if is_call else np.maximum(
                    option.payoff.strike - S, 0.0
                )
                V = np.maximum(V, intrinsic)

        return PricingResult(price=float(V[0]), diagnostics={"u": u, "d": d, "p": p})
