import numpy as np

from optpricing.greeks.base import Greeks, GreeksEngine
from optpricing.instruments.exercise import European
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.payoffs.vanilla import Call, Put
from optpricing.processes.base import StochasticProcess
from optpricing.processes.gbm import GBM


class PathwiseGreeks(GreeksEngine):
    """Pathwise-derivative Monte Carlo Greeks for GBM (Call/Put only).

    The idea: differentiate the discounted payoff along each simulated path
    with respect to a parameter theta, then average — instead of bumping
    theta and repricing twice. Since S_T = S_0 * exp((r-q-0.5sigma^2)T +
    sigma*sqrt(T)*Z) for a fixed standard normal Z, S_T is a smooth,
    explicit function of every parameter, so d(discounted payoff)/dtheta can
    be written down directly via the chain rule:

        d/dtheta [e^{-rT} f(S_T)] = e^{-rT} f'(S_T) * dS_T/dtheta   (+ a
        -T*e^{-rT}*f(S_T) term for theta=r, since r also appears in the
        discount factor itself, not just inside S_T)

    This needs f' (the payoff's derivative), which is why supports() only
    allows Call/Put: their payoff is kinked at S_T=K but still differentiable
    almost everywhere (the indicator 1{S_T>K} is a perfectly good f' except
    at the single point S_T=K, which a continuous distribution hits with
    probability zero). A *discontinuous* payoff like a cash-or-nothing
    digital doesn't have this luxury — its "derivative" at the jump is a
    Dirac delta, not a number a Monte Carlo path can sample — which is
    exactly the case MalliavinGreeks exists for instead.

    No Gamma here for the same reason one level up: Gamma would need to
    differentiate the *indicator* f', which has a genuine jump discontinuity
    at S_T=K (not just a kink), so it hits the same wall pathwise Delta
    dodges by only needing f' rather than f''. No Theta either — unlike
    Delta/Vega/Rho, time-to-expiry enters the discretization itself (n
    simulated steps over [0,T]), not just a smooth explicit parameter, so
    there's no analogous single-path derivative to take without resimulating.
    """

    def __init__(self, n_paths: int = 200_000, seed: int | None = None):
        self.n_paths = n_paths
        self.seed = seed

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        return (
            isinstance(process, GBM)
            and isinstance(option.exercise, European)
            and isinstance(option.payoff, (Call, Put))
        )

    def compute(self, option: Option, process: StochasticProcess, market: MarketData) -> Greeks:
        if not self.supports(option, process):
            raise ValueError(
                f"PathwiseGreeks cannot differentiate a {type(option.payoff).__name__} "
                "(needs a differentiable payoff) under this process/exercise combination"
            )
        assert isinstance(process, GBM)

        rng = np.random.default_rng(self.seed)
        T = option.expiry
        sigma, r, q, S0 = process.vol, market.rate, market.dividend_yield, market.spot
        sqrt_T = np.sqrt(T)

        z = rng.standard_normal(self.n_paths)
        S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * sqrt_T * z)

        K = option.payoff.strike
        is_call = isinstance(option.payoff, Call)
        # f'(S_T): the indicator is +1 where the option finishes in the
        # money (payoff slope is 1) and 0 otherwise, for a call; a put is
        # the mirror image with slope -1 in the money.
        f_prime = (S_T > K).astype(float) if is_call else -(S_T < K).astype(float)
        payoff = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)

        discount = np.exp(-r * T)
        price = discount * payoff.mean()

        # dS_T/dS_0 = S_T/S_0 (S_0 only enters multiplicatively out front).
        delta = discount * np.mean(f_prime * S_T / S0)

        # dS_T/dsigma: differentiate the exponent (r-q-0.5sigma^2)T +
        # sigma*sqrt(T)*Z with respect to sigma, giving (-sigma*T + sqrt(T)*Z);
        # S_T's derivative picks up that factor via the chain rule (d/dx e^x = e^x).
        d_exponent_d_sigma = sqrt_T * z - sigma * T
        vega = discount * np.mean(f_prime * S_T * d_exponent_d_sigma)

        # dS_T/dr = S_T * T (r appears as +r*T in the exponent), *and* r
        # appears again in the discount factor e^{-rT} multiplying the whole
        # expectation — product rule across both occurrences:
        #   d/dr [e^{-rT} E[f(S_T)]] = -T*e^{-rT}*E[f(S_T)] + e^{-rT}*E[f'(S_T)*S_T*T]
        rho = -T * price + discount * T * np.mean(f_prime * S_T)

        return Greeks(delta=delta, gamma=None, vega=vega, theta=None, rho=rho)
