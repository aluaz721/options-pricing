import numpy as np

from optpricing.greeks.base import Greeks, GreeksEngine
from optpricing.instruments.exercise import European
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.processes.base import StochasticProcess
from optpricing.processes.gbm import GBM


class MalliavinGreeks(GreeksEngine):
    """Malliavin-weight ("likelihood ratio method") Monte Carlo Greeks for
    GBM, for any path-independent European payoff.

    PathwiseGreeks differentiates the *payoff*: it needs f'(S_T), which
    doesn't exist at a jump (a digital's discontinuity) and only exists in a
    distributional sense for Gamma (differentiating the already-discontinuous
    indicator again). Malliavin weights sidestep this by never differentiating
    f at all — instead they differentiate the *density* Z is drawn from, via
    an integration-by-parts identity on the Gaussian, and move the derivative
    onto a weight that multiplies the raw payoff f(S_T). Since f itself is
    never touched, this works for any payoff that depends on S_T alone —
    Call, Put, and a discontinuous digital all get treated identically here,
    which is exactly the case this technique is for.

    Derivation sketch (Fournié et al. 1999; see also Glasserman, "Monte
    Carlo Methods in Financial Engineering", ch. 7). Write S_T =
    S_0*exp(mu(theta) + s(theta)*Z) for a standard normal Z, where the
    parameter theta = S_0, sigma, or r enters only through the mean mu and
    scale s of the log-price. Stein's identity for a standard normal,
    E[Z*h(Z)] = E[h'(Z)] (and its second-order sibling E[Z*h'(Z)] =
    E[(Z^2-1)*h(Z)]), turns

        d/dtheta E[f(S_T)] = E[f'(S_T)*S_T*(mu'(theta) + s'(theta)*Z)]

    (a *pathwise* derivative, needing f') into the equivalent

        d/dtheta E[f(S_T)] = E[f(S_T) * ( (mu'/s)*Z + (s'/s)*(Z^2-1) )]

    which needs no derivative of f at all — just f(S_T) itself, multiplied
    by a "weight" built purely from Z and the known dependence of (mu, s) on
    theta. Applying this once gives Delta/Vega/Rho; applying the same
    integration by parts a second time (differentiating the *weight* itself
    the same way) gives Gamma, which is where this method's payoff-agnosticism
    really pays for itself — Gamma is unavailable via PathwiseGreeks at all.

    No Theta: unlike Delta/Vega/Rho, time-to-expiry enters the simulation's
    discretization itself, not just as a smooth explicit parameter of a
    single terminal draw, so the same trick doesn't directly apply.
    """

    def __init__(self, n_paths: int = 200_000, seed: int | None = None):
        self.n_paths = n_paths
        self.seed = seed

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        return (
            isinstance(process, GBM)
            and isinstance(option.exercise, European)
            and not option.payoff.is_path_dependent
        )

    def compute(self, option: Option, process: StochasticProcess, market: MarketData) -> Greeks:
        if not self.supports(option, process):
            raise ValueError(
                f"MalliavinGreeks needs a path-independent European payoff under GBM; "
                f"got {type(option.payoff).__name__} / {type(option.exercise).__name__} "
                f"/ {type(process).__name__}"
            )
        assert isinstance(process, GBM)

        rng = np.random.default_rng(self.seed)
        T = option.expiry
        sigma, r, q, S0 = process.vol, market.rate, market.dividend_yield, market.spot
        sqrt_T = np.sqrt(T)

        z = rng.standard_normal(self.n_paths)
        S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * sqrt_T * z)
        # Same Payoff interface every other engine uses (paths of length 1,
        # i.e. "already at the terminal date") — the weights below multiply
        # this raw payoff value, never a derivative of it.
        payoff = option.payoff(S_T.reshape(-1, 1))

        discount = np.exp(-r * T)
        price = discount * payoff.mean()

        # Delta: theta=S_0 enters only through mu = ln(S_0) + ..., so
        # mu'=1/S_0 and s (=sigma*sqrt(T)) doesn't depend on S_0 at all
        # (s'=0). Weight = (mu'/s)*Z = Z / (S_0*sigma*sqrt(T)).
        weight_delta = z / (S0 * sigma * sqrt_T)
        delta = discount * np.mean(payoff * weight_delta)

        # Vega: theta=sigma enters both mu (mu'=-sigma*T) and s (s'=sqrt(T)).
        # Weight = (mu'/s)*Z + (s'/s)*(Z^2-1) = -sqrt(T)*Z + (Z^2-1)/sigma.
        weight_vega = (z**2 - 1) / sigma - z * sqrt_T
        vega = discount * np.mean(payoff * weight_vega)

        # Gamma: apply the same integration-by-parts trick a second time to
        # the Delta weight itself (see class docstring) — the derivation is
        # in the module's accompanying tests/comments rather than repeated
        # inline; the result is the standard second-order weight from
        # Fournié et al. / Glasserman ch. 7.
        weight_gamma = ((z**2 - 1) / (sigma**2 * T) - z / (sigma * sqrt_T)) / S0**2
        gamma = discount * np.mean(payoff * weight_gamma)

        # Rho: r enters both mu (mu'=T, since S_T has a +r*T term) *and* the
        # discount factor directly — same product-rule split as
        # PathwiseGreeks.rho, just with the weighted E[payoff*weight]
        # standing in for the pathwise E[f'(S_T)*S_T*T] term.
        weight_rho = sqrt_T * z / sigma
        rho = -T * price + discount * np.mean(payoff * weight_rho)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=None, rho=rho)
