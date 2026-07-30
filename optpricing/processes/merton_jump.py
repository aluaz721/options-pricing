from dataclasses import dataclass

import numpy as np

from optpricing.market import MarketData
from optpricing.processes.base import SimulatedPaths, StochasticProcess


@dataclass(frozen=True)
class MertonJump(StochasticProcess):
    vol: float  # diffusive vol
    jump_intensity: float  # lambda, jumps per year
    jump_mean: float  # mean of log-jump size
    jump_std: float  # std of log-jump size

    def simulate(
        self,
        market: MarketData,
        t: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = False,
    ) -> SimulatedPaths:
        """Merton (1976): dS/S = (r-q-lambda*kappa_bar) dt + sigma dW + dJ,
        where J is a compound Poisson process — jumps arrive at rate lambda
        (per year), and each jump multiplies S by e^Y for Y ~ N(jump_mean,
        jump_std^2).

        Like GBM, this is simulated *exactly*, not discretized: over any
        interval dt (not just an infinitesimal one), the diffusive log-return
        is exactly Normal (same as GBM), the number of jumps in that
        interval is exactly Poisson(lambda*dt), and the sum of that many
        i.i.d. Normal(jump_mean, jump_std^2) jump sizes is again exactly
        Normal (sums of Gaussians are Gaussian). So there's no scheme to get
        wrong here the way Heston's variance process needs one — n_steps
        only controls path *resolution* for path-dependent payoffs, not
        terminal-distribution accuracy.

        kappa_bar = E[e^Y - 1] is the jump's average *relative* contribution
        to the price. Subtracting lambda*kappa_bar from the drift (the
        "compensator") keeps E[S_T] = S_0*e^{(r-q)T} despite the jumps —
        the same role q plays for dividends, but here compensating for the
        jump component's own average drift contribution so it doesn't leak
        into the risk-neutral drift.
        """
        dt = t / n_steps
        r, q = market.rate, market.dividend_yield
        sigma, lam, mu_j, sigma_j = self.vol, self.jump_intensity, self.jump_mean, self.jump_std

        # kappa_bar = E[e^Y - 1] for Y ~ N(mu_j, sigma_j^2): this is
        # exp(mu_j + 0.5*sigma_j^2) - 1, the standard lognormal mean formula
        # (the moment-generating function of a Normal, evaluated at 1).
        kappa_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        drift = (r - q - lam * kappa_bar - 0.5 * sigma**2) * dt

        if antithetic and n_paths % 2 != 0:
            raise ValueError("antithetic=True requires an even n_paths (paths are drawn in +Z/-Z pairs)")
        half = n_paths // 2

        log_paths = np.zeros((n_paths, n_steps + 1))
        for step in range(n_steps):
            # Only the diffusive Brownian increment is mirrored for
            # antithetic pairing — a Poisson jump *count* has no natural
            # "antithetic" partner the way a symmetric Normal does (there's
            # no analogue of "-Z" for a count of arrivals), so the jump
            # draws below stay independent between the two halves of a
            # pair. This still reduces variance whenever the diffusive
            # component is a meaningful share of the total (i.e. jumps
            # aren't overwhelmingly frequent/large) — just not as
            # completely as GBM's fully-mirrored antithetic variates.
            if antithetic:
                z_half = rng.standard_normal(half)
                z = np.concatenate([z_half, -z_half])
            else:
                z = rng.standard_normal(n_paths)
            diffusive = drift + sigma * np.sqrt(dt) * z

            n_jumps = rng.poisson(lam * dt, size=n_paths)
            # Sum of n_jumps i.i.d. N(mu_j, sigma_j^2) draws is exactly
            # N(n_jumps*mu_j, n_jumps*sigma_j^2) — sampled directly rather
            # than by actually looping over each jump. scale=0 (n_jumps=0)
            # is handled by numpy as a deterministic draw at loc, i.e. 0
            # contribution, exactly as it should be for a path with no
            # jumps this step.
            jump_sum = rng.normal(loc=n_jumps * mu_j, scale=np.sqrt(n_jumps) * sigma_j)

            log_paths[:, step + 1] = log_paths[:, step] + diffusive + jump_sum

        spot_paths = market.spot * np.exp(log_paths)
        return SimulatedPaths(spot=spot_paths)

    def characteristic_function(self, u: complex, t: float, market: MarketData) -> complex:
        # TODO: closed-form Merton CF, needed by an FFT/COS engine. Not
        # required for Monte Carlo pricing, which only needs simulate() above.
        raise NotImplementedError("MertonJump.characteristic_function is not implemented yet")
