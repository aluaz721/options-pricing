from dataclasses import dataclass

import numpy as np

from optpricing.market import MarketData
from optpricing.processes.base import SimulatedPaths, StochasticProcess

# Andersen (2008) switching threshold between the two branches of the QE
# scheme (see _step_variance below). 1.5 is the value Andersen recommends
# after empirical testing
_PSI_CRITICAL = 1.5


@dataclass(frozen=True)
class Heston(StochasticProcess):
    v0: float  # initial variance
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    xi: float  # vol-of-vol
    rho: float  # correlation between spot and variance Brownian motions

    def simulate(
        self,
        market: MarketData,
        t: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = False,
    ) -> SimulatedPaths:
        """Andersen's (2008) Quadratic-Exponential (QE) scheme.

        The Heston SDEs are:
            dS = (r-q) S dt + sqrt(v) S dW^S
            dv = kappa(theta - v) dt + xi sqrt(v) dW^v,   corr(dW^S, dW^v) = rho

        The variance process v is a CIR (square-root) process. A naive Euler
        discretization of dv can send v negative (then sqrt(v) is undefined),
        which happens often in practice — most calibrated parameter sets
        violate the Feller condition 2*kappa*theta > xi^2 that would
        otherwise keep the *continuous-time* process strictly positive. QE
        sidesteps this by simulating v_{t+dt} from a distribution
        moment-matched to its true (known) conditional law instead of an
        Euler step, so it is positive by construction, not by clamping.
        """
        if antithetic:
            # GBM's antithetic variance reduction relies on log(S_T) being a
            # *linear, monotone* function of a single standard normal Z — Z
            # and -Z then land on opposite sides of the payoff by
            # construction. Here, v_{t+dt} is a nonlinear (and, in the
            # exponential branch, non-monotone) function of the randomness
            # driving it — see _step_variance — so mirroring that randomness
            # has no proven relationship to the resulting variance path, let
            # alone the terminal price. Rather than silently apply a
            # mirroring that might not help (or could even hurt), this
            # raises until a scheme with an actual variance-reduction
            # guarantee for Heston is implemented.
            raise NotImplementedError(
                "antithetic variates are not implemented for Heston — the QE scheme's "
                "variance step is a nonlinear function of its driving randomness, so naive "
                "Z -> -Z mirroring has no guaranteed variance-reduction benefit the way it "
                "does for GBM's linear log-price map"
            )
        dt = t / n_steps
        r, q = market.rate, market.dividend_yield
        kappa, theta, xi, rho = self.kappa, self.theta, self.xi, self.rho
        exp_kdt = np.exp(-kappa * dt)

        # --- log-price discretization coefficients (Andersen 2008, eq. 33) ---
        # Write X = ln(S). Conditional on the *pair* (v_t, v_{t+dt}) — both
        # already known once the variance step below has run — X_{t+dt} - X_t
        # is (to this discretization's approximation) Gaussian, with the
        # correlation rho absorbed entirely into the K1*v_t + K2*v_{t+dt}
        # term rather than into the noise driving X directly. That's what
        # makes this scheme need only one *extra*, independent standard
        # normal for X on top of whatever randomness produced v_{t+dt} — a
        # second, separately-correlated normal is not used, and would double
        # count rho if it were.
        #
        # gamma1/gamma2 interpolate between a fully "old" (gamma1=1) and
        # fully "new" (gamma2=1) trapezoidal approximation of the integrated
        # variance over [t, t+dt]; 0.5/0.5 (the "central" scheme) is
        # Andersen's standard recommendation.
        gamma1 = gamma2 = 0.5
        K0 = -rho * kappa * theta * dt / xi
        K1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
        K2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
        K3 = gamma1 * dt * (1.0 - rho**2)
        K4 = gamma2 * dt * (1.0 - rho**2)

        variance = np.empty((n_paths, n_steps + 1))
        log_spot = np.empty((n_paths, n_steps + 1))
        variance[:, 0] = self.v0
        log_spot[:, 0] = np.log(market.spot)

        for step in range(n_steps):
            v = variance[:, step]
            v_next = self._step_variance(v, exp_kdt, dt, rng)
            variance[:, step + 1] = v_next

            z = rng.standard_normal(n_paths)  # independent of whatever drove v_next — see note above
            log_spot[:, step + 1] = (
                log_spot[:, step]
                + (r - q) * dt
                + K0
                + K1 * v
                + K2 * v_next
                # K3*v + K4*v_next is a variance and is guaranteed >= 0 in
                # theory, but the discretization above is an approximation
                # and can occasionally produce a tiny negative value from
                # floating-point roundoff; clip rather than let sqrt emit NaN.
                + np.sqrt(np.maximum(K3 * v + K4 * v_next, 0.0)) * z
            )

        return SimulatedPaths(spot=np.exp(log_spot), extra={"variance": variance})

    def _step_variance(
        self, v: np.ndarray, exp_kdt: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        """One QE step v_t -> v_{t+dt} for the CIR variance process.

        The CIR process has a known conditional distribution (non-central
        chi-squared), but that's expensive to sample from exactly at scale.
        QE instead matches the *first two moments* of that true distribution
        with a cheap-to-sample family, switching between two regimes
        depending on how "peaked" the distribution is (measured by
        psi = Var/Mean^2, the squared coefficient of variation):

        - Low psi (v_t comfortably away from 0): the true distribution looks
          roughly Gaussian-squared-shaped, so QE fits it with (b + Z)^2 for a
          standard normal Z — always >= 0, and its mean/variance can be
          matched to the CIR moments exactly by solving for a, b.
        - High psi (v_t near 0, where the true distribution develops a spike
          at 0 plus an exponential-ish tail): a squared-Gaussian can't
          reproduce that shape, so QE instead uses a distribution that is
          exactly a point mass at 0 with probability p, and exponential
          otherwise — again moment-matched to the true mean/variance.

        Both branches are exact samples from *some* distribution with the
        right conditional mean and variance; the approximation is in using
        those two moments to stand in for the true (more complex) shape.
        """
        kappa, theta, xi = self.kappa, self.theta, self.xi

        # True conditional mean/variance of the CIR process (exact, not an
        # approximation — these come from the known moment-generating
        # function of a CIR/square-root diffusion).
        m = theta + (v - theta) * exp_kdt
        s2 = (
            v * xi**2 * exp_kdt / kappa * (1.0 - exp_kdt)
            + theta * xi**2 / (2.0 * kappa) * (1.0 - exp_kdt) ** 2
        )
        psi = s2 / m**2

        v_next = np.empty_like(v)
        quadratic = psi <= _PSI_CRITICAL

        if np.any(quadratic):
            psi_q = psi[quadratic]
            # Solve for b^2, a such that a*(b+Z)^2 has mean m and variance s2
            # (Andersen 2008, eq. 27) — a closed-form moment match, not a fit.
            b2 = 2.0 / psi_q - 1.0 + np.sqrt(2.0 / psi_q) * np.sqrt(2.0 / psi_q - 1.0)
            a = m[quadratic] / (1.0 + b2)
            z = rng.standard_normal(int(quadratic.sum()))
            v_next[quadratic] = a * (np.sqrt(b2) + z) ** 2

        if np.any(~quadratic):
            psi_e = psi[~quadratic]
            m_e = m[~quadratic]
            # p is the probability mass placed exactly at v=0; beta is the
            # rate of the exponential tail on (0, inf) — chosen so this
            # mixture's mean/variance again match m, s2 exactly (Andersen
            # 2008, eq. 29). Sampled here via inverse-CDF: draw U~Uniform(0,1),
            # map U <= p to the point mass at 0, otherwise invert the
            # exponential tail's CDF.
            p = (psi_e - 1.0) / (psi_e + 1.0)
            beta = (1.0 - p) / m_e
            u = rng.uniform(size=int((~quadratic).sum()))
            v_next[~quadratic] = np.where(u <= p, 0.0, np.log((1.0 - p) / (1.0 - u)) / beta)

        return v_next

    def characteristic_function(self, u: complex, t: float, market: MarketData) -> complex:
        # TODO: closed-form Heston CF (Heston 1993), needed by an FFT/COS
        # engine. Not required for Monte Carlo pricing, which only needs
        # simulate() above.
        raise NotImplementedError("Heston.characteristic_function is not implemented yet")
