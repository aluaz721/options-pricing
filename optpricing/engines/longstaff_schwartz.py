import numpy as np

from optpricing.engines._variance_reduction import antithetic_aware_stderr
from optpricing.engines.base import PricingEngine, PricingResult
from optpricing.instruments.exercise import American
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.payoffs.vanilla import Call, Put
from optpricing.processes.base import StochasticProcess


class LongstaffSchwartzEngine(PricingEngine):
    """Least-Squares Monte Carlo (Longstaff & Schwartz, 2001) for American
    exercise.

    CrankNicolsonEngine and BinomialTreeEngine solve the free-boundary
    problem by working backward on a fixed grid/lattice, where every node
    has a well-defined set of "future" nodes to take an expectation over.
    Simulated Monte Carlo paths don't have that structure — each path is an
    independent draw, so there's no shared grid to do backward induction on
    directly. LSM's trick is to still work backward through the *time
    steps* (not the paths), and at each step estimate the continuation value
    — "what is it worth to keep holding, on average, given where the
    underlying is right now" — via a cross-sectional least-squares
    regression of realized future payoffs against a polynomial in the
    current spot. That regression is standing in for the conditional
    expectation a PDE/lattice method gets for free from its grid structure.

    This is the only American-exercise engine in the library that doesn't
    care what process it's pricing under: it only calls process.simulate(),
    so American exercise under Heston or a multi-asset process would need
    zero changes to this engine, just a process that can simulate itself.
    Only vanilla Call/Put are wired up (the intrinsic-value formula is
    hard-coded per payoff type below, the same pattern CrankNicolsonEngine
    uses) — supporting arbitrary American exotics would need a
    Payoff.exercise_value(S) hook that doesn't exist yet.
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 50,
        basis_degree: int = 2,
        seed: int | None = None,
        antithetic: bool = False,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.basis_degree = basis_degree
        self.seed = seed
        self.antithetic = antithetic

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        return isinstance(option.exercise, American) and isinstance(option.payoff, (Call, Put))

    def price(self, option: Option, process: StochasticProcess, market: MarketData) -> PricingResult:
        self._check_supported(option, process)

        rng = np.random.default_rng(self.seed)
        paths = process.simulate(
            market, option.expiry, self.n_steps, self.n_paths, rng, antithetic=self.antithetic
        ).spot
        # paths[:, k] is the simulated spot at time k*dt, k = 0..n_steps;
        # paths[:, 0] is today's known spot (not random).

        dt = option.expiry / self.n_steps
        discount = np.exp(-market.rate * dt)
        K = option.payoff.strike
        is_call = isinstance(option.payoff, Call)

        def intrinsic(S: np.ndarray) -> np.ndarray:
            return np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)

        # cash_flow[j] tracks, for path j, the (running) present value *as of
        # the time step currently being examined* of whatever that path will
        # actually receive under the exercise policy decided so far. It
        # starts as the payoff at expiry (the value if never exercised
        # early) and gets overwritten with the exercise value on any step
        # where that path is found to exercise.
        cash_flow = intrinsic(paths[:, -1])

        # Walk backward from the second-to-last step to the first (t=0 is
        # excluded: it's the valuation date with a single known spot, not a
        # simulated decision to regress over — handled separately below).
        for step in range(self.n_steps - 1, 0, -1):
            S = paths[:, step]
            cash_flow = cash_flow * discount  # roll the future cash flow back one step
            exercise_value = intrinsic(S)
            itm = exercise_value > 0.0  # only in-the-money paths are candidates for exercise

            # Regressing on OTM paths would waste fitting power on curvature
            # that never affects a decision (you'd never exercise there
            # anyway), and is the source of the well-documented instability
            # in naive "regress on everything" implementations — this is
            # Longstaff & Schwartz's own restriction, not an approximation
            # on top of their method.
            n_itm = int(itm.sum())
            if n_itm > self.basis_degree + 1:
                S_itm = S[itm]
                # Center/scale before building the polynomial basis: raw
                # powers of S (~100, ~10000, ~1e6, ...) are badly conditioned
                # for least squares once basis_degree gets past 2 or 3;
                # standardizing first keeps the design matrix well-scaled
                # regardless of the spot level or degree chosen.
                mean, std = S_itm.mean(), S_itm.std()
                S_scaled = (S_itm - mean) / std if std > 0 else S_itm - mean
                basis = np.vander(S_scaled, self.basis_degree + 1, increasing=True)

                coeffs, *_ = np.linalg.lstsq(basis, cash_flow[itm], rcond=None)
                continuation_value = basis @ coeffs

                exercise_now = exercise_value[itm] > continuation_value
                itm_indices = np.flatnonzero(itm)
                cash_flow[itm_indices[exercise_now]] = exercise_value[itm][exercise_now]
            # else: too few ITM paths to fit a stable regression at this
            # step — leave cash_flow untouched, i.e. assume continuation
            # (the conservative default when there isn't enough information
            # to estimate the exercise boundary).

        cash_flow = cash_flow * discount  # discount from t=dt back to t=0

        # At t=0 there's a single known spot, not a distribution — no
        # regression needed, just compare the simulated continuation value
        # against the day-0 intrinsic value directly.
        price = max(float(cash_flow.mean()), float(intrinsic(np.array(market.spot))))
        # The exercise decisions above never reorder paths, so cash_flow
        # still has the [Z-paths, -Z-paths] pairing StochasticProcess.simulate
        # promises when antithetic=True — antithetic_aware_stderr can use it
        # directly, same as MonteCarloEngine does.
        std_error = antithetic_aware_stderr(cash_flow, self.antithetic)

        return PricingResult(price=price, std_error=std_error)
