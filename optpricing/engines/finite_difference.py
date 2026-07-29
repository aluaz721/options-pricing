import numpy as np

from optpricing.engines.base import PricingEngine, PricingResult, UnsupportedCombination
from optpricing.instruments.exercise import American, European
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.payoffs.binary import CashOrNothingCall, CashOrNothingPut
from optpricing.payoffs.path_dependent import UpAndOutCall
from optpricing.payoffs.vanilla import Call, Put
from optpricing.processes.base import StochasticProcess
from optpricing.processes.gbm import GBM


def _thomas_solve(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray, floor: np.ndarray | None = None
) -> np.ndarray:
    """Tridiagonal solve via the Thomas algorithm (O(n), no iteration).

    If `floor` is given, each unknown is clipped to it during back-
    substitution — the Brennan-Schwartz (1977) trick for American options.
    It turns a free-boundary/optimal-stopping problem into a single forward-
    elimination + backward-substitution sweep, instead of an iterative
    projected solve (PSOR) that has to be run to convergence at every time
    step. It relies on the payoff being the kind of monotone obstacle a
    vanilla American put/call is; it isn't a general LCP solver.
    """
    n = len(diag)
    c_star = np.empty(n)
    d_star = np.empty(n)
    c_star[0] = upper[0] / diag[0]
    d_star[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c_star[i - 1]
        c_star[i] = upper[i] / denom
        d_star[i] = (rhs[i] - lower[i] * d_star[i - 1]) / denom

    x = np.empty(n)
    x[-1] = d_star[-1] if floor is None else max(floor[-1], d_star[-1])
    for i in range(n - 2, -1, -1):
        value = d_star[i] - c_star[i] * x[i + 1]
        x[i] = value if floor is None else max(floor[i], value)
    return x


class CrankNicolsonEngine(PricingEngine):
    """Crank-Nicolson finite differences on the Black-Scholes PDE, in S-space.

    Covers European and American vanilla (Call/Put) — American via Brennan-
    Schwartz — plus European-only up-and-out calls and cash-or-nothing
    digitals. Barrier support isn't a different PDE, just a different grid
    domain: capping S_max at the barrier with a zero Dirichlet boundary there
    *is* the knock-out condition. Heston (2D ADI) and general LCP-based
    American barriers are out of scope for this single-factor grid.
    """

    def __init__(self, n_space_steps: int = 200, n_time_steps: int = 200, s_max_multiple: float = 3.0):
        self.n_space_steps = n_space_steps
        self.n_time_steps = n_time_steps
        self.s_max_multiple = s_max_multiple

    def supports(self, option: Option, process: StochasticProcess) -> bool:
        if not isinstance(process, GBM):
            return False
        payoff = option.payoff
        if isinstance(payoff, (Call, Put)):
            return isinstance(option.exercise, (European, American))
        if isinstance(payoff, (UpAndOutCall, CashOrNothingCall, CashOrNothingPut)):
            return isinstance(option.exercise, European)
        return False

    def price(self, option: Option, process: StochasticProcess, market: MarketData) -> PricingResult:
        self._check_supported(option, process)
        assert isinstance(process, GBM)

        M, N, T = self.n_space_steps, self.n_time_steps, option.expiry
        sigma, r, q = process.vol, market.rate, market.dividend_yield
        is_american = isinstance(option.exercise, American)

        S_max, terminal, boundary_lower, boundary_upper = self._grid_setup(option.payoff, market)
        dt = T / N
        S = np.linspace(0.0, S_max, M + 1)
        W = terminal(S)  # W(S, tau) is price as a function of time-to-maturity tau = T - t
        intrinsic = W.copy()  # constant in tau for standard vanilla; only used if is_american

        # Node index i cancels dS out of the discretized coefficients (S_i =
        # i*dS), so these depend only on i, not on the grid spacing.
        i = np.arange(1, M)
        a = 0.5 * sigma**2 * i**2 - 0.5 * (r - q) * i
        b = -(sigma**2) * i**2 - r
        c = 0.5 * sigma**2 * i**2 + 0.5 * (r - q) * i

        # Crank-Nicolson = average of the implicit and explicit Euler updates
        # in tau: (I - 0.5*dt*L) W^{n+1} = (I + 0.5*dt*L) W^n, tridiagonal in
        # both operators since L only couples neighboring grid points.
        lower_lhs, diag_lhs, upper_lhs = -0.5 * dt * a, 1 - 0.5 * dt * b, -0.5 * dt * c
        lower_rhs, diag_rhs, upper_rhs = 0.5 * dt * a, 1 + 0.5 * dt * b, 0.5 * dt * c

        t_grid = np.empty(N + 1)
        surface = np.empty((N + 1, M + 1))
        t_grid[0], surface[0] = T, W  # tau=0 is expiry, i.e. calendar time t=T

        for n in range(1, N + 1):
            tau = n * dt
            w0, wM = boundary_lower(tau), boundary_upper(tau)

            rhs = lower_rhs * W[: M - 1] + diag_rhs * W[1:M] + upper_rhs * W[2 : M + 1]
            # Boundary values are known, not unknowns in the tridiagonal
            # system — fold their contribution into the RHS of the first/last
            # interior equations instead of solving for them.
            rhs[0] += 0.5 * dt * a[0] * w0
            rhs[-1] += 0.5 * dt * c[-1] * wM

            floor = intrinsic[1:M] if is_american else None
            interior = _thomas_solve(lower_lhs, diag_lhs, upper_lhs, rhs, floor)
            W = np.concatenate(([w0], interior, [wM]))

            t_grid[n], surface[n] = T - tau, W

        price = float(np.interp(market.spot, S, W))

        # t_grid was built decreasing (tau increases from 0); sort ascending
        # in calendar time so callers (e.g. a 3D plot) get t=0..T left-to-right.
        order = np.argsort(t_grid)
        diagnostics = {"S_grid": S, "t_grid": t_grid[order], "V_surface": surface[order]}
        return PricingResult(price=price, diagnostics=diagnostics)

    def _grid_setup(self, payoff, market: MarketData):
        """Turn a payoff's economics into a concrete grid domain (S_max), a
        terminal condition W(S, tau=0), and Dirichlet boundary conditions at
        S=0 and S=S_max as functions of tau. This is the one place a payoff's
        contractual details get translated into PDE boundary data — the
        payoff classes themselves stay numerics-agnostic.
        """
        r, q = market.rate, market.dividend_yield

        if isinstance(payoff, Call):
            K = payoff.strike
            S_max = self.s_max_multiple * max(market.spot, K)
            return (
                S_max,
                lambda S: np.maximum(S - K, 0.0),
                lambda tau: 0.0,
                lambda tau: S_max * np.exp(-q * tau) - K * np.exp(-r * tau),
            )
        if isinstance(payoff, Put):
            K = payoff.strike
            S_max = self.s_max_multiple * max(market.spot, K)
            return (
                S_max,
                lambda S: np.maximum(K - S, 0.0),
                lambda tau: K * np.exp(-r * tau),
                lambda tau: 0.0,
            )
        if isinstance(payoff, UpAndOutCall):
            K, barrier = payoff.strike, payoff.barrier
            # The domain simply stops at the barrier: V(barrier, t) = 0 for
            # all t *is* the knock-out condition, not an approximation of it.
            return (
                barrier,
                lambda S: np.where(S < barrier, np.maximum(S - K, 0.0), 0.0),
                lambda tau: 0.0,
                lambda tau: 0.0,
            )
        if isinstance(payoff, CashOrNothingCall):
            K, cash = payoff.strike, payoff.cash
            S_max = self.s_max_multiple * max(market.spot, K)
            return (
                S_max,
                lambda S: np.where(S > K, cash, 0.0),
                lambda tau: 0.0,
                lambda tau: cash * np.exp(-r * tau),
            )
        if isinstance(payoff, CashOrNothingPut):
            K, cash = payoff.strike, payoff.cash
            S_max = self.s_max_multiple * max(market.spot, K)
            return (
                S_max,
                lambda S: np.where(S < K, cash, 0.0),
                lambda tau: cash * np.exp(-r * tau),
                lambda tau: 0.0,
            )
        # Unreachable given supports() already filtered payoff types; kept as
        # a defensive check rather than silently mispricing.
        raise UnsupportedCombination(f"no FD grid setup for payoff type {type(payoff).__name__}")
