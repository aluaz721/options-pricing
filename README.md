# optpricing

A Python library for exploring how different numerical methods price options under different
stochastic models — and a place to see the connecting mathematics (SDEs, PDEs, Monte Carlo,
Malliavin calculus) implemented side by side rather than in isolation.

The organizing idea: a price is really a function of four independent choices —

1. **Payoff** — what the contract pays (call, put, digital, up-and-out, Asian, ...)
2. **Exercise style** — when it can be exercised (European, American)
3. **Dynamics** — what the underlying is assumed to do (GBM, Heston, Merton jump-diffusion, ...)
4. **Method** — how the resulting price is computed (closed form, tree, PDE, Monte Carlo, ...)

`optpricing` keeps these four as separate, composable objects (`Payoff`, `Option`,
`StochasticProcess`, `PricingEngine`) rather than one pricing function per model, so the same
option can be priced multiple ways and the ways cross-validate each other. That cross-validation
is used throughout the test suite: independent methods (a lattice vs. a PDE vs. simulation) are
checked against each other, not just against a single "known-good" reference.

Live demo (Streamlit): **https://options-pricing-aluaz721.streamlit.app/**

```python
from optpricing.engines import BlackScholesEngine, MonteCarloEngine
from optpricing.instruments import Option
from optpricing.market import MarketData
from optpricing.payoffs import Call
from optpricing.processes import GBM

market = MarketData(spot=100.0, rate=0.04, dividend_yield=0.01)
process = GBM(vol=0.2)
option = Option(payoff=Call(strike=100.0), expiry=1.0)

BlackScholesEngine().price(option, process, market).price   # closed form
MonteCarloEngine(n_paths=100_000).price(option, process, market).price  # simulation
```

---

## The mathematics

### Stochastic processes

Every process implements `simulate()` (draw paths under the risk-neutral measure) and, where one
exists in closed form, `characteristic_function()`.

**Geometric Brownian motion — `GBM`.** The Black-Scholes-Merton dynamics [[1]](#references),
[[2]](#references):

$$dS_t = (r - q)\,S_t\,dt + \sigma S_t\,dW_t$$

Simulated exactly (not discretized): $\ln S_t$ solves in closed form, so `GBM.simulate` draws
$\ln S_{t+\Delta t} = \ln S_t + (r-q-\tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z$ directly,
with zero bias regardless of step count — the step count only matters for path-dependent payoffs
that need intermediate resolution, not for terminal-distribution accuracy.

**Heston stochastic volatility — `Heston`.** Adds a mean-reverting variance process correlated
with the spot [[3]](#references):

$$dS_t = (r-q)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad d\langle W^S, W^v\rangle_t = \rho\,dt$$

$v_t$ is a CIR / square-root process [[4]](#references); naive Euler discretization can push it
negative whenever the Feller condition $2\kappa\theta > \xi^2$ is violated, which most calibrated
parameter sets do in practice. `Heston.simulate` instead uses Andersen's (2008) **Quadratic-Exponential
(QE) scheme** [[5]](#references): at each step, $v_{t+\Delta t}$ is drawn from a distribution
moment-matched to the CIR process's *known, exact* conditional mean and variance — a
squared-Gaussian $a(b+Z)^2$ when the distribution is comfortably away from zero, and a
mixture of a point mass at 0 with an exponential tail when it isn't — so it is non-negative
by construction rather than by clamping. The log-price step then uses Andersen's
correlation-preserving discretization, absorbing $\rho$ into deterministic coefficients
($K_0,\dots,K_4$) of $v_t$ and $v_{t+\Delta t}$ rather than correlating two noise terms directly.

**Merton jump-diffusion — `MertonJump`.** Adds a compound Poisson jump component [[6]](#references):

$$\frac{dS_t}{S_{t^-}} = (r - q - \lambda\bar\kappa)\,dt + \sigma\,dW_t + dJ_t$$

where $J_t$ jumps at rate $\lambda$ (per year) and each jump multiplies $S$ by $e^Y$,
$Y\sim\mathcal N(\mu_J,\sigma_J^2)$. $\bar\kappa = \mathbb E[e^Y-1] = e^{\mu_J+\frac12\sigma_J^2}-1$
is the jump's average relative contribution to the price; subtracting $\lambda\bar\kappa$ from the
drift (the *compensator*) keeps $\mathbb E[S_T]=S_0e^{(r-q)T}$ despite the jumps. Like GBM, this is
simulated exactly: over any interval, the jump count is exactly Poisson and the sum of that many
i.i.d. jump sizes is again exactly Normal, so there's no discretization scheme to get wrong here
the way Heston's variance process needs one.

### Pricing methods

**Closed form — `BlackScholesEngine`.** The Black-Scholes-Merton formula [[1]](#references),
[[2]](#references) for European calls/puts, plus the analogous formula for cash-or-nothing
digitals (both share the same $d_1,d_2$):

$$C = Se^{-qT}N(d_1) - Ke^{-rT}N(d_2), \qquad d_{1,2} = \frac{\ln(S/K) + (r-q\pm\frac12\sigma^2)T}{\sigma\sqrt T}$$

**Monte Carlo — `MonteCarloEngine`.** Simulate, discount, average:
$V = e^{-rT}\,\mathbb E[\text{payoff}(S_T)]$, estimated as a sample mean. Deliberately payoff- and
process-agnostic — it only calls `process.simulate()` and `option.payoff(paths)`, so it prices any
path-dependent payoff (Asian, barrier) under any process that can simulate itself, with no
per-combination code.

**Finite differences — `CrankNicolsonEngine`.** Solves the Black-Scholes PDE directly on a grid
[[7]](#references), [[8]](#references):

$$\frac{\partial V}{\partial t} + \tfrac12\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV = 0$$

using the second-order-accurate Crank-Nicolson scheme (the average of implicit and explicit Euler
in time-to-maturity). American exercise is handled via the **Brennan-Schwartz (1977)** trick
[[9]](#references): a single forward-elimination/backward-substitution sweep of the tridiagonal
system, clipping each value to the intrinsic payoff during back-substitution — no iterative
projected solve (PSOR) required. Barrier options are handled by capping the grid domain at the
barrier with a zero Dirichlet boundary there — the knock-out condition *is* the boundary
condition, not a separate approximation.

**Binomial trees — `BinomialTreeEngine`.** The Cox-Ross-Rubinstein lattice [[10]](#references):
$u=e^{\sigma\sqrt{\Delta t}}$, $d=1/u$, risk-neutral probability
$p = \frac{e^{(r-q)\Delta t}-d}{u-d}$, backward-induced with an elementwise $\max(\cdot,\text{intrinsic})$
at every node for American exercise. An independent cross-check on the PDE method above — same
free-boundary problem, entirely different numerical machinery.

**Least-Squares Monte Carlo — `LongstaffSchwartzEngine`.** Longstaff & Schwartz (2001)
[[11]](#references): simulated paths have no shared grid to run backward induction on directly,
so LSM instead works backward through *time steps*, and at each step estimates the continuation
value via a cross-sectional least-squares regression of realized (discounted) future cash flows
against a polynomial basis in the current spot — standing in for the conditional expectation a
lattice/PDE method gets for free from its grid structure. Deliberately process-agnostic (only
calls `process.simulate()`), so it prices American exercise under Heston or Merton jump-diffusion
with no engine-side changes — only the process needs to know how to simulate itself.

### Greeks

**Finite difference — `FiniteDifferenceGreeks`.** Bump-and-reprice with central differences;
wraps *any* `PricingEngine`, since it only perturbs `MarketData`/`Option` and reprices.

**Pathwise derivatives — `PathwiseGreeks`.** Differentiates the discounted payoff along each
simulated path instead of bumping and repricing [[12]](#references). For GBM, $S_T$ is an
explicit smooth function of $Z\sim\mathcal N(0,1)$, so e.g.

$$\Delta = e^{-rT}\,\mathbb E\!\left[f'(S_T)\frac{S_T}{S_0}\right]$$

Needs $f'$, which is why it's restricted to Call/Put: their kink is fine almost everywhere, but
a discontinuous payoff (a digital's jump) has no classical derivative, and Gamma would need to
differentiate the already-discontinuous $f'$ a second time — exactly the wall this method hits
and Malliavin weights exist to get around.

**Malliavin / likelihood-ratio weights — `MalliavinGreeks`.** Fournié, Lasry, Lebuchoux, Lions &
Touzi (1999) [[13]](#references); see also Broadie & Glasserman (1996) [[12]](#references) and
Glasserman's textbook treatment [[14]](#references). Rather than differentiating the payoff, this
differentiates the *Gaussian density* $Z$ is drawn from via an integration-by-parts identity
(Stein's identity, $\mathbb E[Zh(Z)]=\mathbb E[h'(Z)]$), which moves the derivative onto a *weight*
multiplying the untouched, raw payoff:

$$\frac{\partial}{\partial\theta}\mathbb E[f(S_T)] = \mathbb E\!\left[f(S_T)\cdot\left(\frac{\mu'(\theta)}{s}Z + \frac{s'(\theta)}{s}(Z^2-1)\right)\right]$$

for $S_T=S_0e^{\mu(\theta)+s(\theta)Z}$. Since $f$ itself is never differentiated, this works for
*any* payoff depending only on $S_T$ — continuous or not — which is what lets it price a
discontinuous digital's Delta with no special-casing, and, by applying the same identity a second
time, produce **Gamma** at all (unavailable via the pathwise method above, since that would
require differentiating a discontinuous indicator).

### Variance reduction

**Antithetic variates.** For a monotone payoff, $\text{payoff}(Z)$ and $\text{payoff}(-Z)$ are
negatively correlated, so averaging mirrored $\pm Z$ pairs gives an estimator with strictly lower
variance than the same number of independent draws, at the same simulation cost. Implemented as an
opt-in `antithetic=True` on `StochasticProcess.simulate` (the process owns the random draws, so it
has to be the one to mirror them): full support for GBM (measured ~25% standard-error reduction),
partial support for Merton (only the Brownian increment is mirrored — a Poisson jump count has no
natural antithetic partner — still ~24% reduction empirically), and explicitly unsupported for
Heston, since the QE scheme's variance step is a nonlinear function of its driving randomness and
naive mirroring has no proven variance-reduction guarantee there. Computing the *standard error*
correctly under antithetic pairing needs its own care — the two halves of each pair are correlated
by construction, so treating all paths as i.i.d. overstates the error; `_variance_reduction.py`
instead computes it over the pair *averages*.

---

## References

1. Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.
2. Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141–183.
3. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327–343.
4. Cox, J. C., Ingersoll, J. E. and Ross, S. A. (1985). "A Theory of the Term Structure of Interest Rates." *Econometrica*, 53(2), 385–407.
5. Andersen, L. (2008). "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance*, 11(3), 1–42.
6. Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous." *Journal of Financial Economics*, 3(1-2), 125–144.
7. Crank, J. and Nicolson, P. (1947). "A Practical Method for Numerical Evaluation of Solutions of Partial Differential Equations of the Heat-Conduction Type." *Mathematical Proceedings of the Cambridge Philosophical Society*, 43(1), 50–67.
8. Wilmott, P., Howison, S. and Dewynne, J. (1995). *The Mathematics of Financial Derivatives: A Student Introduction*. Cambridge University Press.
9. Brennan, M. J. and Schwartz, E. S. (1977). "The Valuation of American Put Options." *The Journal of Finance*, 32(2), 449–462.
10. Cox, J. C., Ross, S. A. and Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics*, 7(3), 229–263.
11. Longstaff, F. A. and Schwartz, E. S. (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach." *Review of Financial Studies*, 14(1), 113–147.
12. Broadie, M. and Glasserman, P. (1996). "Estimating Security Price Derivatives Using Simulation." *Management Science*, 42(2), 269–285.
13. Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L. and Touzi, N. (1999). "Applications of Malliavin Calculus to Monte Carlo Methods in Finance." *Finance and Stochastics*, 3(4), 391–412.
14. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer, Chapter 7 ("Estimating Sensitivities").

---

## Project structure

```
optpricing/           the library
  payoffs/              what the contract pays (Call, Put, Asian, barrier, digital)
  instruments/           exercise style + expiry (Option, European/American)
  processes/              underlying dynamics + simulation (GBM, Heston, MertonJump)
  engines/                 pricing methods (BlackScholes, MonteCarlo, CrankNicolson, Binomial, LongstaffSchwartz)
  greeks/                   sensitivity methods (FiniteDifference, Pathwise, Malliavin)
dashboard/             a Streamlit frontend, kept separate from the library — only imports
                       optpricing's public API, the way an external user would
tests/                 one file per engine/concern; conftest.py holds shared fixtures
examples/              minimal end-to-end usage script
```

Every `PricingEngine`/`GreeksEngine` declares what it supports via `supports(option, process)` and
raises `UnsupportedCombination` otherwise — not every method is meaningful for every combination
(there's no closed form for American exercise, no CRR tree for Heston), and the library is
explicit about that rather than silently producing a wrong number.

## What's implemented

Payoffs: vanilla call/put, Asian (arithmetic/geometric), up-and-out barrier, cash-or-nothing
digital. Exercise: European, American (Bermudan is modeled but not yet wired into any engine).
Dynamics: GBM, Heston, Merton jump-diffusion. Methods: closed form, Monte Carlo, Crank-Nicolson
finite differences, CRR binomial trees, Longstaff-Schwartz. Greeks: finite-difference,
pathwise, Malliavin. Variance reduction: antithetic variates.

Not yet implemented: multi-asset/basket options, non-equity asset classes (FX, commodities,
rates), local volatility / SABR / variance-gamma dynamics, characteristic-function (FFT/COS)
pricing, ADI methods, and other variance-reduction techniques (control variates, QMC).

## Development

```bash
pip install -e ".[dev]"
pytest
```

For the dashboard, see [dashboard/README.md](dashboard/README.md).
