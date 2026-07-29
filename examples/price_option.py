from optpricing.engines import BlackScholesEngine, MonteCarloEngine
from optpricing.greeks import FiniteDifferenceGreeks
from optpricing.instruments import Option
from optpricing.market import MarketData
from optpricing.payoffs import AsianCall, Put
from optpricing.processes import GBM

market = MarketData(spot=105.0, rate=0.04, dividend_yield=0.015)
process = GBM(vol=0.22)

# A vanilla European put — closed form and Monte Carlo should agree.
put = Option(payoff=Put(strike=100.0), expiry=0.75)

bs_result = BlackScholesEngine().price(put, process, market)
mc_result = MonteCarloEngine(n_paths=200_000, n_steps=1, seed=42).price(put, process, market)

print(f"BS price:  {bs_result.price:.4f}")
print(f"MC price:  {mc_result.price:.4f}  (stderr {mc_result.std_error:.4f})")

greeks = FiniteDifferenceGreeks(BlackScholesEngine()).compute(put, process, market)
print(f"Greeks:    delta={greeks.delta:.4f} gamma={greeks.gamma:.4f} "
      f"vega={greeks.vega:.4f} theta={greeks.theta:.4f} rho={greeks.rho:.4f}")

# Swap the payoff for something BS can't touch — MonteCarloEngine doesn't
# care, since it only needs the process to simulate and the payoff to
# evaluate a path.
asian_call = Option(payoff=AsianCall(strike=100.0), expiry=0.75)
asian_result = MonteCarloEngine(n_paths=100_000, n_steps=180, seed=42).price(asian_call, process, market)
print(f"Asian call MC price: {asian_result.price:.4f}  (stderr {asian_result.std_error:.4f})")
