from dataclasses import replace

from optpricing.engines.base import PricingEngine
from optpricing.greeks.base import Greeks, GreeksEngine
from optpricing.instruments.option import Option
from optpricing.market import MarketData
from optpricing.processes.base import StochasticProcess


class FiniteDifferenceGreeks(GreeksEngine):
    """Bump-and-reprice. Works with any PricingEngine, since it only touches
    MarketData/Option and never looks inside the engine or process.

    Vega is the exception: it needs a process-specific "volatility" parameter
    to bump, so it only works for processes exposing a `.vol` field (GBM,
    Merton's diffusive leg). Heston has no single vol to bump — its vol-of-vol,
    v0, and theta sensitivities are each their own Greek, left for later.
    """

    def __init__(self, engine: PricingEngine, bump: float = 1e-4):
        self.engine = engine
        self.bump = bump

    def compute(self, option: Option, process: StochasticProcess, market: MarketData) -> Greeks:
        base = self.engine.price(option, process, market).price

        # Central differences throughout: same cost as one-sided (two extra
        # prices either way) but cancels the O(h^2) error term, so delta/rho/
        # vega are O(h^2)-accurate instead of O(h). Gamma reuses `base` as its
        # midpoint rather than a third fresh price.
        h_spot = self.bump * market.spot  # relative bump, so it scales with the spot level
        price_up = self.engine.price(option, process, replace(market, spot=market.spot + h_spot)).price
        price_down = self.engine.price(option, process, replace(market, spot=market.spot - h_spot)).price
        delta = (price_up - price_down) / (2 * h_spot)
        gamma = (price_up - 2 * base + price_down) / h_spot**2

        h_rate = self.bump
        rho_up = self.engine.price(option, process, replace(market, rate=market.rate + h_rate)).price
        rho_down = self.engine.price(option, process, replace(market, rate=market.rate - h_rate)).price
        rho = (rho_up - rho_down) / (2 * h_rate)

        # One-sided here, not central: bumping expiry *up* would ask the
        # engine to price a longer-dated option than was requested, which is
        # a fine calculation but a confusing thing for this method to do
        # silently. Shortening expiry by h approximates -dV/dT, i.e. the
        # value lost as time passes and less time remains to expiry — the
        # conventional theta sign (typically negative for a long option).
        h_time = min(1e-3, option.expiry / 2)
        price_theta = self.engine.price(replace(option, expiry=option.expiry - h_time), process, market).price
        theta = (price_theta - base) / h_time

        # Only defined when the process exposes a single `.vol` to bump
        # (GBM, Merton's diffusive leg) — see class docstring for why Heston
        # can't do this the same way.
        vega = None
        if hasattr(process, "vol"):
            h_vol = self.bump
            price_vol_up = self.engine.price(option, replace(process, vol=process.vol + h_vol), market).price
            price_vol_down = self.engine.price(option, replace(process, vol=process.vol - h_vol), market).price
            vega = (price_vol_up - price_vol_down) / (2 * h_vol)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
