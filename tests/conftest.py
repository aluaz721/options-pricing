import pytest

from optpricing.market import MarketData
from optpricing.processes import GBM

# Shared across every test module: one canonical market/process pair so a
# reader comparing two test files doesn't have to check whether "market"
# means the same thing in both. Individual tests still override anything
# they need (e.g. a specific spot, a barrier level) explicitly rather than
# relying on a second, differently-parameterized fixture.


@pytest.fixture
def market():
    return MarketData(spot=100.0, rate=0.05, dividend_yield=0.02)


@pytest.fixture
def process():
    return GBM(vol=0.2)
