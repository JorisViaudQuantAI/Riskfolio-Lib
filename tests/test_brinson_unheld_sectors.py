import numpy as np
import pandas as pd
import pytest

from riskfolio.src.RiskFunctions import BrinsonAttribution


@pytest.fixture
def inputs():
    prices = pd.DataFrame(
        [[100., 100., 100., 100.], [110., 95., 103., 108.]],
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
        columns=["A", "B", "C", "D"],
    )
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=prices.columns)
    wb = pd.Series([0.25] * 4, index=prices.columns)
    classes = pd.DataFrame(
        {"Assets": prices.columns, "Sector": ["Tech", "Tech", "Health", "Health"]}
    )
    return prices, w, wb, classes


def attribution(prices, w, wb, classes, **kwargs):
    return BrinsonAttribution(
        prices, w, wb, kwargs.pop("start", "2025-01-03"),
        kwargs.pop("end", "2025-01-06"), classes, "Sector", **kwargs
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_unheld_sector_reconciles_to_active_return(inputs, reverse):
    prices, _, wb, classes = inputs
    w = pd.Series([0.5, 0.5, 0., 0.], index=prices.columns)
    if reverse:
        w, wb = wb, w
    result, _ = attribution(prices, w, wb, classes)
    returns = prices.iloc[-1] / prices.iloc[0] - 1
    expected = float((w - wb) @ returns)
    assert result.loc["Total Excess Return", "Total"] == pytest.approx(expected)
    assert result.loc["Asset Allocation", "Total"] == pytest.approx(expected)
    assert result.loc["Security Selection", "Total"] == pytest.approx(0.0)
    assert result.loc["Interaction", "Total"] == pytest.approx(0.0)


def test_sector_absent_from_both_sides(inputs):
    prices, _, _, classes = inputs
    w = pd.Series([0.6, 0.4, 0., 0.], index=prices.columns)
    wb = pd.Series([0.5, 0.5, 0., 0.], index=prices.columns)
    result, _ = attribution(prices, w, wb, classes)
    np.testing.assert_allclose(result["Health"], 0.0)
    assert result.loc["Total Excess Return", "Total"] == pytest.approx(0.015)


def test_ordinary_sector_returns_are_unchanged(inputs):
    prices, w, wb, classes = inputs
    result, _ = attribution(prices, w, wb, classes)
    assert result.loc["Total Excess Return", "Total"] == pytest.approx(-0.001)
