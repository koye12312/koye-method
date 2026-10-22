"""Metrics against hand calculations, and grid mechanics on tiny scripted bars."""
import math

import numpy as np
import pandas as pd
import pytest

from core.grid_strategy import GridConfig, run_grid_backtest
from core.performance import bars_per_year, cagr, max_drawdown, sharpe, sortino
from core.regime_classifier import cross_validate


# ----------------------------- metrics ------------------------------------ #
def test_bars_per_year_is_measured_from_the_data():
    idx = pd.date_range("2024-01-01", periods=366 * 24 + 1, freq="h")  # 2024 is a leap year
    assert bars_per_year(idx) == pytest.approx(366 * 24 / (366 / 365.25))


def test_cagr_uses_calendar_time():
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])
    assert cagr(pd.Series([100.0, 200.0], index=idx)) == pytest.approx(math.sqrt(2) - 1, rel=1e-3)


def test_max_drawdown():
    assert max_drawdown(pd.Series([100, 120, 90, 130.0])) == pytest.approx(-0.25)


def test_sharpe_ignores_floating_point_noise():
    assert sharpe(pd.Series([0.001] * 20)) == 0.0


def test_sortino_uses_downside_deviation():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    downside_dev = math.sqrt((0.01**2 + 0.02**2) / 4)
    assert sortino(r, periods_per_year=100) == pytest.approx(10 * r.mean() / downside_dev)


def test_sortino_penalises_deeper_losses():
    shallow = pd.Series([0.02, -0.010, -0.011, 0.031] * 10)
    deep = pd.Series([0.04, -0.020, -0.021, 0.031] * 10)
    assert sortino(deep) < sortino(shallow)


# ----------------------------- cross-validation ---------------------------- #
def test_cross_validation_leaves_a_gap_between_train_and_test(monkeypatch):
    seen = []
    from core import regime_classifier as rc
    real_split = rc.TimeSeriesSplit.split

    def spy(self, X, y=None, groups=None):
        for tr, te in real_split(self, X, y, groups):
            seen.append((tr.max(), te.min()))
            yield tr, te

    monkeypatch.setattr(rc.TimeSeriesSplit, "split", spy)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = pd.Series(rng.integers(0, 2, 300))
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        cross_validate(X, y, n_splits=3, gap=8, n_estimators=5)
    assert seen and all(test_start - train_end > 8 for train_end, test_start in seen)


# ----------------------------- grid mechanics ------------------------------ #
def bars(rows):
    idx = pd.date_range("2024-01-02 10:00", periods=len(rows), freq="h")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1)


def run(df, centre=100.0, width_pct=0.04, horizon=4, cost=0.0, stop_mult=1.5):
    n = len(df)
    return run_grid_backtest(
        df,
        proba_range_bound=pd.Series([1.0] + [0.0] * (n - 1), index=df.index),
        range_centre=pd.Series(centre, index=df.index),
        range_width_pct=pd.Series(width_pct, index=df.index),
        cfg=GridConfig(n_levels=5, confidence_threshold=0.5, horizon=horizon, stop_mult=stop_mult),
        cost_per_trade=cost,
    )


def test_buy_low_sell_one_level_up_is_a_round_trip():
    # Grid 98, 99, 100, 101, 102 (spacing 1). Bar 1 dips to 98.5 -> buys at 99;
    # bar 2 rises to 100.2 without revisiting 99 -> sells that lot at 100.
    df = bars([[100, 100, 100, 100], [100, 100, 98.5, 99], [99.6, 100.2, 99.5, 100], [100, 100, 100, 100],
               [100, 100, 100, 100], [100, 100, 100, 100]])
    log = run(df)
    assert log.attrs["total_round_trips"] == 1
    assert log["equity"].iloc[-1] > 10_000


def test_hard_stop_liquidates_on_a_breakout():
    # Stop is 1.5 * half-width = 3 below centre (97). Bar 2 crashes through it.
    df = bars([[100, 100, 100, 100], [100, 100, 98.5, 99], [99, 99, 90, 91], [91, 91, 91, 91],
               [91, 91, 91, 91]])
    log = run(df)
    assert log.attrs["total_breakouts"] == 1
    assert log["position_value"].iloc[2] == 0
    assert log["equity"].iloc[-1] < 10_000


def test_trading_costs_reduce_returns():
    df = bars([[100, 100, 100, 100], [100, 100, 98.5, 99], [99, 100.2, 99, 100], [100, 100, 100, 100],
               [100, 100, 100, 100], [100, 100, 100, 100]])
    assert run(df, cost=0.001)["equity"].iloc[-1] < run(df, cost=0.0)["equity"].iloc[-1]
