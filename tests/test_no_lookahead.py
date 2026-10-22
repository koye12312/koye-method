"""Every input the strategy acts on at bar t must depend only on bars <= t.

The test pattern: change every price AFTER bar k, rerun, and check nothing at
or before bar k changed. If anything did, it was using the future.
"""
import numpy as np
import pandas as pd
import pytest

from core.config import HORIZON, split_point, training_index
from core.features import build_features, trailing_range
from core.grid_strategy import GridConfig, run_grid_backtest
from core.regime_labeller import LabelConfig, label_regimes
from random_walk_check import random_walk_ohlcv

K = 600


@pytest.fixture(scope="module")
def prices():
    return random_walk_ohlcv(n=1000, seed=3)


def rewrite_future(df, k=K):
    """Same data up to bar k, very different data afterwards."""
    out = df.copy()
    scale = np.linspace(0.6, 1.8, len(df) - k - 1)
    for col in ["Open", "High", "Low", "Close"]:
        out.iloc[k + 1:, out.columns.get_loc(col)] *= scale
    return out


def test_features_are_causal(prices):
    a = build_features(prices)
    b = build_features(rewrite_future(prices))
    pd.testing.assert_frame_equal(a.iloc[: K + 1], b.iloc[: K + 1])


def test_grid_levels_are_causal(prices):
    a = trailing_range(prices["Close"], HORIZON)
    b = trailing_range(rewrite_future(prices)["Close"], HORIZON)
    pd.testing.assert_frame_equal(a.iloc[: K + 1], b.iloc[: K + 1])


def test_labeller_range_is_from_the_future(prices):
    """The labeller's range describes bars t+1..t+H. This is why it may be a
    training target but must never be used to place trades."""
    a = label_regimes(prices["Close"], LabelConfig(horizon=HORIZON))
    b = label_regimes(rewrite_future(prices)["Close"], LabelConfig(horizon=HORIZON))
    # Labels well before k are untouched...
    pd.testing.assert_frame_equal(a.iloc[: K - HORIZON + 1], b.iloc[: K - HORIZON + 1])
    # ...but the label at bar k already changes, because it looks ahead.
    assert a["range_centre"].iat[K] != b["range_centre"].iat[K]


def test_backtest_equity_is_causal(prices):
    def run(df):
        rng = trailing_range(df["Close"], HORIZON)
        proba = pd.Series(1.0, index=df.index)
        return run_grid_backtest(
            df, proba,
            range_centre=rng["range_centre"].fillna(df["Close"]),
            range_width_pct=rng["range_width_pct"].fillna(0.02),
            cfg=GridConfig(confidence_threshold=0.0, horizon=HORIZON),
        )

    a, b = run(prices), run(rewrite_future(prices))
    pd.testing.assert_series_equal(a["equity"].iloc[: K + 1], b["equity"].iloc[: K + 1])


def test_training_rows_never_touch_the_hold_out():
    idx = pd.date_range("2024-01-01", periods=1000, freq="h")
    train = training_index(idx)
    last_train_pos = idx.get_loc(train[-1])
    # The label at the last training bar looks HORIZON bars ahead; all of
    # those bars must still be before the hold-out starts.
    assert last_train_pos + HORIZON < split_point(idx)


def test_random_walk_shows_no_edge():
    """Regression test for the original leak: with the future range, the
    always-on grid made +28% to +42% on every random walk tried."""
    from random_walk_check import run_once
    result = run_once(seed=0)
    assert abs(result["always_on"]) < 0.15
    assert abs(result["koye"]) < 0.15
