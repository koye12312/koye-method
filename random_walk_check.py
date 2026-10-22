"""Sanity check: run the whole pipeline on random-walk prices.

A random walk has no pattern to exploit, so an honest backtest should show
no consistent edge: some seeds win, some lose, and the average is roughly
zero minus trading costs. If a strategy reliably makes money here, the
backtest is leaking future information.

This needs no internet connection:  python random_walk_check.py
"""
import contextlib
import io

import numpy as np
import pandas as pd

from core.config import HORIZON, test_index, training_index
from core.features import FEATURE_NAMES, build_features, trailing_range
from core.grid_strategy import GridConfig, run_grid_backtest
from core.regime_classifier import cross_validate
from core.regime_labeller import LabelConfig, label_regimes


def random_walk_ohlcv(n: int = 3500, seed: int = 0) -> pd.DataFrame:
    """Hourly OHLCV bars from a geometric random walk (no drift, no memory)."""
    rng = np.random.default_rng(seed)
    close = 200 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="h")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)


def run_once(seed: int) -> dict:
    df = random_walk_ohlcv(seed=seed)
    labels = label_regimes(df["Close"], LabelConfig(horizon=HORIZON))
    feats = build_features(df)

    train = feats.loc[training_index(df.index)].join(labels[["is_range_bound"]]).dropna()
    with contextlib.redirect_stdout(io.StringIO()):
        result = cross_validate(train[FEATURE_NAMES], train["is_range_bound"].astype(int),
                                gap=HORIZON, n_estimators=100)

    test = test_index(df.index)
    ft = feats.loc[test].dropna()
    proba = pd.Series(0.0, index=test)
    proba.loc[ft.index] = result.final_model.predict_proba(ft[FEATURE_NAMES])[:, 1]

    rng = trailing_range(df["Close"], HORIZON).loc[test]
    grid_inputs = dict(range_centre=rng["range_centre"].fillna(df.loc[test, "Close"]),
                       range_width_pct=rng["range_width_pct"].fillna(0.02))
    always = run_grid_backtest(df.loc[test], pd.Series(1.0, index=test),
                               cfg=GridConfig(confidence_threshold=0.0, horizon=HORIZON), **grid_inputs)
    koye = run_grid_backtest(df.loc[test], proba,
                             cfg=GridConfig(confidence_threshold=0.55, horizon=HORIZON), **grid_inputs)
    close = df.loc[test, "Close"]
    return {
        "seed": seed,
        "buy_hold": close.iloc[-1] / close.iloc[0] - 1,
        "always_on": always["equity"].iloc[-1] / always["equity"].iloc[0] - 1,
        "koye": koye["equity"].iloc[-1] / koye["equity"].iloc[0] - 1,
        "cv_f1": result.summary()["f1"].mean(),
        "positive_rate": train["is_range_bound"].mean(),
    }


if __name__ == "__main__":
    rows = pd.DataFrame([run_once(s) for s in range(10)]).set_index("seed")
    print(rows.to_string(float_format=lambda v: f"{v:7.1%}"))
    print("\nMean over 10 random walks:")
    print(rows.mean().to_string(float_format=lambda v: f"{v:7.1%}"))
