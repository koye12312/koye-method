"""Step 4: backtest the Koye Method against buy-and-hold and an always-on grid baseline.

Three strategies are compared:
  1. Buy & Hold      - always invested
  2. Always-on grid  - deploy a grid every H bars regardless of regime (no ML)
  3. Koye Method     - ML-gated grid (only deploy when classifier predicts range-bound)

The always-on baseline is the critical comparison: if the Koye Method does
not beat it, the ML gating is not adding value.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_fetcher import DATA_DIR, load_hourly
from core.features import FEATURE_NAMES, build_features
from core.grid_strategy import GridConfig, run_grid_backtest
from core.performance import (
    plot_equity_comparison,
    print_comparison,
    summarize,
)
from core.regime_classifier import load_model

TICKER = "TSLA"
INITIAL_CAPITAL = 10_000.0
TEST_FRACTION = 0.30   # last 30% of the data is held out for the backtest

if __name__ == "__main__":
    df = load_hourly(TICKER)
    labels = pd.read_csv(DATA_DIR / f"{TICKER}_labels.csv", parse_dates=["Datetime"], index_col="Datetime")
    feats = build_features(df)

    # Hold-out split: train was on the first 70%, test is on the last 30%.
    n = len(df)
    split = int(n * (1 - TEST_FRACTION))
    test_idx = df.index[split:]
    print(f"Test period: {test_idx.min()} -> {test_idx.max()} "
          f"({len(test_idx):,} bars, {100 * TEST_FRACTION:.0f}% hold-out)")

    df_test = df.loc[test_idx]
    feats_test = feats.loc[test_idx]
    labels_test = labels.loc[test_idx]

    # ---- Get classifier probabilities on the test set ----
    model = load_model()
    feats_clean = feats_test.dropna()
    proba = pd.Series(np.nan, index=feats_test.index)
    proba.loc[feats_clean.index] = model.predict_proba(feats_clean[FEATURE_NAMES])[:, 1]

    range_centre = labels_test["range_centre"]
    range_width_pct = labels_test["range_width_pct"]

    # ---- Strategy 1: Buy & Hold ----
    print("\nRunning Buy & Hold...")
    bh_returns = df_test["Close"].pct_change().fillna(0.0)
    bh_equity = (1 + bh_returns).cumprod() * INITIAL_CAPITAL
    bh_equity.name = "BuyHold"

    # ---- Strategy 2: Always-on grid (no ML gate) ----
    print("Running Always-on Grid (no ML gate)...")
    cfg_always = GridConfig(confidence_threshold=0.0)   # always deploy
    log_always = run_grid_backtest(
        df_test,
        proba_range_bound=pd.Series(1.0, index=df_test.index),
        range_centre=range_centre.fillna(df_test["Close"]),
        range_width_pct=range_width_pct.fillna(0.02),
        initial_capital=INITIAL_CAPITAL,
        cfg=cfg_always,
    )

    # ---- Strategy 3: Koye Method (ML-gated grid) ----
    print("Running Koye Method (ML-gated grid)...")
    cfg_koye = GridConfig(confidence_threshold=0.55)
    log_koye = run_grid_backtest(
        df_test,
        proba_range_bound=proba.fillna(0.0),
        range_centre=range_centre.fillna(df_test["Close"]),
        range_width_pct=range_width_pct.fillna(0.02),
        initial_capital=INITIAL_CAPITAL,
        cfg=cfg_koye,
    )

    # ---- Compare ----
    stats = {
        "Buy & Hold":     summarize(bh_equity),
        "Always-on Grid": summarize(log_always["equity"].dropna()),
        "Koye Method":    summarize(log_koye["equity"].dropna()),
    }
    print(f"\n=== Backtest results on {TICKER} hourly bars (last {100 * TEST_FRACTION:.0f}% of data) ===\n")
    print_comparison(stats)

    print(f"\nAlways-on Grid: {log_always.attrs['total_round_trips']} round-trips, "
          f"{log_always.attrs['total_breakouts']} breakouts")
    print(f"Koye Method:    {log_koye.attrs['total_round_trips']} round-trips, "
          f"{log_koye.attrs['total_breakouts']} breakouts")
    in_grid_pct = 100 * log_koye["in_grid"].mean()
    print(f"Koye Method was deployed {in_grid_pct:.1f}% of the test period")

    plot_equity_comparison(
        {
            "Buy & Hold": bh_equity,
            "Always-on Grid": log_always["equity"].dropna(),
            "Koye Method": log_koye["equity"].dropna(),
        },
        title=f"{TICKER} hourly: Koye Method vs baselines",
    )
