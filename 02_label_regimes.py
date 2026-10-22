"""Step 2: label every historical hour as range-bound or trending using K-means."""
from pathlib import Path

import pandas as pd

from core.config import HORIZON, TICKER
from core.data_fetcher import DATA_DIR, load_hourly
from core.performance import plot_regime_timeline
from core.regime_labeller import LabelConfig, label_regimes

if __name__ == "__main__":
    print(f"Loading {TICKER} hourly bars...")
    df = load_hourly(TICKER)
    print(f"  Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}\n")

    cfg = LabelConfig(horizon=HORIZON, switch_threshold=0.20, min_width_pct=0.005)
    print(f"Labelling regimes with config: {cfg}")
    labels = label_regimes(df["Close"], cfg=cfg)

    n_total = labels["is_range_bound"].notna().sum()
    n_pos = (labels["is_range_bound"] == 1).sum()
    pct = 100 * n_pos / n_total if n_total else 0.0
    print(f"\n  Range-bound bars: {n_pos:,} / {n_total:,} ({pct:.1f}%)")

    out_path = DATA_DIR / f"{TICKER}_labels.csv"
    labels.to_csv(out_path)
    print(f"\nSaved labels -> {out_path}")

    plot_regime_timeline(df["Close"], labels["is_range_bound"])
    print("\nNext step: python 03_train_classifier.py")
