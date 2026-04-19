"""Step 3: train the Random Forest regime classifier with TimeSeriesSplit CV."""
from pathlib import Path

import pandas as pd

from core.data_fetcher import DATA_DIR, load_hourly
from core.features import FEATURE_NAMES, build_features
from core.regime_classifier import (
    cross_validate,
    plot_confusion_matrix,
    plot_feature_importance,
    save_model,
)

TICKER = "TSLA"

if __name__ == "__main__":
    df = load_hourly(TICKER)
    labels = pd.read_csv(DATA_DIR / f"{TICKER}_labels.csv", parse_dates=["Datetime"], index_col="Datetime")

    print("Building features...")
    feats = build_features(df)
    print(f"  {len(feats.columns)} features: {list(feats.columns)}")

    # Align: drop rows with any NaN feature OR a missing label.
    full = feats.join(labels[["is_range_bound"]]).dropna()
    X = full[FEATURE_NAMES]
    y = full["is_range_bound"].astype(int)

    print(f"\nTraining set: {len(X):,} rows, "
          f"{(y == 1).sum():,} positives ({100 * (y == 1).mean():.1f}%)")

    print("\nRunning TimeSeriesSplit cross-validation (5 folds)...")
    result = cross_validate(X, y, n_splits=5, n_estimators=200, max_depth=8)

    print("\n=== CV summary ===")
    print(result.summary().to_string(index=False, float_format="%.3f"))
    print(f"\nMean precision: {result.summary()['precision'].mean():.3f}")
    print(f"Mean recall:    {result.summary()['recall'].mean():.3f}")
    print(f"Mean F1:        {result.summary()['f1'].mean():.3f}")

    plot_confusion_matrix(result)
    plot_feature_importance(result)
    save_model(result.final_model)

    print("\nNext step: python 04_run_backtest.py")
