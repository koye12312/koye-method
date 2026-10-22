"""Step 3: train the Random Forest regime classifier with purged TimeSeriesSplit CV.

Only the training period is used. The last TEST_FRACTION of bars is held out
for 04_run_backtest.py and is never seen here, not even by the final model.
"""
import pandas as pd

from core.config import HORIZON, TEST_FRACTION, TICKER, test_index, training_index
from core.data_fetcher import DATA_DIR, load_hourly
from core.features import FEATURE_NAMES, build_features
from core.regime_classifier import (
    cross_validate,
    plot_confusion_matrix,
    plot_feature_importance,
    save_model,
)

if __name__ == "__main__":
    df = load_hourly(TICKER)
    labels = pd.read_csv(DATA_DIR / f"{TICKER}_labels.csv", parse_dates=["Datetime"], index_col="Datetime")

    print("Building features...")
    feats = build_features(df)
    print(f"  {len(feats.columns)} features: {list(feats.columns)}")

    train_idx = training_index(df.index)
    hold_out = test_index(df.index)
    print(f"\nTraining period: {train_idx.min()} -> {train_idx.max()}")
    print(f"Hold-out (never used here): {hold_out.min()} -> {hold_out.max()} "
          f"({100 * TEST_FRACTION:.0f}%, plus {HORIZON} purged bars before it)")

    # Align: drop rows with any NaN feature OR a missing label.
    full = feats.loc[train_idx].join(labels[["is_range_bound"]]).dropna()
    X = full[FEATURE_NAMES]
    y = full["is_range_bound"].astype(int)

    print(f"\nTraining set: {len(X):,} rows, "
          f"{(y == 1).sum():,} positives ({100 * (y == 1).mean():.1f}%)")

    print(f"\nRunning purged TimeSeriesSplit cross-validation (5 folds, gap={HORIZON})...")
    result = cross_validate(X, y, n_splits=5, gap=HORIZON, n_estimators=200, max_depth=8)

    print("\n=== CV summary ===")
    print(result.summary().to_string(index=False, float_format="%.3f"))
    print(f"\nMean precision: {result.summary()['precision'].mean():.3f}")
    print(f"Mean recall:    {result.summary()['recall'].mean():.3f}")
    print(f"Mean F1:        {result.summary()['f1'].mean():.3f}")
    print(f"Positive rate:  {(y == 1).mean():.3f}  (an F1 near this means no real skill)")

    plot_confusion_matrix(result)
    plot_feature_importance(result)
    save_model(result.final_model)

    print("\nNext step: python 04_run_backtest.py")
