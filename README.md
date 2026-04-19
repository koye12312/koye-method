# The Method

An ML-gated mean-reversion grid trader for high-volatility equities.

The system has two stages: an **unsupervised regime labeller** uses K-means
clustering to classify historical hourly windows of TSLA as range-bound or
trending, and a **Random Forest classifier** learns to predict the regime of
the *next* eight hours from causal features (volatility, autocorrelation,
ADX, Hurst exponent, …). When the classifier is confident the next window
will be range-bound, the system deploys a grid of mean-reversion trades
inside the predicted range, with a hard stop on regime breakdown.

The classifier is evaluated with `TimeSeriesSplit` cross-validation to
prevent any look-ahead bias. The strategy is benchmarked against both
buy-and-hold *and* an always-on grid baseline — the always-on baseline is
the critical comparison, because if the Koye Method does not beat it then
the ML gating is not adding value.

---

## Why this project exists

Most introductory quant projects test a moving-average crossover and call
it a day. That kind of project answers a question — "would this rule have
worked?" — to which the answer is almost always "no, it underperforms
buy-and-hold." This project asks a more interesting question:

> *Can a classifier, trained on engineered features that describe recent
> price behaviour, predict short-horizon market regimes well enough to
> profitably gate a mean-reversion strategy?*

The honest answer might still be no. That is fine. The point is to build
a system that can *answer the question rigorously*: with proper time-series
cross-validation, with class-imbalance-aware evaluation metrics, with
explicit baselines that isolate the contribution of the ML component, and
with a backtester that does not cheat.

---

## Project structure

```
koye-method/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── 01_download_data.py        # entry: fetch TSLA hourly bars
├── 02_label_regimes.py        # entry: cluster + label every window
├── 03_train_classifier.py     # entry: train RF with TimeSeriesSplit
├── 04_run_backtest.py         # entry: backtest all three strategies
│
├── core/
│   ├── __init__.py
│   ├── data_fetcher.py        # yfinance hourly downloader + loader
│   ├── features.py            # 9 causal features (vol, ADX, Hurst, …)
│   ├── regime_labeller.py     # K-means switch-rate labelling
│   ├── regime_classifier.py   # Random Forest + TimeSeriesSplit + plots
│   ├── grid_strategy.py       # grid execution with stops
│   └── performance.py         # metrics + comparison plots
│
├── data/                      # (generated, gitignored) CSVs
├── models/                    # (generated, gitignored) pickled RF
├── reports/                   # (generated, kept) all PNG outputs
└── notebooks/                 # space for ad-hoc analysis
```

---

## Installation

Requires Python 3.10+.

```bash
git clone <repo-url>
cd koye-method

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running the pipeline

The four numbered scripts are the pipeline. Run them in order; each one
reads the artefacts produced by the previous step.

```bash
python 01_download_data.py        # ~10s, downloads ~2 years of TSLA hourly OHLCV
python 02_label_regimes.py        # ~30s, clusters every window and saves labels + timeline plot
python 03_train_classifier.py     # ~15s, runs CV and saves model + diagnostic plots
python 04_run_backtest.py         # ~60s, backtests all three strategies and saves comparison plot
```

After all four run, `reports/` will contain four publication-quality plots:

- `regime_timeline.png` — TSLA price with range-bound regions shaded
- `confusion_matrix.png` — classifier performance summed across CV folds
- `feature_importance.png` — which features the Random Forest actually used
- `backtest_comparison.png` — equity curves for all three strategies

---

## Methodology

### Stage 1 — Regime labelling (unsupervised, `core/regime_labeller.py`)

For every hour `t`, look at the *next* `H = 8` close prices. Run K-means
with `k = 2` on those eight prices and walk forward in time, counting how
many times the cluster assignment switches:

- **Many switches** → prices oscillate between two levels → **range-bound**
- **Few switches** → prices monotonically move from one cluster to the
  other (early bars in cluster A, later bars in cluster B) → **trending**

The switch rate is `switches / (H - 1)`. A window is labelled range-bound
if its switch rate exceeds 0.20 *and* its width is at least 0.5% of price
(to filter out degenerate near-flat windows). The labeller also outputs
the centre and width of the predicted range, which the grid strategy
uses later as its trade levels.

This is a deliberate, principled use of K-means: clustering is being used
to discover a structural property of price evolution (oscillation vs
monotonicity), not as a black box.

### Stage 2 — Causal features (`core/features.py`)

Nine features, each computed only from data up to and including time `t`:

| Feature | Window | What it measures |
|---|---|---|
| `vol_24h` | 24 bars | Std of log returns |
| `atr_14` | 14 bars | Average True Range (Wilder) |
| `bb_width_20` | 20 bars | Bollinger band width / mean |
| `adx_14` | 14 bars | Average Directional Index — trend strength |
| `dist_mean_48` | 48 bars | Z-score of price vs rolling mean |
| `autocorr_24` | 24 bars | Lag-1 autocorrelation of returns |
| `skew_48` | 48 bars | Return distribution skew |
| `kurt_48` | 48 bars | Return distribution kurtosis |
| `hurst_100` | 100 bars | Hurst exponent via R/S analysis |

The Hurst exponent deserves a note: `H < 0.5` indicates mean-reversion,
`H > 0.5` indicates trending. It is the most theoretically grounded
range-vs-trend feature in the set.

### Stage 3 — Classifier (`core/regime_classifier.py`)

`RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced")`.

- **`class_weight="balanced"`** handles the natural imbalance — range-bound
  windows are typically rarer than trending ones, and naive accuracy-based
  training would just predict "trending" everywhere.
- **`TimeSeriesSplit(n_splits=5)`** is used in place of standard k-fold.
  This is non-negotiable for time-series data: every train fold must be
  strictly in the past relative to its test fold. Standard `KFold` with
  `shuffle=True` would let the model learn from the future to predict
  the past, which is the same look-ahead bias that invalidates most
  amateur backtests.
- **Reported metrics are precision, recall, and F1 on the minority class.**
  Accuracy is deliberately not the headline number — see your data
  analysis module's class imbalance lecture for why.

### Stage 4 — Grid execution (`core/grid_strategy.py`)

When the classifier predicts P(range-bound) ≥ 0.55 for the next window,
the strategy:

1. Computes the predicted range centre and width from the labeller.
2. Places `N = 5` evenly-spaced limit orders between
   `centre − width/2` and `centre + width/2`.
3. Sets a hard stop at `centre ± 1.5 × (width / 2)`. If the price breaks
   this band the entire grid is liquidated immediately.
4. Sets a time stop at `t + H` bars. If neither the take-profit nor the
   hard stop triggers, the grid is closed cleanly at the close of bar
   `t + H`.
5. Each round-trip captures the spacing between adjacent grid levels as
   profit (less transaction costs).

The hard stop is the single most important risk-control element in the
strategy. Without it, mean-reversion strategies blow up spectacularly
when a regime breaks — see LTCM, 1998.

### Stage 5 — Backtest (`04_run_backtest.py`)

Three strategies are compared on the held-out final 30% of the data:

1. **Buy & Hold** — always invested, the standard benchmark.
2. **Always-on Grid** — deploy the grid every `H` bars regardless of
   regime prediction. This isolates the contribution of the grid logic
   from the contribution of the ML gating.
3. **Koye Method** — ML-gated grid, deployed only when the classifier
   predicts a range-bound regime.

Performance is reported in CAGR, Sharpe, Sortino, and Max Drawdown,
all annualised assuming 252 × 6.5 = 1,638 hourly periods per year.

---

## What honest results look like

Performance numbers depend heavily on the time period and on real-world
TSLA data. As a rough guide for what to expect when running on real data:

- **Random-baseline classifier metrics:** if your CV F1 is roughly equal
  to the positive-class prevalence, the classifier has not learned
  anything useful and the Koye Method should not be expected to beat the
  always-on grid.
- **A classifier with real signal:** F1 in the 0.45–0.65 range on the
  minority class with mean precision noticeably above the prevalence rate
  is a meaningful result.
- **Strategy realism check:** if the backtest reports a Sharpe above ~3.0
  on real TSLA data, *something is wrong*. Recheck the look-ahead
  guarantees in the labeller and the feature builder before celebrating.

The plots in `reports/` shipped with this repo were produced from a
synthetic test dataset (Yahoo Finance was unreachable from the build
environment). Numbers from your own run on real TSLA data will differ.

---

## Key design decisions and their justifications

**Why K-means for labelling rather than a simple range-vs-volatility ratio?**
A ratio captures dispersion but not structure. K-means with the switch-rate
heuristic explicitly distinguishes between "prices oscillate around a
centre" and "prices walk from low to high" — two windows can have the same
volatility but very different structure.

**Why Random Forest rather than a deep network?**
Bias-variance trade-off (Week 7 of any serious ML curriculum). With ~3,000
training samples and 9 features, a deep network has too much capacity and
will overfit. A Random Forest with `max_depth=8` is the natural sweet spot:
non-linear interactions are captured, but variance is controlled by
bagging across 200 trees.

**Why hourly bars rather than daily or minute bars?**
Daily bars give too little data (~500 bars in 2 years) for a 9-feature
classifier. Minute bars introduce microstructure noise that the features
in this project are not designed to handle. Hourly is the natural middle
ground for a swing-trading horizon.

**Why a hard stop at 1.5× the half-width?**
Wide enough that ordinary intra-range volatility doesn't trigger it,
narrow enough that a real breakout doesn't grow into a catastrophic loss.
1.5× is a reasonable starting point that should be tuned per asset.

---

## What this project does not do

- **No real-money trading.** Hourly bar data has gaps, no slippage is
  modelled beyond a flat 5 bps per fill, and order-book dynamics are
  ignored entirely. This is a research backtester, not an execution system.
- **No multi-asset portfolio.** The system trades a single ticker. Pairs
  trading or basket mean-reversion would be a natural extension.
- **No alternative-data features.** All features are derived from price
  and volume. Order-flow imbalance, options-implied volatility surfaces,
  or news-sentiment features would likely improve classifier performance.
- **No hyperparameter search.** The grid configuration (`N = 5` levels,
  confidence threshold of 0.55, stop multiplier of 1.5×) was set by
  reasoning, not by walk-forward optimisation. Adding a proper
  hyperparameter sweep with held-out validation is a sensible next step.

---

## Acknowledgements

The methodology draws explicitly on material from the *Data Analysis &
Visualisation* module: data preprocessing and feature engineering (Week 2),
classification and cross-validation (Weeks 3–5), bias/variance and
ensemble methods (Week 7), and clustering (Week 9). The use of
`TimeSeriesSplit` instead of `KFold` is a deliberate adaptation to
time-series data not covered in the standard curriculum.

The look-ahead-safe accounting in the backtester follows the same
convention as the predecessor [quant-backtester](https://github.com/koye12312/quant-backtester)
project: positions are lagged by one bar, every metric is computed from
strategy returns rather than from raw signals.

## License

MIT — see [LICENSE](LICENSE).
