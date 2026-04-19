"""Label historical windows as range-bound or trending using K-means clustering.

Method
------
For each rolling window of the *future* H bars (the prediction horizon),
we run K-means with k=2 on the close prices in the window. We then count
how many times the cluster assignment switches as we walk forward in time
through the window:

  - Many switches  -> prices oscillate between two levels  -> RANGE-BOUND
  - Few switches   -> prices monotonically move from one cluster to the
                       other (early bars in cluster A, later in cluster B)
                       -> TRENDING

The "switch rate" = number_of_switches / (window_length - 1) is then
thresholded to produce a binary label.

We also compute the *centre* and *width* of the predicted range, which
the grid strategy uses later as its trade levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class LabelConfig:
    horizon: int = 8                 # how many bars ahead to label
    switch_threshold: float = 0.20   # switch rate above this -> range-bound
    min_width_pct: float = 0.005     # ignore degenerate ranges narrower than 0.5%


def _switch_rate_and_range(prices: np.ndarray) -> Tuple[float, float, float]:
    """Return (switch_rate, centre, width_pct) for a price window."""
    if len(prices) < 4 or np.std(prices) == 0:
        return 0.0, float(np.mean(prices)), 0.0

    km = KMeans(n_clusters=2, n_init=5, random_state=0)
    labels = km.fit_predict(prices.reshape(-1, 1))
    switches = int(np.sum(labels[1:] != labels[:-1]))
    switch_rate = switches / (len(prices) - 1)

    centre = float(np.mean(km.cluster_centers_))
    width = float(prices.max() - prices.min())
    width_pct = width / centre if centre else 0.0
    return switch_rate, centre, width_pct


def label_regimes(close: pd.Series, cfg: LabelConfig = LabelConfig()) -> pd.DataFrame:
    """Return a frame indexed like ``close`` with regime labels and range info.

    Columns:
      - is_range_bound : 1 if the *future* H bars are range-bound, else 0
      - switch_rate    : raw switch rate over the future window
      - range_centre   : centre of the predicted range
      - range_width_pct: width of the predicted range as fraction of centre

    Rows where the future window can't be computed (last H bars) are NaN.
    """
    H = cfg.horizon
    out = pd.DataFrame(
        index=close.index,
        columns=["is_range_bound", "switch_rate", "range_centre", "range_width_pct"],
        dtype=float,
    )

    prices = close.values
    n = len(prices)
    for i in range(n - H):
        window = prices[i + 1 : i + 1 + H]   # strictly future bars
        sr, centre, width_pct = _switch_rate_and_range(window)
        is_range = (sr >= cfg.switch_threshold) and (width_pct >= cfg.min_width_pct)
        out.iat[i, 0] = float(is_range)
        out.iat[i, 1] = sr
        out.iat[i, 2] = centre
        out.iat[i, 3] = width_pct

    return out
