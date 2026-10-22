"""Settings shared by every pipeline step, so training and backtesting can't
disagree about where the hold-out period starts."""
from __future__ import annotations

import pandas as pd

TICKER = "TSLA"
HORIZON = 8              # bars ahead that a label describes; also the grid's lifetime
TEST_FRACTION = 0.30     # last 30% of bars are never used for training
INITIAL_CAPITAL = 10_000.0


def split_point(index: pd.Index, test_fraction: float = TEST_FRACTION) -> int:
    """Position of the first hold-out bar."""
    return int(len(index) * (1 - test_fraction))


def training_index(index: pd.Index, horizon: int = HORIZON,
                   test_fraction: float = TEST_FRACTION) -> pd.Index:
    """Bars the model may learn from.

    A label at bar t describes bars t+1..t+H, so the last H training labels
    would describe prices inside the hold-out period. Those rows are purged.
    """
    return index[: max(split_point(index, test_fraction) - horizon, 0)]


def test_index(index: pd.Index, test_fraction: float = TEST_FRACTION) -> pd.Index:
    return index[split_point(index, test_fraction):]
