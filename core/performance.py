"""Performance metrics tailored for hourly equity curves."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Fallback only. Yahoo's hourly US bars give about 7 bars per trading day
# (09:30, 10:30 ... 15:30), so the real rate is estimated from the data.
HOURS_PER_YEAR = 252 * 7


def bars_per_year(index: pd.Index, fallback: float = HOURS_PER_YEAR) -> float:
    """How many bars the data actually has per calendar year."""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
        if years > 0:
            return (len(index) - 1) / years
    return fallback


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def cagr(equity: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    """Compound annual growth rate, using calendar time when the index has dates."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    if isinstance(equity.index, pd.DatetimeIndex):
        years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    else:
        years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    # Tolerance, not == 0: float noise makes a constant series' std ~1e-19.
    if returns.empty or returns.std() < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def sortino(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    """Annualised Sortino ratio using downside deviation: the root-mean-square
    of below-zero returns over *all* periods (gains count as zero)."""
    if returns.empty:
        return 0.0
    downside_dev = float(np.sqrt((np.minimum(returns, 0.0) ** 2).mean()))
    if downside_dev < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / downside_dev)


def summarize(equity: pd.Series) -> Dict[str, float]:
    rets = returns_from_equity(equity)
    ppy = bars_per_year(equity.index)
    return {
        "final_value": float(equity.iloc[-1]) if len(equity) else 0.0,
        "CAGR":        cagr(equity, ppy),
        "MaxDrawdown": max_drawdown(equity),
        "Sharpe":      sharpe(rets, ppy),
        "Sortino":     sortino(rets, ppy),
    }


def print_comparison(stats_by_name: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print a side-by-side table of strategies."""
    names = list(stats_by_name.keys())
    metrics = [
        ("Final value", "final_value", "${:>12,.2f}"),
        ("CAGR",        "CAGR",        "{:>13.2%}"),
        ("Max DD",      "MaxDrawdown", "{:>13.2%}"),
        ("Sharpe",      "Sharpe",      "{:>13.2f}"),
        ("Sortino",     "Sortino",     "{:>13.2f}"),
    ]

    col_w = 16
    header = f"{'Metric':<14}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * len(header))
    for label, key, fmt in metrics:
        row = f"{label:<14}"
        for n in names:
            row += f"{fmt.format(stats_by_name[n][key]):>{col_w}}"
        print(row)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_equity_comparison(
    equity_curves: Dict[str, pd.Series],
    title: str = "Strategy Comparison",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    save_path = save_path or (REPORTS_DIR / "backtest_comparison.png")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    styles = ["-", "--", ":"]
    for (name, eq), style in zip(equity_curves.items(), styles):
        eq.plot(ax=ax, label=name, linestyle=style)
    ax.set_title(title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot -> {save_path}")
    return save_path


def plot_regime_timeline(
    close: pd.Series,
    regime_label: pd.Series,
    title: str = "TSLA hourly close coloured by regime label",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Show the price line with range-bound regions shaded."""
    save_path = save_path or (REPORTS_DIR / "regime_timeline.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close.index, close.values, color="black", linewidth=0.8, label="Close")

    # Shade contiguous range-bound spans.
    in_range = (regime_label.fillna(0) == 1).astype(int).values
    starts = []
    ends = []
    i = 0
    n = len(in_range)
    while i < n:
        if in_range[i] == 1:
            j = i
            while j < n and in_range[j] == 1:
                j += 1
            starts.append(close.index[i])
            ends.append(close.index[min(j - 1, n - 1)])
            i = j
        else:
            i += 1

    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.18, color="orange")

    ax.set_title(title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Price ($)")
    ax.grid(alpha=0.3)
    # Custom legend entry for the shading.
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="orange", alpha=0.3, label="Range-bound (label=1)"))
    ax.legend(handles=handles, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot -> {save_path}")
    return save_path
