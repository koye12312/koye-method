"""Download and load hourly OHLCV price data from Yahoo Finance."""
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop yfinance's multi-index columns into single-level OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        # Pick the field level (Open/High/Low/Close/Volume) regardless of order.
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(set(df.columns.get_level_values(0))):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        elif {"Open", "High", "Low", "Close", "Volume"}.issubset(set(df.columns.get_level_values(-1))):
            df = df.copy()
            df.columns = df.columns.get_level_values(-1)
        else:
            raise ValueError(f"Unexpected column structure for {ticker}: {df.columns.tolist()}")
    return df


def download_hourly(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download hourly OHLCV bars and save to ``data/<TICKER>_1h.csv``.

    Yahoo only allows ~730 days of hourly history. ``period`` accepts standard
    yfinance values like ``"60d"``, ``"6mo"``, ``"1y"``, ``"2y"``.
    """
    print(f"Downloading {ticker} hourly bars (period={period})...")
    df = yf.download(
        ticker,
        period=period,
        interval="1h",
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}.")

    df = _flatten_columns(df, ticker)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[keep].dropna()

    df.index.name = "Datetime"
    out_path = DATA_DIR / f"{ticker}_1h.csv"
    df.to_csv(out_path)
    print(f"  Saved {len(df):,} bars -> {out_path}")
    return df


def load_hourly(ticker: str) -> pd.DataFrame:
    """Load a previously-downloaded hourly OHLCV CSV."""
    path = DATA_DIR / f"{ticker}_1h.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python 01_download_data.py` first."
        )
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return df.sort_index()
