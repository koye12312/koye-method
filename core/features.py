"""Causal feature engineering for the regime classifier.

Every feature here uses ONLY data up to and including time t. No look-ahead.
Features are designed to capture characteristics of recent price behaviour
that should help discriminate between range-bound and trending regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Volatility / range features
# --------------------------------------------------------------------------- #
def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rolling_volatility(close: pd.Series, window: int = 24) -> pd.Series:
    """Rolling standard deviation of log returns."""
    return log_returns(close).rolling(window).std()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range — Wilder's classic volatility measure."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def bollinger_width(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Width of Bollinger bands as a fraction of the moving average."""
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (2 * n_std * sd) / ma


# --------------------------------------------------------------------------- #
# Trend features
# --------------------------------------------------------------------------- #
def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index — high values indicate strong trend (regardless of direction)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)

    atr_w = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr_w)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr_w)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()


def dist_from_mean(close: pd.Series, window: int = 48) -> pd.Series:
    """Z-score of current price vs its rolling mean."""
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - ma) / sd


# --------------------------------------------------------------------------- #
# Distribution / persistence features
# --------------------------------------------------------------------------- #
def rolling_autocorr(close: pd.Series, window: int = 24, lag: int = 1) -> pd.Series:
    """Lag-1 autocorrelation of log returns over a rolling window."""
    rets = log_returns(close)
    return rets.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False)


def return_skew(close: pd.Series, window: int = 48) -> pd.Series:
    return log_returns(close).rolling(window).skew()


def return_kurtosis(close: pd.Series, window: int = 48) -> pd.Series:
    return log_returns(close).rolling(window).kurt()


def hurst_exponent(close: pd.Series, window: int = 100) -> pd.Series:
    """Hurst exponent over a rolling window via the rescaled-range (R/S) method.

    Interpretation:
      H < 0.5  -> mean-reverting series
      H ~ 0.5  -> random walk
      H > 0.5  -> trending / persistent series
    """
    log_p = np.log(close)

    def _hurst(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 20:
            return np.nan
        # Use a small set of lags (powers of 2 work well, capped at n//2).
        max_lag = min(40, n // 2)
        lags = np.unique(np.logspace(0.7, np.log10(max_lag), num=10).astype(int))
        lags = lags[lags >= 2]
        if len(lags) < 4:
            return np.nan
        tau = []
        for lag in lags:
            diff = arr[lag:] - arr[:-lag]
            sd = np.std(diff)
            if sd <= 0:
                return np.nan
            tau.append(sd)
        slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(slope)

    return log_p.rolling(window).apply(_hurst, raw=True)


# --------------------------------------------------------------------------- #
# Build the full feature frame
# --------------------------------------------------------------------------- #
FEATURE_NAMES = [
    "vol_24h",
    "atr_14",
    "bb_width_20",
    "adx_14",
    "dist_mean_48",
    "autocorr_24",
    "skew_48",
    "kurt_48",
    "hurst_100",
]


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature frame from an OHLCV dataframe."""
    o, h, l, c = ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"]

    feats = pd.DataFrame(index=ohlcv.index)
    feats["vol_24h"]      = rolling_volatility(c, 24)
    feats["atr_14"]       = atr(h, l, c, 14)
    feats["bb_width_20"]  = bollinger_width(c, 20)
    feats["adx_14"]       = adx(h, l, c, 14)
    feats["dist_mean_48"] = dist_from_mean(c, 48)
    feats["autocorr_24"]  = rolling_autocorr(c, 24)
    feats["skew_48"]      = return_skew(c, 48)
    feats["kurt_48"]      = return_kurtosis(c, 48)
    feats["hurst_100"]    = hurst_exponent(c, 100)

    return feats
