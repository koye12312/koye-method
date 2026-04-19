"""The Koye Method: ML-gated mean-reversion grid execution.

When the regime classifier predicts a high-confidence range-bound window,
we deploy a grid of buy and sell orders inside the predicted range:

  - N evenly-spaced levels between (centre - width/2) and (centre + width/2)
  - Buy levels below the centre, sell levels above
  - Equal capital allocation per level
  - HARD STOP: close everything if price breaks centre +/- stop_mult * width
  - TIME STOP: close everything cleanly after H bars

Each round-trip (buy at level L, later sell back at level L + spacing)
captures the spacing as profit. Breakouts are accepted as the cost of
playing this strategy and are bounded by the stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class GridConfig:
    n_levels: int = 5                # total number of price levels in the grid
    confidence_threshold: float = 0.55  # min P(range-bound) to deploy
    stop_mult: float = 1.5           # hard stop at centre +/- stop_mult * (width/2)
    horizon: int = 8                 # close out after this many bars
    capital_fraction: float = 0.95   # fraction of capital deployed per grid


@dataclass
class GridState:
    centre: float
    half_width: float
    stop_low: float
    stop_high: float
    bars_remaining: int
    levels: List[float]              # sorted ascending
    holdings: dict = field(default_factory=dict)   # level -> shares held (0 if not filled)
    cost_basis: dict = field(default_factory=dict) # level -> avg cost
    capital_per_level: float = 0.0


def _build_grid(centre: float, width: float, n_levels: int) -> List[float]:
    half = width / 2.0
    return list(np.linspace(centre - half, centre + half, n_levels))


def run_grid_backtest(
    ohlcv: pd.DataFrame,
    proba_range_bound: pd.Series,
    range_centre: pd.Series,
    range_width_pct: pd.Series,
    initial_capital: float = 10_000.0,
    cfg: GridConfig = GridConfig(),
    cost_per_trade: float = 0.0005,
) -> pd.DataFrame:
    """Run the Koye Method grid strategy bar by bar.

    Returns a frame indexed by datetime with columns:
      equity, cash, position_value, in_grid, n_round_trips, breakout
    """
    idx = ohlcv.index
    cash = float(initial_capital)
    realised_pnl = 0.0
    state: Optional[GridState] = None

    log = pd.DataFrame(
        index=idx,
        columns=["equity", "cash", "position_value", "in_grid", "n_round_trips", "breakout"],
        dtype=float,
    )
    log["in_grid"] = 0.0
    log["n_round_trips"] = 0.0
    log["breakout"] = 0.0

    round_trips_total = 0
    breakouts_total = 0

    for t, ts in enumerate(idx):
        bar = ohlcv.iloc[t]
        bar_open, bar_high, bar_low, bar_close = bar["Open"], bar["High"], bar["Low"], bar["Close"]

        # -------- MANAGE EXISTING GRID --------
        if state is not None:
            broke_out = False

            # Check stops first using bar high/low.
            if bar_low <= state.stop_low or bar_high >= state.stop_high:
                # Liquidate everything at the stop level (worst-case fill).
                exit_price = state.stop_low if bar_low <= state.stop_low else state.stop_high
                for lvl, shares in list(state.holdings.items()):
                    if shares > 0:
                        proceeds = shares * exit_price * (1 - cost_per_trade)
                        realised_pnl += proceeds - state.cost_basis[lvl] * shares
                        cash += proceeds
                        state.holdings[lvl] = 0
                broke_out = True
                breakouts_total += 1
                state = None
                log.iat[t, log.columns.get_loc("breakout")] = 1.0

            else:
                # Process limit-order fills inside the bar.
                # Sells first (take profit on existing holdings).
                for lvl in state.levels:
                    if state.holdings.get(lvl, 0) > 0:
                        # Sell when price rises one level above the buy level.
                        sell_target = lvl + (state.levels[1] - state.levels[0])
                        if bar_high >= sell_target and sell_target <= state.stop_high:
                            shares = state.holdings[lvl]
                            proceeds = shares * sell_target * (1 - cost_per_trade)
                            cost = state.cost_basis[lvl] * shares
                            realised_pnl += proceeds - cost
                            cash += proceeds
                            state.holdings[lvl] = 0
                            round_trips_total += 1

                # Buys: fill levels below centre that are touched and not already held.
                for lvl in state.levels:
                    if lvl < state.centre and state.holdings.get(lvl, 0) == 0 and bar_low <= lvl:
                        # Spend capital_per_level on shares at this level.
                        if cash >= state.capital_per_level:
                            shares = state.capital_per_level / lvl
                            cost = shares * lvl * (1 + cost_per_trade)
                            cash -= cost
                            state.holdings[lvl] = shares
                            state.cost_basis[lvl] = lvl * (1 + cost_per_trade)

                state.bars_remaining -= 1
                # Time stop: liquidate cleanly at close.
                if state.bars_remaining <= 0:
                    for lvl, shares in list(state.holdings.items()):
                        if shares > 0:
                            proceeds = shares * bar_close * (1 - cost_per_trade)
                            realised_pnl += proceeds - state.cost_basis[lvl] * shares
                            cash += proceeds
                            state.holdings[lvl] = 0
                    state = None

        # -------- DEPLOY NEW GRID? --------
        if state is None and t < len(idx) - cfg.horizon:
            p = proba_range_bound.iat[t] if t < len(proba_range_bound) else np.nan
            centre = range_centre.iat[t] if t < len(range_centre) else np.nan
            width_pct = range_width_pct.iat[t] if t < len(range_width_pct) else np.nan

            if (
                not np.isnan(p)
                and p >= cfg.confidence_threshold
                and not np.isnan(centre)
                and width_pct > 0
            ):
                width = centre * width_pct
                half = width / 2.0
                levels = _build_grid(centre, width, cfg.n_levels)
                deployable = cash * cfg.capital_fraction
                state = GridState(
                    centre=centre,
                    half_width=half,
                    stop_low=centre - cfg.stop_mult * half,
                    stop_high=centre + cfg.stop_mult * half,
                    bars_remaining=cfg.horizon,
                    levels=levels,
                    holdings={lvl: 0.0 for lvl in levels},
                    cost_basis={lvl: 0.0 for lvl in levels},
                    capital_per_level=deployable / cfg.n_levels,
                )

        # -------- LOG STATE --------
        position_value = 0.0
        if state is not None:
            for lvl, shares in state.holdings.items():
                position_value += shares * bar_close
        log.iat[t, log.columns.get_loc("equity")] = cash + position_value
        log.iat[t, log.columns.get_loc("cash")] = cash
        log.iat[t, log.columns.get_loc("position_value")] = position_value
        log.iat[t, log.columns.get_loc("in_grid")] = 1.0 if state is not None else 0.0
        log.iat[t, log.columns.get_loc("n_round_trips")] = round_trips_total

    log.attrs["total_round_trips"] = round_trips_total
    log.attrs["total_breakouts"] = breakouts_total
    return log
