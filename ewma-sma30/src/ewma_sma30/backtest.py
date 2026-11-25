"""Backtest helpers. Interval return excludes the start day's return.

The arithmetic matches the frozen package's sample conventions. The signal
being evaluated is the live EWMA rule, and the numbers are not a registered
track record.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import ANN_DAYS, NAV_START


def performance_stats(returns: pd.Series, ann: int = ANN_DAYS) -> dict[str, Any]:
    """Annualize with ``ann`` sessions. Vol and Sharpe use sample std ``ddof=1``. Sharpe rf is 0.

    Max drawdown is the minimum of nav/nav.cummax()-1 on the cumprod of the
    evaluated returns, the same definition used by the frozen package.
    """
    series = returns.dropna()
    if series.empty:
        raise ValueError("no returns to evaluate")
    n = len(series)
    nav = (1.0 + series).cumprod()
    end = float(nav.iloc[-1])
    cagr = end ** (ann / n) - 1.0
    sd = float(series.std(ddof=1))
    vol = float(sd * np.sqrt(ann))
    sharpe = float(series.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1.0
    return {
        "n": n,
        "end_nav": end,
        "cagr": float(cagr),
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(drawdown.min()),
        "start": str(series.index[0].date()),
        "end": str(series.index[-1].date()),
    }


def slice_returns(returns: pd.Series, start: pd.Timestamp | str = NAV_START) -> pd.Series:
    """Returns after ``start``. The start session's return is excluded, so NAV there is 1.

    If ``start`` is not a session, the first session on or after it is used
    as the NAV=1 date.
    """
    start_ts = pd.Timestamp(start)
    if start_ts not in returns.index:
        later = returns.loc[start_ts:]
        if later.empty:
            raise KeyError(f"no dates on or after {start_ts.date()}")
        start_ts = pd.Timestamp(later.index[0])
    loc = returns.index.get_loc(start_ts)
    if isinstance(loc, slice):
        loc = loc.start
    return returns.iloc[int(loc) + 1 :]


def cash_and_switches(applied: pd.Series) -> dict[str, float]:
    """Cash share and position changes, counted the same way as the frozen package.

    The first evaluated session counts as a switch because it differs from a
    missing prior position. Cash share ignores missing positions.
    """
    position = pd.to_numeric(applied, errors="coerce")
    n_cash = int((position == 1.0).sum())
    n_switches = int(position.ne(position.shift(1)).fillna(False).sum())
    pct_cash = float(position.mean()) if position.notna().any() else float("nan")
    return {"pct_cash": pct_cash, "n_cash": float(n_cash), "n_switches": float(n_switches)}


def run_backtest(
    daily: pd.DataFrame,
    end: str | pd.Timestamp | None = None,
    start: str | pd.Timestamp = NAV_START,
) -> dict[str, Any]:
    """Evaluate the live rule against buy-and-hold SPY.

    Default start is 2007-05-31 (NAV=1, first counted return the next session).
    Default end is the last row of ``daily``, which is the four-asset common
    sample end of the published signal.
    """
    frame = daily.sort_index()
    if end is not None:
        frame = frame.loc[: pd.Timestamp(end)]
    rule_r = slice_returns(frame["rule_ret"], start).dropna()
    spy_r = slice_returns(frame["spy_ret"], start).reindex(rule_r.index).dropna()
    rule = performance_stats(rule_r)
    spy = performance_stats(spy_r)
    applied = frame["applied_cash"] if "applied_cash" in frame.columns else frame["signal_cash"].shift(1)
    rule.update(cash_and_switches(applied.reindex(rule_r.index)))
    return {"rule": rule, "spy": spy}
