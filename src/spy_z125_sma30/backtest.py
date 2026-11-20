"""Official-sample backtest helpers. Interval return excludes the start day's return."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import ANN_DAYS, CRISIS_WINDOWS, NAV_START, OFFICIAL_SAMPLE_END


def performance_stats(returns: pd.Series, ann: int = ANN_DAYS) -> dict[str, Any]:
    r = returns.dropna()
    if r.empty:
        raise ValueError("no returns to evaluate")
    n = len(r)
    nav = (1 + r).cumprod()
    end = float(nav.iloc[-1])
    cagr = end ** (ann / n) - 1
    vol = float(r.std(ddof=1) * np.sqrt(ann))
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")
    dd = nav / nav.cummax() - 1
    return {
        "n": n,
        "end_nav": end,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(dd.min()),
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
    }


def slice_returns(returns: pd.Series, start: pd.Timestamp | str = NAV_START) -> pd.Series:
    """Returns after ``start`` (start day's return excluded). NAV at start is 1."""
    start_ts = pd.Timestamp(start)
    if start_ts not in returns.index:
        later = returns.loc[start_ts:]
        if later.empty:
            raise KeyError(f"no dates on or after {start_ts.date()}")
        start_ts = later.index[0]
    loc = returns.index.get_loc(start_ts)
    if isinstance(loc, slice):
        loc = loc.start
    return returns.iloc[int(loc) + 1 :]


def _align_applied(daily: pd.DataFrame, index: pd.Index) -> pd.Series:
    if "applied_cash" in daily.columns:
        applied = daily["applied_cash"]
    else:
        applied = daily["signal_cash"].shift(1)
    return applied.reindex(index)


def cash_and_switches(applied: pd.Series) -> dict[str, float]:
    a = applied
    n_cash = int(a.fillna(False).sum())
    return {
        "pct_cash": float(a.mean()) if a.notna().any() else float("nan"),
        "n_cash": n_cash,
        "n_switches": int((a.ne(a.shift(1))).fillna(False).sum()),
    }


def official_backtest(
    daily: pd.DataFrame,
    end: str | pd.Timestamp | None = OFFICIAL_SAMPLE_END,
    start: str | pd.Timestamp = NAV_START,
) -> dict[str, Any]:
    """Full official sample (or a capped end) for the rule versus buy-and-hold SPY."""
    df = daily.sort_index()
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    rule_r = slice_returns(df["rule_ret"], start).dropna()
    spy_r = slice_returns(df["spy_ret"], start).reindex(rule_r.index).dropna()
    rule = performance_stats(rule_r)
    spy = performance_stats(spy_r)
    applied = _align_applied(df, rule_r.index)
    rule.update(cash_and_switches(applied))
    return {"rule": rule, "spy": spy}


def window_backtest(
    daily: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    """Interval return and intra-window max drawdown; history before start may form signals."""
    df = daily.sort_index()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts not in df.index:
        later = df.loc[start_ts:end_ts]
        if later.empty:
            raise KeyError(f"no dates in [{start_ts.date()}, {end_ts.date()}]")
        start_ts = later.index[0]
    rule_r = slice_returns(df["rule_ret"], start_ts).loc[:end_ts].dropna()
    spy_r = slice_returns(df["spy_ret"], start_ts).loc[:end_ts].reindex(rule_r.index).dropna()
    rule = performance_stats(rule_r)
    spy = performance_stats(spy_r)
    applied = _align_applied(df, rule_r.index)
    rule.update(cash_and_switches(applied))
    return {"rule": rule, "spy": spy, "window_start": str(start_ts.date()), "window_end": str(end_ts.date())}


def crisis_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, start, end in CRISIS_WINDOWS:
        try:
            out = window_backtest(daily, start, end)
        except (KeyError, ValueError):
            continue
        rule, spy = out["rule"], out["spy"]
        rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "rule_ret": rule["end_nav"] - 1,
                "rule_max_dd": rule["max_dd"],
                "spy_ret": spy["end_nav"] - 1,
                "spy_max_dd": spy["max_dd"],
                "pct_cash": rule["pct_cash"],
                "n_cash": rule["n_cash"],
            }
        )
    return pd.DataFrame(rows)
