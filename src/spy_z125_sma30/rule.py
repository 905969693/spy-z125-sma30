"""Compute the frozen cash/invested signal. Parameters are not configurable."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    BASKET,
    CORR_WINDOW,
    DELTA_S_LAG,
    SMA_WINDOW,
    Z_LOOKBACK,
    Z_THRESHOLD,
)


def _pair_corr(rets: pd.DataFrame, a: str, b: str) -> pd.Series:
    return rets[a].rolling(CORR_WINDOW, min_periods=CORR_WINDOW).corr(rets[b])


def composite_s(prices: pd.DataFrame) -> pd.Series:
    """s = -corr(SPY,TLT) -corr(SPY,DXY) +corr(SPY,GLD) +corr(TLT,DXY)."""
    rets = prices.loc[:, list(BASKET)].pct_change()
    spy_tlt = _pair_corr(rets, "SPY", "TLT")
    spy_dxy = _pair_corr(rets, "SPY", "DXY")
    spy_gld = _pair_corr(rets, "SPY", "GLD")
    tlt_dxy = _pair_corr(rets, "TLT", "DXY")
    s = (-spy_tlt - spy_dxy + spy_gld + tlt_dxy).rename("s")
    return s


def speed_z(s: pd.Series) -> pd.Series:
    """Causal Z of the 21-day change in s versus the prior 180 observations."""
    dx = s - s.shift(DELTA_S_LAG)
    mu = dx.shift(1).rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).mean()
    sd = dx.shift(1).rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).std(ddof=1)
    return ((dx - mu) / sd).rename("z")


def compute_rule(closes: pd.DataFrame) -> pd.DataFrame:
    """Build the daily rule series from adjusted closes.

    Cash signal at close of t is true only if Z_t > 1.25 and SPY_t is strictly
    below its 30-day SMA (including t). The signal is applied on t+1. Cash
    earns BIL total return. Cost is 0. Missing Z keeps the close signal missing.
    """
    missing = [c for c in (*BASKET, "BIL") if c not in closes.columns]
    if missing:
        raise ValueError(f"closes is missing columns: {missing}")

    px = closes.loc[:, list(BASKET)].sort_index().dropna(how="any")
    rets = px.pct_change()
    spy_tlt = _pair_corr(rets, "SPY", "TLT")
    spy_dxy = _pair_corr(rets, "SPY", "DXY")
    spy_gld = _pair_corr(rets, "SPY", "GLD")
    tlt_dxy = _pair_corr(rets, "TLT", "DXY")
    s = (-spy_tlt - spy_dxy + spy_gld + tlt_dxy).rename("s")
    z = speed_z(s)

    spy = px["SPY"].rename("spy")
    sma30 = spy.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean().rename("spy_ma30")
    raw_z = (z > Z_THRESHOLD).where(z.notna())
    below = (spy < sma30).where(sma30.notna()).rename("below_ma30")
    signal_cash = ((raw_z == True) & (below == True)).where(raw_z.notna())  # noqa: E712
    signal_cash = signal_cash.rename("signal_cash")
    applied_cash = signal_cash.shift(1).rename("applied_cash")

    # Returns are computed on each asset's own calendar, then joined to the
    # four-asset index. This matches the official series: a SPY session that
    # is missing from the inner join (e.g. 2016-10-10, 2016-11-11) still
    # affects the next joined day's SPY return.
    spy_ret = (
        pd.to_numeric(closes["SPY"], errors="coerce")
        .sort_index()
        .dropna()
        .pct_change()
        .reindex(px.index)
        .rename("spy_ret")
    )
    bil_ret = (
        pd.to_numeric(closes["BIL"], errors="coerce")
        .sort_index()
        .dropna()
        .pct_change()
        .reindex(px.index)
        .rename("bil_ret")
    )
    rule_ret = pd.Series(
        np.where(applied_cash, bil_ret, spy_ret),
        index=px.index,
        dtype="float64",
        name="rule_ret",
    )
    rule_ret = rule_ret.where(applied_cash.notna())

    return pd.DataFrame(
        {
            "s": s,
            "z": z,
            "SPY-TLT": spy_tlt,
            "SPY-DXY": spy_dxy,
            "SPY-GLD": spy_gld,
            "TLT-DXY": tlt_dxy,
            "spy": spy,
            "spy_ma30": sma30,
            "spy_ret": spy_ret,
            "bil_ret": bil_ret,
            "below_ma30": below,
            "signal_cash": signal_cash,
            "applied_cash": applied_cash,
            "rule_ret": rule_ret,
        }
    )
