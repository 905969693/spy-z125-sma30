"""Live cash/invested rule: rule 1 percentile latch OR rule 2 five-day drop.

Parameters are the module constants, not function arguments. Signal at the
close of t is applied on t+1. This module does not place orders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .composite import basket_prices, compute_composite
from .constants import (
    CASH_ASSET,
    DROP_TRIGGER,
    DROP_WINDOW,
    PCT_RISK_ON,
    RULE2_EXIT_PCT,
    SMA_WINDOW,
)
from .data import prepare_closes


def _float_series(series: pd.Series, index: pd.Index | None = None) -> pd.Series:
    aligned = pd.Series(series)
    if index is not None:
        aligned = aligned.reindex(index)
    return pd.to_numeric(aligned, errors="coerce").astype(float)


def spy_sma(spy: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    """Simple mean of SPY's own closes, including today. Needs ``window`` sessions."""
    own = _float_series(spy).dropna().sort_index()
    own = own[~own.index.duplicated(keep="last")]
    return own.rolling(window, min_periods=window).mean()


def own_calendar_simple_return(prices: pd.Series, index: pd.Index) -> pd.Series:
    """Simple return on the asset's own calendar, then aligned to ``index``."""
    own = _float_series(prices).dropna().sort_index()
    own = own[~own.index.duplicated(keep="last")]
    return own.pct_change().reindex(index)


def rule1_risk_off(pct: pd.Series) -> pd.Series:
    """1 when percentile > 0.9, 0 when it is present and not above 0.9, else missing."""
    pct = _float_series(pct)
    flag = (pct > PCT_RISK_ON).astype(float).where(pct.notna())
    flag.name = "risk_off_r1"
    return flag


def apply_latch(risk_off: pd.Series, below_ma: pd.Series) -> pd.DataFrame:
    """Cash latch for one rule-1 risk-off episode.

    The latch turns on the first session of the episode with SPY strictly
    below SMA30, and stays on after price rises back above the average.
    It clears only when rule-1 risk_off returns to 0. A missing risk_off
    value leaves the latch unchanged and writes a missing signal for that day.
    Before the latch trips, rule 1's close signal is invested.
    """
    risk = _float_series(risk_off)
    below = _float_series(below_ma, risk.index)
    latched = np.full(len(risk), np.nan, dtype=np.float64)
    signal = np.full(len(risk), np.nan, dtype=np.float64)
    latch = 0.0
    risk_values = risk.to_numpy(dtype=np.float64)
    below_values = below.to_numpy(dtype=np.float64)
    for i, (state, is_below) in enumerate(zip(risk_values, below_values)):
        if np.isnan(state):
            continue
        if state == 0.0:
            latch = 0.0
            latched[i] = 0.0
            signal[i] = 0.0
            continue
        if np.isfinite(is_below) and is_below == 1.0:
            latch = 1.0
        latched[i] = latch
        signal[i] = latch
    return pd.DataFrame({"latched": latched, "signal_cash": signal}, index=risk.index)


def rule2_state(
    pct: pd.Series,
    below_ma: pd.Series,
    spy_close: pd.Series,
    sma30: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rule 2 close signal, the 5-session percentile high, and drop5.

    Exits are applied before same-day re-entry. If both fire, the close
    signal is cash when the entry test still holds. A missing percentile or
    drop5 does not change the position state; that session's signal is missing.
    """
    pct = _float_series(pct)
    below = _float_series(below_ma, pct.index)
    close = _float_series(spy_close, pct.index)
    sma = _float_series(sma30, pct.index)
    high5 = pct.rolling(DROP_WINDOW, min_periods=DROP_WINDOW).max()
    drop5 = (pct - high5).rename("drop5")
    high5 = high5.rename("high5")

    signal = np.full(len(pct), np.nan, dtype=np.float64)
    in_cash = 0.0
    pct_values = pct.to_numpy(dtype=np.float64)
    drop_values = drop5.to_numpy(dtype=np.float64)
    below_values = below.to_numpy(dtype=np.float64)
    close_values = close.to_numpy(dtype=np.float64)
    sma_values = sma.to_numpy(dtype=np.float64)
    for i in range(len(pct)):
        if np.isnan(pct_values[i]) or np.isnan(drop_values[i]):
            continue
        if in_cash == 1.0:
            previous = pct_values[i - 1] if i > 0 else np.nan
            above_ma = (
                np.isfinite(close_values[i])
                and np.isfinite(sma_values[i])
                and close_values[i] > sma_values[i]
            )
            crossed = (
                np.isfinite(previous)
                and previous < RULE2_EXIT_PCT
                and pct_values[i] >= RULE2_EXIT_PCT
            )
            if crossed or above_ma:
                in_cash = 0.0
        if in_cash == 0.0:
            entered = (
                np.isfinite(drop_values[i])
                and np.isfinite(below_values[i])
                and drop_values[i] <= DROP_TRIGGER
                and below_values[i] == 1.0
            )
            if entered:
                in_cash = 1.0
        signal[i] = in_cash
    return pd.Series(signal, index=pct.index, name="rule2_sig"), high5, drop5


def combine_signals(rule1: pd.Series, rule2: pd.Series, pct: pd.Series) -> pd.Series:
    """Close signal is rule 1 OR rule 2. Missing percentile stays missing."""
    pct = _float_series(pct)
    first = _float_series(rule1, pct.index)
    second = _float_series(rule2, pct.index)
    combined = ((first.fillna(0.0) == 1.0) | (second.fillna(0.0) == 1.0)).astype(float)
    out = combined.where(pct.notna())
    out.name = "signal_cash"
    return out


def apply_next_day(
    signal_cash: pd.Series,
    spy_ret: pd.Series,
    bil_ret: pd.Series,
) -> pd.DataFrame:
    """Position is yesterday's close signal. Cash earns BIL; otherwise SPY.

    The session that flips the close signal still earns the previous position.
    Until a previous signal exists, the rule return is missing.
    """
    applied = signal_cash.shift(1).rename("applied_cash")
    spy = _float_series(spy_ret, signal_cash.index)
    bil = _float_series(bil_ret, signal_cash.index)
    rule_ret = pd.Series(
        np.where(applied.to_numpy(dtype=float) == 1.0, bil.to_numpy(dtype=float), spy.to_numpy(dtype=float)),
        index=signal_cash.index,
        dtype="float64",
        name="rule_ret",
    )
    rule_ret = rule_ret.where(applied.notna())
    return pd.DataFrame({"applied_cash": applied, "rule_ret": rule_ret})


def build_signals(
    pct: pd.Series,
    spy_close: pd.Series,
    sma: pd.Series,
    below: pd.Series | None = None,
) -> pd.DataFrame:
    """Rule columns from an already-built percentile, close, and SMA30.

    ``below`` is derived from a strict close < SMA comparison when omitted.
    Equality is neither below nor above.
    """
    pct = _float_series(pct)
    close = _float_series(spy_close, pct.index).rename("spy_close")
    average = _float_series(sma, pct.index).rename("spy_sma30")
    if below is None:
        below_flag = (close < average).astype(float).where(average.notna())
    else:
        below_flag = _float_series(below, pct.index)
    below_flag = below_flag.rename("below_ma")
    risk = rule1_risk_off(pct)
    latch = apply_latch(risk, below_flag)
    rule2, high5, drop5 = rule2_state(pct, below_flag, close, average)
    signal = combine_signals(latch["signal_cash"], rule2, pct)
    return pd.DataFrame(
        {
            "pct": pct.rename("pct"),
            "high5": high5,
            "drop5": drop5,
            "risk_off_r1": risk,
            "latched": latch["latched"],
            "rule1_sig": latch["signal_cash"].rename("rule1_sig"),
            "rule2_sig": rule2,
            "below_ma": below_flag,
            "spy_close": close,
            "spy_sma30": average,
            "signal_cash": signal,
        }
    )


def compute_rule(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily live EWMA signal and next-day SPY/BIL returns.

    The composite uses the four-asset inner join. SMA30 and the strategy
    returns use each asset's own calendar, then line up with the published
    composite dates. Cost is zero. Nothing here submits an order.
    """
    prices = prepare_closes(closes)
    panel = basket_prices(prices)
    published = compute_composite(prices)

    spy_own = prices["SPY"].dropna().sort_index()
    spy_own = spy_own[~spy_own.index.duplicated(keep="last")]
    sma_own = spy_own.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    below_own = (spy_own < sma_own).astype(float).where(sma_own.notna())

    signals = build_signals(
        published["pct"],
        panel["SPY"].reindex(published.index),
        sma_own.reindex(published.index),
        below_own.reindex(published.index),
    )
    spy_ret = own_calendar_simple_return(prices["SPY"], published.index).rename("spy_ret")
    bil_ret = own_calendar_simple_return(prices[CASH_ASSET], published.index).rename("bil_ret")
    position = apply_next_day(signals["signal_cash"], spy_ret, bil_ret)

    frame = published.join(signals.drop(columns=["pct"]), how="left")
    frame["spy_ret"] = spy_ret
    frame["bil_ret"] = bil_ret
    frame["applied_cash"] = position["applied_cash"]
    frame["rule_ret"] = position["rule_ret"]
    frame.index.name = "date"
    return frame
