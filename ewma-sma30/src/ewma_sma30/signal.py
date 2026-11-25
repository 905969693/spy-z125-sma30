"""Latest close signal for the live EWMA rule, effective the next session."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .constants import (
    BASKET,
    DROP_TRIGGER,
    LAMBDA,
    PCT_RISK_ON,
    RULE2_EXIT_PCT,
    SESSION_CLOSE_HOUR,
    SESSION_CLOSE_MINUTE,
    SMA_WINDOW,
)
from .data import prepare_closes

_NY = ZoneInfo("America/New_York")


def last_allowed_session_date(now: datetime | None = None) -> pd.Timestamp:
    """Last US cash session that should already be closed.

    Weekend clocks, and weekday clocks before 16:15 America/New_York, use the
    previous weekday. A naive ``now`` is read as New York local time. 16:15
    itself counts as closed.
    """
    if now is None:
        now_ny = datetime.now(_NY)
    elif now.tzinfo is None:
        now_ny = now.replace(tzinfo=_NY)
    else:
        now_ny = now.astimezone(_NY)
    session = now_ny.date()
    close_time = time(SESSION_CLOSE_HOUR, SESSION_CLOSE_MINUTE)
    if now_ny.weekday() >= 5 or now_ny.time() < close_time:
        session = session - timedelta(days=1)
        while session.weekday() >= 5:
            session = session - timedelta(days=1)
    return pd.Timestamp(session)


def closes_through_last_session(closes: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop sessions after the last completed US cash close."""
    limit = last_allowed_session_date(now)
    return closes.loc[closes.index <= limit]


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _ma_relation(close: float | None, sma: float | None) -> str:
    if close is None or sma is None:
        return "unknown"
    if close < sma:
        return "strictly_below"
    if close > sma:
        return "strictly_above"
    return "equal"


def latest_signal(daily: pd.DataFrame, asof: pd.Timestamp | str | None = None) -> dict[str, Any]:
    """Read one formed close signal. Default is the last row with a signal.

    ``next_session`` repeats the close signal because that signal is applied
    on the following session. This does not submit an order.
    """
    frame = daily.sort_index()
    required = ("signal_cash", "S", "pct", "spy_close", "spy_sma30", "rule1_sig", "rule2_sig")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("daily frame is missing columns: " + ", ".join(missing))
    if asof is None:
        usable = frame.loc[frame["signal_cash"].notna()]
        if usable.empty:
            raise ValueError("no formed close signal yet")
        asof_ts = pd.Timestamp(usable.index[-1])
    else:
        asof_ts = pd.Timestamp(asof)
        if asof_ts not in frame.index or pd.isna(frame.loc[asof_ts, "signal_cash"]):
            raise ValueError(f"no formed close signal on {asof_ts.date()}")
    row = frame.loc[asof_ts]
    cash = bool(row["signal_cash"] == 1)
    spy = _optional_float(row["spy_close"])
    sma = _optional_float(row["spy_sma30"])
    rule2_raw = row["rule2_sig"]
    return {
        "asof": str(asof_ts.date()),
        "S": float(row["S"]),
        "pct": float(row["pct"]),
        "drop5": _optional_float(row["drop5"]) if "drop5" in frame.columns else None,
        "high5": _optional_float(row["high5"]) if "high5" in frame.columns else None,
        "z_spy_tlt": _optional_float(row["z_spy_tlt"]) if "z_spy_tlt" in frame.columns else None,
        "z_spy_dxy": _optional_float(row["z_spy_dxy"]) if "z_spy_dxy" in frame.columns else None,
        "z_spy_gld": _optional_float(row["z_spy_gld"]) if "z_spy_gld" in frame.columns else None,
        "z_tlt_dxy": _optional_float(row["z_tlt_dxy"]) if "z_tlt_dxy" in frame.columns else None,
        "spy": spy,
        "spy_sma30": sma,
        "below_ma30": bool(row["below_ma"] == 1) if "below_ma" in frame.columns and pd.notna(row["below_ma"]) else False,
        "ma_relation": _ma_relation(spy, sma),
        "risk_off_r1": bool(row["risk_off_r1"] == 1) if "risk_off_r1" in frame.columns and pd.notna(row["risk_off_r1"]) else False,
        "latched": bool(row["latched"] == 1) if "latched" in frame.columns and pd.notna(row["latched"]) else False,
        "rule1_triggered": bool(row["rule1_sig"] == 1),
        "rule2_triggered": None if pd.isna(rule2_raw) else bool(rule2_raw == 1),
        "close_signal": "cash" if cash else "invested",
        "next_session": "cash" if cash else "invested",
        "signal_cash": cash,
        "params": {
            "sma": SMA_WINDOW,
            "lambda": LAMBDA,
            "drop": DROP_TRIGGER,
            "pct_risk_on": PCT_RISK_ON,
            "rule2_exit_pct": RULE2_EXIT_PCT,
        },
    }


def latest_complete_signal(closes: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """Signal on the last SPY session in ``closes``.

    If that session is missing any of SPY, TLT, GLD, or DXY, raise instead of
    silently reusing an older common session.
    """
    prices = prepare_closes(closes)
    spy = prices["SPY"].dropna().sort_index()
    if spy.empty:
        raise RuntimeError("no SPY closes in the loaded sample")
    asof = pd.Timestamp(spy.index[-1])
    missing = []
    for name in BASKET:
        series = prices[name].dropna()
        if asof not in series.index:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"last SPY session {asof.date()} is not in the four-asset sample; missing "
            + ", ".join(missing)
        )
    if asof not in daily.index or pd.isna(daily.loc[asof, "signal_cash"]):
        raise RuntimeError(f"EWMA signal is not formed on the last complete session {asof.date()}")
    return latest_signal(daily, asof=asof)
