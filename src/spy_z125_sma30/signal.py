"""Latest close signal for the frozen rule (effective the next session)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import Z_THRESHOLD


def latest_signal(daily: pd.DataFrame) -> dict[str, Any]:
    """Read the last row with a non-missing close signal."""
    df = daily.sort_index()
    if "signal_cash" not in df.columns:
        raise ValueError("daily frame has no signal_cash column")
    usable = df.loc[df["signal_cash"].notna()]
    if usable.empty:
        raise ValueError("no formed close signal yet")
    row = usable.iloc[-1]
    dt = usable.index[-1]
    z = float(row["z"]) if pd.notna(row["z"]) else float("nan")
    s = float(row["s"]) if pd.notna(row["s"]) else float("nan")
    spy = float(row["spy"])
    sma = float(row["spy_ma30"]) if pd.notna(row["spy_ma30"]) else float("nan")
    below = bool(row["below_ma30"]) if pd.notna(row["below_ma30"]) else False
    z_trigger = bool(z > Z_THRESHOLD) if pd.notna(z) else False
    cash = bool(row["signal_cash"])
    return {
        "asof": str(pd.Timestamp(dt).date()),
        "s": s,
        "z": z,
        "spy": spy,
        "spy_ma30": sma,
        "below_ma30": below,
        "z_gt_125": z_trigger,
        "close_signal": "cash" if cash else "invested",
        "next_session": "cash" if cash else "invested",
        "signal_cash": cash,
    }
