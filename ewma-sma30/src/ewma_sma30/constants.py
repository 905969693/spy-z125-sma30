"""Live EWMA signal parameters.

These are the current live defaults. They are not a frozen rule, not a
pre-registered test, and not an out-of-sample optimum. Do not add a search
over lambda, the drop trigger, the moving-average length, or a minimum hold.
"""
from __future__ import annotations

from typing import Final

import pandas as pd

BASKET: Final[tuple[str, ...]] = ("SPY", "TLT", "GLD", "DXY")
CASH_ASSET: Final[str] = "BIL"
NEEDED: Final[tuple[str, ...]] = (*BASKET, CASH_ASSET)

# DXY is the ICE U.S. Dollar Index, not the UUP ETF.
YFINANCE_TICKERS: Final[dict[str, str]] = {
    "SPY": "SPY",
    "TLT": "TLT",
    "GLD": "GLD",
    "DXY": "DX-Y.NYB",
    "BIL": "BIL",
}

# S = -z(SPY,TLT) - z(SPY,DXY) + z(SPY,GLD) + z(TLT,DXY)
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("SPY", "TLT"),
    ("SPY", "DXY"),
    ("SPY", "GLD"),
    ("TLT", "DXY"),
)
PAIR_SIGN: Final[dict[tuple[str, str], float]] = {
    ("SPY", "TLT"): -1.0,
    ("SPY", "DXY"): -1.0,
    ("SPY", "GLD"): 1.0,
    ("TLT", "DXY"): 1.0,
}

LAMBDA: Final[float] = 0.94
BURNIN: Final[int] = 60
Z_WINDOW: Final[int] = 504
Z_MIN_PERIODS: Final[int] = 252
PCTL_WINDOW: Final[int] = 252
PCTL_MIN_PERIODS: Final[int] = 126
SD_FLOOR: Final[float] = 1e-12

SMA_WINDOW: Final[int] = 30
# Rule 1 is in its risk-off state only when the percentile is strictly above this.
PCT_RISK_ON: Final[float] = 0.9
# Rule 2 enters when pct - max(pct over 5 sessions including today) <= this.
DROP_TRIGGER: Final[float] = -0.70
DROP_WINDOW: Final[int] = 5
# Rule 2 exits on an upward cross of this level: previous < level and today >= level.
RULE2_EXIT_PCT: Final[float] = 0.25

ANN_DAYS: Final[int] = 252
NAV_START: Final[pd.Timestamp] = pd.Timestamp("2007-05-31")

# A US cash session is treated as complete at 16:15 America/New_York, inclusive.
SESSION_CLOSE_HOUR: Final[int] = 16
SESSION_CLOSE_MINUTE: Final[int] = 15

# Never write downloaded prices into these research caches or the frozen fixture.
FORBIDDEN_WRITE_ROOTS: Final[tuple[str, ...]] = (
    "/workspace/dual-momentum/cache",
    "/workspace/ratio-momentum/cache",
    "/workspace/cross-asset-corr-dxy-check/cache",
    "/workspace/spy-z125-sma30/cache",
)
