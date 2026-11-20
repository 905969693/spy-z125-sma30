"""Frozen parameters and official-sample conventions.

Do not change these values unless VY explicitly asks to unfreeze the rule.
"""
from __future__ import annotations

from typing import Final

import pandas as pd

BASKET: Final[tuple[str, ...]] = ("SPY", "TLT", "GLD", "DXY")

# DXY is the ICE U.S. Dollar Index.
YFINANCE_TICKERS: Final[dict[str, str]] = {
    "SPY": "SPY",
    "TLT": "TLT",
    "GLD": "GLD",
    "DXY": "DX-Y.NYB",
    "BIL": "BIL",
}

CORR_WINDOW: Final[int] = 63
DELTA_S_LAG: Final[int] = 21
Z_LOOKBACK: Final[int] = 180
Z_THRESHOLD: Final[float] = 1.25
SMA_WINDOW: Final[int] = 30
ANN_DAYS: Final[int] = 252
NAV_START: Final[pd.Timestamp] = pd.Timestamp("2007-05-31")

# Read-only official caches on the research machine. Never overwrite these.
OFFICIAL_CACHE_PATHS: Final[dict[str, str]] = {
    "SPY": "/workspace/dual-momentum/cache/SPY_max.csv",
    "TLT": "/workspace/ratio-momentum/cache/TLT_max.csv",
    "GLD": "/workspace/ratio-momentum/cache/GLD_max.csv",
    "DXY": "/workspace/cross-asset-corr-dxy-check/cache/DXY_max.csv",
    "BIL": "/workspace/dual-momentum/cache/BIL_max.csv",
}

# Official full-sample window used in the frozen-rule document (0-cost).
# Common-sample end was 2026-11-13 in the caches used to freeze the rule.
OFFICIAL_SAMPLE_FIRST_RETURN: Final[str] = "2007-06-01"
OFFICIAL_SAMPLE_END: Final[str] = "2026-11-13"

# Exact recomputes from gate_z_grid_daily.csv, official sample, 0 cost.
# Documented rounded figures are in docs/冻结规则_Z125_SPY30日均线.md.
OFFICIAL_RULE_STATS: Final[dict[str, float]] = {
    "n": 4894,
    "end_nav": 9.485260585694732,
    "cagr": 0.12281925563083029,
    "vol": 0.1762881675919777,
    "sharpe": 0.7456411196258603,
    "max_dd": -0.41405775386894783,
    "pct_cash": 0.04147936248467511,
    "n_cash": 203,
    "n_switches": 113,
}
OFFICIAL_SPY_STATS: Final[dict[str, float]] = {
    "n": 4894,
    "end_nav": 7.004619553668809,
    "cagr": 0.1054273923416249,
    "vol": 0.19682659718009568,
    "sharpe": 0.6078136906569952,
    "max_dd": -0.5518944037124107,
}

# Pre-specified crisis windows from the frozen-rule document (not fitted to this rule).
CRISIS_WINDOWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("金融危机", "2007-10-01", "2009-06-30"),
    ("欧债危机（2011 年美股主跌段）", "2011-04-15", "2011-11-30"),
    ("新冠疫情", "2020-02-14", "2020-04-30"),
    ("日本央行加息后日股闪崩", "2024-07-15", "2024-08-31"),
    ("Liberation Day", "2025-02-19", "2025-05-15"),
    ("伊朗战争", "2026-02-13", "2026-04-15"),
)
