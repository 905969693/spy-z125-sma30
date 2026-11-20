"""Frozen SPY cash/invested rule: Z>1.25 of four-asset correlation speed, gated by SPY 30-day SMA.

Parameters are frozen. Do not change the basket, composite formula, windows,
threshold, or gate unless explicitly requested. Official definition:
``docs/冻结规则_Z125_SPY30日均线.md``.
"""

from .backtest import official_backtest, performance_stats, slice_returns
from .constants import (
    ANN_DAYS,
    BASKET,
    CORR_WINDOW,
    DELTA_S_LAG,
    NAV_START,
    SMA_WINDOW,
    Z_LOOKBACK,
    Z_THRESHOLD,
)
from .data import load_official_prices, load_prices
from .rule import compute_rule
from .signal import latest_signal

__version__ = "1.0.0"
__all__ = [
    "ANN_DAYS",
    "BASKET",
    "CORR_WINDOW",
    "DELTA_S_LAG",
    "NAV_START",
    "SMA_WINDOW",
    "Z_LOOKBACK",
    "Z_THRESHOLD",
    "compute_rule",
    "latest_signal",
    "load_official_prices",
    "load_prices",
    "official_backtest",
    "performance_stats",
    "slice_returns",
]
