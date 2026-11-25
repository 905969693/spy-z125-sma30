"""Live SPY cash/invested signal from an EWMA signed covariance.

This package is not frozen and is not a replacement for the repository's
Pearson Z>1.25 + SMA30 rule. It only computes signals and backtests.
It does not place or modify orders.
"""

from .backtest import performance_stats, run_backtest, slice_returns
from .composite import compute_composite
from .constants import (
    ANN_DAYS,
    BASKET,
    BURNIN,
    DROP_TRIGGER,
    LAMBDA,
    NAV_START,
    PCTL_MIN_PERIODS,
    PCTL_WINDOW,
    PCT_RISK_ON,
    RULE2_EXIT_PCT,
    SD_FLOOR,
    SMA_WINDOW,
    Z_MIN_PERIODS,
    Z_WINDOW,
)
from .data import load_prices
from .rule import compute_rule
from .signal import latest_signal

__version__ = "0.1.0"
__all__ = [
    "ANN_DAYS",
    "BASKET",
    "BURNIN",
    "DROP_TRIGGER",
    "LAMBDA",
    "NAV_START",
    "PCTL_MIN_PERIODS",
    "PCTL_WINDOW",
    "PCT_RISK_ON",
    "RULE2_EXIT_PCT",
    "SD_FLOOR",
    "SMA_WINDOW",
    "Z_MIN_PERIODS",
    "Z_WINDOW",
    "compute_composite",
    "compute_rule",
    "latest_signal",
    "load_prices",
    "performance_stats",
    "run_backtest",
    "slice_returns",
]
