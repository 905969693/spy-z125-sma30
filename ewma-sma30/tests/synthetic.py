"""Synthetic positive prices for tests. No cache and no network."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_closes(n: int = 900, seed: int = 0, start: str = "2004-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    data = {}
    for name in ("SPY", "TLT", "GLD", "DXY"):
        shock = rng.normal(0.0002, 0.01, n)
        data[name] = 100.0 * np.exp(np.cumsum(shock))
    data["BIL"] = 100.0 * np.cumprod(1.0 + np.full(n, 0.00008))
    return pd.DataFrame(data, index=idx)
