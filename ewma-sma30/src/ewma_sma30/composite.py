"""EWMA signed-covariance composite S and its rolling percentile.

The trading rule lives in ``rule.py``. This module does not apply a
hysteresis band and does not trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    BASKET,
    BURNIN,
    LAMBDA,
    PAIRS,
    PAIR_SIGN,
    PCTL_MIN_PERIODS,
    PCTL_WINDOW,
    SD_FLOOR,
    Z_MIN_PERIODS,
    Z_WINDOW,
)
from .data import strip_tz


def _pair_name(left: str, right: str) -> str:
    return f"{left.lower()}_{right.lower()}"


def basket_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Inner join of SPY, TLT, GLD, and DXY. Missing sessions are dropped, not filled."""
    missing = [name for name in BASKET if name not in prices.columns]
    if missing:
        raise ValueError("prices missing columns: " + ", ".join(missing))
    px = prices.loc[:, list(BASKET)].apply(pd.to_numeric, errors="coerce")
    px.index = strip_tz(px.index)
    px = px[~px.index.duplicated(keep="last")].sort_index().dropna(how="any")
    if px.empty:
        raise ValueError("four-asset inner join is empty")
    if bool((px <= 0).to_numpy().any()):
        raise ValueError("log returns require strictly positive prices")
    return px


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns on the already inner-joined panel. The first session is dropped."""
    returns = np.log(prices.astype(float)).diff().dropna(how="any")
    returns.index.name = prices.index.name
    return returns


def ewma_cov_matrices(returns: pd.DataFrame) -> np.ndarray:
    """RiskMetrics recursion: Sigma_0 = r_0 r_0', then Sigma_t = λ Sigma_{t-1} + (1-λ) r_t r_t'.

    Lambda is the live constant 0.94. Returns are not demeaned.
    """
    values = returns.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("EWMA covariance needs at least one return row")
    n_obs, n_assets = values.shape
    cov = np.empty((n_obs, n_assets, n_assets), dtype=np.float64)
    cov[0] = np.outer(values[0], values[0])
    retain = float(LAMBDA)
    shock = 1.0 - retain
    for i in range(1, n_obs):
        row = values[i]
        cov[i] = retain * cov[i - 1] + shock * np.outer(row, row)
    return cov


def pair_covariance_frame(
    returns: pd.DataFrame,
    cov: np.ndarray,
    burnin: int = BURNIN,
) -> pd.DataFrame:
    """Off-diagonal pair covariances after the burn-in rows are discarded.

    Discarding those rows does not reset the recursion. The first kept matrix
    already includes the burned-in shocks.
    """
    if burnin < 0 or burnin >= len(returns):
        raise ValueError("burn-in must be shorter than the return sample")
    columns = {name: i for i, name in enumerate(returns.columns)}
    data: dict[str, np.ndarray] = {}
    for left, right in PAIRS:
        data[f"cov_{_pair_name(left, right)}"] = cov[:, columns[left], columns[right]]
    frame = pd.DataFrame(data, index=returns.index)
    return frame.iloc[burnin:]


def signed_zscore(
    series: pd.Series,
    window: int = Z_WINDOW,
    min_periods: int | None = None,
) -> pd.Series:
    """Causal z-score. Mean and std use only observations before t; std is floored."""
    if min_periods is None:
        min_periods = window // 2
    if window <= 1 or min_periods <= 1 or min_periods > window:
        raise ValueError("z-score window and min_periods are inconsistent")
    prior_mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    prior_std = series.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    prior_std = prior_std.clip(lower=SD_FLOOR)
    return (series - prior_mean) / prior_std


def composite_s(z: pd.DataFrame) -> pd.Series:
    """S = -z(SPY,TLT) - z(SPY,DXY) + z(SPY,GLD) + z(TLT,DXY)."""
    score = pd.Series(0.0, index=z.index, dtype=float)
    for left, right in PAIRS:
        column = f"z_{_pair_name(left, right)}"
        score = score + PAIR_SIGN[(left, right)] * z[column]
    score.name = "S"
    return score


def rolling_percentile(
    score: pd.Series,
    window: int = PCTL_WINDOW,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling percentile of S. Includes S_t and excludes every later value.

    Ties use the average rank (pandas ``rank(pct=True)``).
    """
    if min_periods is None:
        min_periods = window // 2
    if window <= 1 or min_periods <= 1 or min_periods > window:
        raise ValueError("percentile window and min_periods are inconsistent")
    pct = score.rolling(window, min_periods=min_periods).rank(method="average", pct=True)
    pct.name = "pct"
    return pct


def compute_composite(prices: pd.DataFrame) -> pd.DataFrame:
    """Published S, percentile, and pair z-scores.

    The percentile is computed on the full post-burn-in score, then the frame
    is cut at ``iloc[Z_WINDOW]``. Every published z therefore has a full
    504-day causal baseline, and the first published percentile still sees
    the earlier scores.
    """
    px = basket_prices(prices)
    returns = log_returns(px)
    if returns.empty:
        raise ValueError("no four-asset log returns")
    cov = ewma_cov_matrices(returns)
    pairs = pair_covariance_frame(returns, cov, burnin=BURNIN)

    z = pd.DataFrame(index=pairs.index)
    for left, right in PAIRS:
        key = _pair_name(left, right)
        z[f"z_{key}"] = signed_zscore(
            pairs[f"cov_{key}"],
            window=Z_WINDOW,
            min_periods=Z_MIN_PERIODS,
        )
    score = composite_s(z)
    pct = rolling_percentile(score, window=PCTL_WINDOW, min_periods=PCTL_MIN_PERIODS)
    post = pd.concat([z, score, pct], axis=1)
    if len(post) <= Z_WINDOW:
        raise ValueError(
            "not enough common history for a published EWMA signal: "
            f"need more than {BURNIN + Z_WINDOW} four-asset log-return days "
            "(60-day covariance burn-in, then a 504-day z baseline); "
            f"got {len(returns)}"
        )
    published = post.iloc[Z_WINDOW:].copy()
    published.index.name = "date"
    return published
