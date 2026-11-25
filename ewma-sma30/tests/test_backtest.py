"""Sample conventions shared with the frozen package. No published performance figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ewma_sma30.backtest import cash_and_switches, performance_stats, run_backtest, slice_returns
from ewma_sma30.constants import ANN_DAYS, NAV_START


def test_slice_excludes_the_start_session():
    idx = pd.to_datetime(["2007-05-31", "2007-06-01", "2007-06-04"])
    returns = pd.Series([0.50, 0.10, -0.02], index=idx)
    sliced = slice_returns(returns, "2007-05-31")
    assert list(sliced.index) == list(idx[1:])
    assert sliced.iloc[0] == pytest.approx(0.10)
    # Missing start date: the first later session becomes NAV=1 and is excluded.
    later = slice_returns(returns.iloc[1:], "2007-05-31")
    assert list(later.index) == [idx[2]]


def test_performance_stats_match_the_sample_formulas():
    idx = pd.bdate_range("2020-01-02", periods=3)
    returns = pd.Series([0.10, -0.05, 0.02], index=idx)
    stats = performance_stats(returns)
    nav = (1.0 + returns).cumprod()
    sd = float(returns.std(ddof=1))
    assert stats["n"] == 3
    assert stats["end_nav"] == pytest.approx(float(nav.iloc[-1]))
    assert stats["cagr"] == pytest.approx(float(nav.iloc[-1]) ** (ANN_DAYS / 3) - 1.0)
    assert stats["vol"] == pytest.approx(sd * np.sqrt(ANN_DAYS))
    assert stats["sharpe"] == pytest.approx(float(returns.mean()) / sd * np.sqrt(ANN_DAYS))
    assert stats["max_dd"] == pytest.approx(float((nav / nav.cummax() - 1.0).min()))
    assert ANN_DAYS == 252


def test_cash_switch_count_includes_the_first_session():
    idx = pd.bdate_range("2020-01-02", periods=5)
    applied = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0], index=idx)
    stats = cash_and_switches(applied)
    assert stats["n_cash"] == 2
    assert stats["pct_cash"] == pytest.approx(0.4)
    assert stats["n_switches"] == 3


def test_run_backtest_uses_nav_start_and_drops_that_days_return():
    idx = pd.bdate_range("2007-05-31", periods=6)
    assert idx[0] == NAV_START
    daily = pd.DataFrame(
        {
            "rule_ret": [0.50, 0.01, 0.02, -0.01, 0.03, 0.00],
            "spy_ret": [0.50, 0.02, 0.02, -0.03, 0.01, 0.04],
            "applied_cash": [np.nan, 0.0, 1.0, 1.0, 0.0, 0.0],
            "signal_cash": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    result = run_backtest(daily)
    assert result["rule"]["start"] == str(idx[1].date())
    assert result["rule"]["end"] == str(idx[-1].date())
    assert result["rule"]["n"] == 5
    assert result["rule"]["end_nav"] == pytest.approx(float((1.0 + daily["rule_ret"].iloc[1:]).prod()))
    assert result["spy"]["end_nav"] == pytest.approx(float((1.0 + daily["spy_ret"].iloc[1:]).prod()))
    assert result["rule"]["n_cash"] == 2
