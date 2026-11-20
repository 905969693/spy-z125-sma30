"""Unit tests for the frozen rule: causality, next-day apply, official sample."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spy_z125_sma30.backtest import official_backtest
from spy_z125_sma30.constants import (
    NAV_START,
    OFFICIAL_RULE_STATS,
    OFFICIAL_SAMPLE_END,
    OFFICIAL_SPY_STATS,
    Z_THRESHOLD,
)
from spy_z125_sma30.data import load_official_prices
from spy_z125_sma30.rule import compute_rule, speed_z
from spy_z125_sma30.signal import latest_signal

OFFICIAL_DAILY = Path("/workspace/cross-asset-corr-4asset-check/gate_z_grid_daily.csv")


def test_official_sample_matches_frozen_document():
    daily = compute_rule(load_official_prices())
    result = official_backtest(daily, end=OFFICIAL_SAMPLE_END, start=NAV_START)
    rule, spy = result["rule"], result["spy"]
    assert rule["start"] == "2007-06-01"
    assert rule["end"] == OFFICIAL_SAMPLE_END
    for key, expected in OFFICIAL_RULE_STATS.items():
        assert rule[key] == pytest.approx(expected, rel=1e-9, abs=1e-10), key
    for key, expected in OFFICIAL_SPY_STATS.items():
        assert spy[key] == pytest.approx(expected, rel=1e-9, abs=1e-10), key


@pytest.mark.skipif(not OFFICIAL_DAILY.is_file(), reason="gate_z_grid_daily.csv not present")
def test_matches_published_daily_series():
    published = (
        pd.read_csv(OFFICIAL_DAILY, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc[: OFFICIAL_SAMPLE_END]
    )
    daily = compute_rule(load_official_prices()).loc[published.index]
    keep = published[["s", "z", "spy", "spy_ma30", "z125_g30_in_cash", "z125_g30_ret"]].rename(
        columns=lambda c: c + "_pub"
    )
    aligned = daily.join(keep, how="inner")
    assert (aligned["s"] - aligned["s_pub"]).abs().max() < 1e-8
    assert (aligned["z"] - aligned["z_pub"]).abs().max() < 1e-8
    assert (aligned["spy"] - aligned["spy_pub"]).abs().max() < 1e-8
    assert (aligned["spy_ma30"] - aligned["spy_ma30_pub"]).abs().max() < 1e-8
    sig = aligned["signal_cash"].astype("boolean")
    pub = aligned["z125_g30_in_cash_pub"].astype("boolean")
    assert (sig.fillna(False) == pub.fillna(False)).all()
    assert sig.isna().equals(pub.isna())
    ret_diff = (aligned["rule_ret"] - aligned["z125_g30_ret_pub"]).abs()
    assert ret_diff.max() < 1e-10


def test_z_is_causal():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=300)
    s = pd.Series(rng.normal(size=len(idx)).cumsum() * 0.01, index=idx)
    z = speed_z(s)
    # Last Z must be unchanged if a future s value is altered after it is computed.
    last_valid = z.dropna().index[-2]
    z_last = float(z.loc[last_valid])
    s2 = s.copy()
    s2.iloc[-1] = s2.iloc[-1] + 10.0
    z2 = speed_z(s2)
    assert z2.loc[last_valid] == pytest.approx(z_last)
    assert z2.iloc[-1] != pytest.approx(z.iloc[-1])


def test_cash_applies_next_day():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(1)
    # Build a flat basket so s is defined, then force a known cash signal day.
    px = pd.DataFrame(
        {
            "SPY": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
            "TLT": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx)))),
            "GLD": 100 * np.exp(np.cumsum(rng.normal(0, 0.007, len(idx)))),
            "DXY": 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))),
            "BIL": 100 * np.cumprod(1 + np.full(len(idx), 0.00008)),
        },
        index=idx,
    )
    daily = compute_rule(px)
    usable = daily.loc[daily["signal_cash"].notna()]
    assert not usable.empty
    # applied_cash is the previous close signal
    assert daily["applied_cash"].equals(daily["signal_cash"].shift(1))
    # On a day the close signal first turns cash, that day's rule return is still SPY
    # (or already-applied cash), never the same-day flip.
    turned = (daily["signal_cash"] == True) & (daily["signal_cash"].shift(1) != True)
    turned = turned.fillna(False)
    if turned.any():
        first = daily.index[turned][0]
        if pd.notna(daily.loc[first, "applied_cash"]) and not bool(daily.loc[first, "applied_cash"]):
            assert daily.loc[first, "rule_ret"] == pytest.approx(daily.loc[first, "spy_ret"])


def test_equal_to_sma_is_not_below():
    idx = pd.bdate_range("2020-01-01", periods=40)
    spy = pd.Series(100.0, index=idx)
    spy.iloc[-1] = 100.0  # equal to the 30-day mean of 100
    px = pd.DataFrame({"SPY": spy, "TLT": spy, "GLD": spy, "DXY": spy, "BIL": spy + 0.01})
    daily = compute_rule(px)
    last = daily.iloc[-1]
    assert last["spy"] == pytest.approx(last["spy_ma30"])
    assert bool(last["below_ma30"]) is False


def test_latest_signal_keys():
    idx = pd.bdate_range("2018-01-01", periods=400)
    rng = np.random.default_rng(2)
    px = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for c in ["SPY", "TLT", "GLD", "DXY"]},
        index=idx,
    )
    px["BIL"] = 100 * np.cumprod(1 + np.full(len(idx), 0.00008))
    sig = latest_signal(compute_rule(px))
    assert sig["close_signal"] in {"cash", "invested"}
    assert sig["next_session"] == sig["close_signal"]
    assert "z" in sig and "s" in sig
    if sig["signal_cash"]:
        assert sig["z"] > Z_THRESHOLD
        assert sig["below_ma30"] is True
