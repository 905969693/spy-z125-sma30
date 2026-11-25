"""Rule 1, rule 2, the latch, the OR, and next-day execution. Synthetic only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ewma_sma30.rule import (
    apply_latch,
    apply_next_day,
    build_signals,
    combine_signals,
    compute_rule,
    own_calendar_simple_return,
    rule1_risk_off,
    spy_sma,
)

from synthetic import make_closes


def _signals(pct: list[float], close: list[float], sma: float | list[float] = 100.0) -> pd.DataFrame:
    index = pd.RangeIndex(len(pct))
    average = sma if isinstance(sma, list) else [sma] * len(pct)
    return build_signals(
        pd.Series(pct, index=index, dtype=float),
        pd.Series(close, index=index, dtype=float),
        pd.Series(average, index=index, dtype=float),
    )


def test_rule1_risk_off_is_strictly_above_0_9_and_latch_survives_a_rebound():
    out = _signals(
        pct=[0.50, 0.95, 0.95, 0.96, 0.91, 0.90, 0.95, 0.95],
        close=[101, 101, 101, 99, 101, 99, 101, 99],
    )
    assert out["risk_off_r1"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1]
    assert out["rule1_sig"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert out["latched"].tolist() == out["rule1_sig"].tolist()
    # Day 4 is back above the average and rule 1 stays in cash.
    assert out.loc[4, "below_ma"] == 0
    assert out.loc[4, "rule1_sig"] == 1
    # 0.90 is not above 0.90, so the episode ends even though price is below.
    assert out.loc[5, "risk_off_r1"] == 0
    assert out.loc[5, "rule1_sig"] == 0
    # A new episode does not inherit the old latch.
    assert out.loc[6, "rule1_sig"] == 0
    assert out.loc[7, "rule1_sig"] == 1


def test_latch_ignores_a_future_breach_and_holds_through_a_missing_day():
    risk = pd.Series([1.0] * 6)
    below = pd.Series([0.0] * 6)
    base = apply_latch(risk, below)["signal_cash"]
    future = below.copy()
    future.iloc[-1] = 1.0
    later = apply_latch(risk, future)["signal_cash"]
    assert base.iloc[-2] == 0
    assert later.iloc[-2] == 0
    assert later.iloc[-1] == 1

    paused = pd.Series([0.95, np.nan, 0.95])
    paused_risk = rule1_risk_off(paused)
    paused_below = pd.Series([1.0, 1.0, 0.0])
    held = apply_latch(paused_risk, paused_below)
    assert held["signal_cash"].iloc[0] == 1
    assert pd.isna(held["signal_cash"].iloc[1])
    assert held["signal_cash"].iloc[2] == 1


def test_equal_close_is_neither_below_nor_above():
    out = _signals(pct=[0.2, 0.4, 0.6, 0.8, 0.1], close=[100, 100, 100, 100, 100])
    assert (out["below_ma"] == 0).all()
    assert out["spy_close"].iloc[-1] == pytest.approx(out["spy_sma30"].iloc[-1])
    # drop5 is -0.7 and would enter if the close were strictly below.
    assert out["drop5"].iloc[-1] == pytest.approx(-0.7)
    assert out["rule2_sig"].iloc[-1] == 0


def test_rule2_entry_threshold_includes_today_and_is_inclusive():
    entered = _signals(pct=[1, 1, 1, 1, 0.3], close=[99, 99, 99, 99, 99])
    assert entered["high5"].iloc[-1] == pytest.approx(1.0)
    assert entered["drop5"].iloc[-1] == pytest.approx(-0.70)
    assert entered["rule2_sig"].iloc[-1] == 1

    just_short = _signals(pct=[1, 1, 1, 1, 0.31], close=[99, 99, 99, 99, 99])
    assert just_short["drop5"].iloc[-1] > -0.70
    assert just_short["rule2_sig"].iloc[-1] == 0

    not_below = _signals(pct=[1, 1, 1, 1, 0.3], close=[99, 99, 99, 99, 101])
    assert not_below["rule2_sig"].iloc[-1] == 0

    today_is_the_high = _signals(pct=[0.1, 0.2, 0.3, 0.4, 0.5], close=[99] * 5)
    assert today_is_the_high["drop5"].iloc[-1] == pytest.approx(0.0)
    assert today_is_the_high["rule2_sig"].iloc[-1] == 0
    assert today_is_the_high["rule2_sig"].iloc[:4].isna().all()


def test_rule2_checks_the_exit_before_same_day_reentry():
    # Upward cross through 0.25 and the drop is still deep enough, so the
    # close signal stays cash after the exit-then-reentry pass.
    reenter = _signals(pct=[1, 1, 1, 1, 0.2, 0.25], close=[99] * 6)
    assert reenter["rule2_sig"].iloc[4] == 1
    assert reenter["drop5"].iloc[5] <= -0.70
    assert reenter["rule2_sig"].iloc[5] == 1

    # Same cross, but the drop is no longer deep enough: the exit sticks.
    leave = _signals(pct=[0.95, 0.96, 0.97, 0.98, 0.24, 0.40], close=[99] * 6)
    assert leave["rule2_sig"].iloc[4] == 1
    assert leave["drop5"].iloc[5] > -0.70
    assert leave["rule2_sig"].iloc[5] == 0

    # Previous percentile already at 0.25 is not an upward cross.
    no_cross = _signals(pct=[0.95, 0.96, 0.97, 0.98, 0.25, 0.40], close=[99] * 6)
    assert no_cross["rule2_sig"].iloc[4] == 1
    assert no_cross["rule2_sig"].iloc[5] == 1

    # Strictly above SMA30 exits. Equality does not.
    above = _signals(pct=[1, 1, 1, 1, 0.2, 0.2], close=[99, 99, 99, 99, 99, 101])
    assert above["rule2_sig"].iloc[4] == 1
    assert above["rule2_sig"].iloc[5] == 0
    equal = _signals(pct=[1, 1, 1, 1, 0.2, 0.2], close=[99, 99, 99, 99, 99, 100])
    assert equal["rule2_sig"].iloc[5] == 1


def test_missing_percentile_does_not_reset_rule2():
    pct = [0.80, 0.90, 1.00, 0.95, 0.20, np.nan, 0.50, 0.50, 0.50, 0.50, 0.50]
    out = _signals(pct=pct, close=[99] * len(pct))
    assert out["rule2_sig"].iloc[4] == 1
    assert pd.isna(out["rule2_sig"].iloc[5])
    assert pd.isna(out["drop5"].iloc[9])
    assert out["rule2_sig"].iloc[10] == 1
    assert pd.isna(out["signal_cash"].iloc[5])


def test_combined_signal_is_the_or_and_blocks_on_a_missing_percentile():
    index = pd.RangeIndex(5)
    pct = pd.Series([0.5, 0.5, np.nan, 0.95, 0.2], index=index)
    rule1 = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=index)
    rule2 = pd.Series([0.0, 0.0, 1.0, 1.0, np.nan], index=index)
    signal = combine_signals(rule1, rule2, pct)
    assert signal.tolist()[:2] == [0.0, 1.0]
    assert pd.isna(signal.iloc[2])
    assert signal.iloc[3] == 1.0
    assert signal.iloc[4] == 0.0

    # Integrated path: rule 1 alone, then rule 2 alone, then an SMA exit.
    out = _signals(
        pct=[0.40, 0.50, 0.60, 0.80, 0.95, 0.96, 0.20, 0.22, 0.10],
        close=[101, 101, 101, 101, 101, 99, 99, 99, 101],
    )
    assert out.loc[4, "risk_off_r1"] == 1
    assert out.loc[4, "rule1_sig"] == 0
    assert out.loc[4, "signal_cash"] == 0
    assert out.loc[5, "rule1_sig"] == 1
    assert out.loc[5, "rule2_sig"] == 0
    assert out.loc[5, "signal_cash"] == 1
    assert out.loc[6, "rule1_sig"] == 0
    assert out.loc[6, "rule2_sig"] == 1
    assert out.loc[6, "signal_cash"] == 1
    assert out.loc[7, "signal_cash"] == 1
    assert out.loc[8, "below_ma"] == 0
    assert out.loc[8, "rule2_sig"] == 0
    assert out.loc[8, "signal_cash"] == 0


def test_position_is_yesterdays_signal_and_uses_bil_or_spy():
    index = pd.bdate_range("2024-01-02", periods=4)
    signal = pd.Series([0.0, 1.0, 1.0, 0.0], index=index, name="signal_cash")
    spy = pd.Series([0.01, -0.02, 0.03, 0.04], index=index)
    bil = pd.Series([0.001, 0.001, 0.001, 0.001], index=index)
    position = apply_next_day(signal, spy, bil)
    assert position["applied_cash"].tolist()[1:] == [0.0, 1.0, 1.0]
    assert pd.isna(position["applied_cash"].iloc[0])
    assert pd.isna(position["rule_ret"].iloc[0])
    # The day the close signal turns cash still earns SPY.
    assert position["rule_ret"].iloc[1] == pytest.approx(spy.iloc[1])
    assert position["rule_ret"].iloc[2] == pytest.approx(bil.iloc[2])
    assert position["rule_ret"].iloc[3] == pytest.approx(bil.iloc[3])


def test_own_calendar_return_and_sma_keep_a_spy_only_session():
    prices = make_closes(900, seed=5)
    friday = prices.index[prices.index.day_of_week == 4]
    friday = friday[friday > prices.index[700]][0]
    saturday = friday + pd.Timedelta(days=1)
    monday = friday + pd.Timedelta(days=3)
    assert saturday not in prices.index
    assert monday in prices.index

    closes = prices.copy()
    extra = pd.DataFrame(
        {
            "SPY": [float(prices.loc[friday, "SPY"]) * 1.5],
            "TLT": [np.nan],
            "GLD": [np.nan],
            "DXY": [np.nan],
            "BIL": [np.nan],
        },
        index=pd.DatetimeIndex([saturday]),
    )
    closes = pd.concat([closes, extra]).sort_index()
    daily = compute_rule(closes)

    assert saturday not in daily.index
    assert monday in daily.index
    spy = closes["SPY"].dropna().sort_index()
    expected_ret = spy.loc[monday] / spy.loc[saturday] - 1.0
    spanned = spy.loc[monday] / spy.loc[friday] - 1.0
    assert daily.loc[monday, "spy_ret"] == pytest.approx(expected_ret)
    assert daily.loc[monday, "spy_ret"] != pytest.approx(spanned)
    assert daily.loc[monday, "spy_ret"] == pytest.approx(
        own_calendar_simple_return(closes["SPY"], daily.index).loc[monday]
    )

    bil = closes["BIL"].dropna().sort_index()
    assert daily.loc[monday, "bil_ret"] == pytest.approx(bil.loc[monday] / bil.loc[friday] - 1.0)

    sma_own = spy_sma(closes["SPY"]).loc[monday]
    sma_common = prices["SPY"].rolling(30, min_periods=30).mean().loc[monday]
    assert daily.loc[monday, "spy_sma30"] == pytest.approx(sma_own)
    assert daily.loc[monday, "spy_sma30"] != pytest.approx(sma_common)
    assert bool(daily.loc[monday, "below_ma"] == 1) is bool(spy.loc[monday] < sma_own)


def test_compute_rule_identities_on_a_long_synthetic_sample():
    daily = compute_rule(make_closes(900, seed=6))
    assert daily["applied_cash"].equals(daily["signal_cash"].shift(1))
    combined = (
        (daily["rule1_sig"].fillna(0.0) == 1.0) | (daily["rule2_sig"].fillna(0.0) == 1.0)
    ).astype(float)
    assert daily["signal_cash"].equals(combined.where(daily["pct"].notna()))
    cash = daily["applied_cash"] == 1.0
    invested = daily["applied_cash"] == 0.0
    assert np.allclose(daily.loc[cash, "rule_ret"], daily.loc[cash, "bil_ret"])
    assert np.allclose(daily.loc[invested, "rule_ret"], daily.loc[invested, "spy_ret"])
    assert daily.loc[daily["applied_cash"].isna(), "rule_ret"].isna().all()
    assert daily["pct"].notna().any()
    assert len(daily) > 100
