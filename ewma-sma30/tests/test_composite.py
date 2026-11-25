"""Causality and formula tests for the EWMA composite. Synthetic prices only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ewma_sma30.composite import (
    basket_prices,
    composite_s,
    compute_composite,
    ewma_cov_matrices,
    log_returns,
    pair_covariance_frame,
    rolling_percentile,
    signed_zscore,
)
from ewma_sma30.constants import (
    BURNIN,
    DROP_TRIGGER,
    LAMBDA,
    PCTL_MIN_PERIODS,
    PCTL_WINDOW,
    PCT_RISK_ON,
    RULE2_EXIT_PCT,
    SD_FLOOR,
    SMA_WINDOW,
    Z_MIN_PERIODS,
    Z_WINDOW,
)
from ewma_sma30.rule import compute_rule

from synthetic import make_closes


def test_live_parameters_are_the_adopted_rule():
    assert LAMBDA == 0.94
    assert BURNIN == 60
    assert Z_WINDOW == 504
    assert Z_MIN_PERIODS == 252
    assert PCTL_WINDOW == 252
    assert PCTL_MIN_PERIODS == 126
    assert SD_FLOOR == 1e-12
    assert SMA_WINDOW == 30
    assert PCT_RISK_ON == 0.9
    assert DROP_TRIGGER == -0.70
    assert RULE2_EXIT_PCT == 0.25
    assert Z_MIN_PERIODS == Z_WINDOW // 2
    assert PCTL_MIN_PERIODS == PCTL_WINDOW // 2


def test_ewma_recursion_uses_live_lambda():
    idx = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.DataFrame(
        {
            "SPY": [0.01, -0.02, 0.005],
            "TLT": [0.001, 0.004, -0.003],
            "GLD": [0.002, 0.0, 0.001],
            "DXY": [-0.004, 0.003, 0.002],
        },
        index=idx,
    )
    cov = ewma_cov_matrices(returns)
    first = returns.iloc[0].to_numpy()
    second = returns.iloc[1].to_numpy()
    assert cov.shape == (3, 4, 4)
    assert np.allclose(cov[0], np.outer(first, first))
    assert np.allclose(cov[1], LAMBDA * cov[0] + (1.0 - LAMBDA) * np.outer(second, second))


def test_burnin_discards_rows_without_resetting_the_recursion():
    idx = pd.bdate_range("2020-01-01", periods=8)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, (8, 4)), index=idx, columns=["SPY", "TLT", "GLD", "DXY"])
    cov = ewma_cov_matrices(returns)
    kept = pair_covariance_frame(returns, cov, burnin=3)
    assert len(kept) == 5
    assert kept.index[0] == returns.index[3]
    assert kept["cov_spy_tlt"].iloc[0] == pytest.approx(cov[3, 0, 1])
    with pytest.raises(ValueError):
        pair_covariance_frame(returns, cov, burnin=len(returns))


def test_signed_z_uses_only_the_prior_window():
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(size=80))
    window = 20
    z = signed_zscore(series, window=window)
    t = 50
    prior = series.iloc[t - window : t]
    expected = (series.iloc[t] - prior.mean()) / prior.std(ddof=1)
    assert len(prior) == window
    assert prior.index[-1] == series.index[t - 1]
    assert z.iloc[t] == pytest.approx(expected)

    shocked = series.copy()
    shocked.iloc[t] = series.iloc[t] + 3.0
    z_shocked = signed_zscore(shocked, window=window)
    assert z_shocked.iloc[t - 1] == pytest.approx(z.iloc[t - 1])
    assert z_shocked.iloc[t] != pytest.approx(z.iloc[t])
    assert z_shocked.iloc[t + 1] != pytest.approx(z.iloc[t + 1])

    future = series.copy()
    future.iloc[-1] = series.iloc[-1] + 5.0
    z_future = signed_zscore(future, window=window)
    assert z_future.iloc[:-1].equals(z.iloc[:-1])


def test_z_std_is_floored():
    series = pd.Series([1.0] * 30 + [2.0])
    z = signed_zscore(series, window=10)
    assert z.iloc[-1] == pytest.approx(1.0 / SD_FLOOR)


def test_default_z_min_periods_is_252_and_shifted():
    series = pd.Series(np.arange(260, dtype=float))
    z = signed_zscore(series)
    assert pd.isna(z.iloc[251])
    assert pd.notna(z.iloc[252])


def test_composite_signs():
    z = pd.DataFrame(
        {
            "z_spy_tlt": [1.0, np.nan],
            "z_spy_dxy": [2.0, 2.0],
            "z_spy_gld": [3.0, 3.0],
            "z_tlt_dxy": [4.0, 4.0],
        }
    )
    score = composite_s(z)
    assert score.iloc[0] == pytest.approx(-1.0 - 2.0 + 3.0 + 4.0)
    assert pd.isna(score.iloc[1])


def test_percentile_includes_today_and_not_the_future():
    score = pd.Series(np.arange(300, dtype=float))
    score.iloc[-1] = -1.0
    pct = rolling_percentile(score)
    assert pct.iloc[-1] == pytest.approx(1.0 / PCTL_WINDOW)

    future = score.copy()
    future.iloc[-1] = 10_000.0
    later = rolling_percentile(future)
    assert later.iloc[-2] == pytest.approx(pct.iloc[-2])
    assert later.iloc[-1] != pytest.approx(pct.iloc[-1])


def test_percentile_min_periods_is_126():
    score = pd.Series(np.arange(126, dtype=float))
    pct = rolling_percentile(score)
    assert pd.isna(pct.iloc[124])
    assert pd.notna(pct.iloc[125])


def test_first_published_z_has_a_full_504_day_baseline():
    prices = make_closes(800, seed=3)
    published = compute_composite(prices)
    panel = basket_prices(prices)
    returns = log_returns(panel)
    cov = ewma_cov_matrices(returns)
    pairs = pair_covariance_frame(returns, cov)
    z_full = signed_zscore(pairs["cov_spy_tlt"], window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
    score = composite_s(
        pd.DataFrame(
            {
                name: signed_zscore(pairs[name.replace("z_", "cov_")], window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
                for name in ("z_spy_tlt", "z_spy_dxy", "z_spy_gld", "z_tlt_dxy")
            },
            index=pairs.index,
        )
    )
    pct_full = rolling_percentile(score, window=PCTL_WINDOW, min_periods=PCTL_MIN_PERIODS)

    assert published.index[0] == pairs.index[Z_WINDOW]
    assert len(published) == len(pairs) - Z_WINDOW
    prior = pairs["cov_spy_tlt"].iloc[:Z_WINDOW]
    assert len(prior) == Z_WINDOW
    sd = float(prior.std(ddof=1))
    expected = (float(pairs["cov_spy_tlt"].iloc[Z_WINDOW]) - float(prior.mean())) / max(sd, SD_FLOOR)
    assert published["z_spy_tlt"].iloc[0] == pytest.approx(expected)
    assert published["z_spy_tlt"].iloc[0] == pytest.approx(z_full.iloc[Z_WINDOW])
    assert published["S"].iloc[0] == pytest.approx(score.iloc[Z_WINDOW])
    assert published["pct"].iloc[0] == pytest.approx(pct_full.iloc[Z_WINDOW])
    assert pd.notna(published["pct"].iloc[0])
    recomputed_after_cut = rolling_percentile(published["S"])
    assert pd.isna(recomputed_after_cut.iloc[0])


def test_pipeline_signal_does_not_look_ahead():
    prices = make_closes(800, seed=4)
    base = compute_rule(prices)
    shocked = prices.copy()
    shocked.iloc[-1, shocked.columns.get_loc("SPY")] = prices["SPY"].iloc[-1] * 1.01
    moved = compute_rule(shocked)
    pd.testing.assert_frame_equal(base.iloc[:-1], moved.iloc[:-1])
    assert moved["spy_close"].iloc[-1] != pytest.approx(base["spy_close"].iloc[-1])
    assert moved["spy_ret"].iloc[-1] != pytest.approx(base["spy_ret"].iloc[-1])
