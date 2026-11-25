"""CLI, session clock, and cache-write guard. Synthetic files only."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ewma_sma30.cli import main
from ewma_sma30.constants import NEEDED
from ewma_sma30.data import assert_cache_writable, load_prices
from ewma_sma30.rule import compute_rule
from ewma_sma30.signal import (
    closes_through_last_session,
    last_allowed_session_date,
    latest_complete_signal,
    latest_signal,
)

from synthetic import make_closes

_NY = ZoneInfo("America/New_York")


def test_session_clock_waits_until_1615_new_york():
    assert last_allowed_session_date(datetime(2026, 11, 25, 10, 0, tzinfo=_NY)) == pd.Timestamp("2026-11-24")
    assert last_allowed_session_date(datetime(2026, 11, 25, 16, 14, tzinfo=_NY)) == pd.Timestamp("2026-11-24")
    assert last_allowed_session_date(datetime(2026, 11, 25, 16, 15, tzinfo=_NY)) == pd.Timestamp("2026-11-25")
    assert last_allowed_session_date(datetime(2026, 11, 28, 12, 0, tzinfo=_NY)) == pd.Timestamp("2026-11-27")
    assert last_allowed_session_date(datetime(2026, 11, 23, 9, 0, tzinfo=_NY)) == pd.Timestamp("2026-11-20")
    # A naive clock is read as New York, not UTC.
    assert last_allowed_session_date(datetime(2026, 11, 25, 10, 0)) == pd.Timestamp("2026-11-24")


def test_trim_drops_an_incomplete_session():
    idx = pd.to_datetime(["2026-11-24", "2026-11-25"])
    closes = pd.DataFrame({"SPY": [1.0, 2.0]}, index=idx)
    morning = closes_through_last_session(closes, now=datetime(2026, 11, 25, 10, 0))
    assert list(morning.index) == [pd.Timestamp("2026-11-24")]
    after_close = closes_through_last_session(closes, now=datetime(2026, 11, 25, 16, 15))
    assert len(after_close) == 2


def test_refuses_to_write_official_caches_or_the_frozen_package():
    with pytest.raises(RuntimeError):
        assert_cache_writable(Path("/workspace/dual-momentum/cache"))
    with pytest.raises(RuntimeError):
        assert_cache_writable(Path("/workspace/spy-z125-sma30/cache/nested"))
    frozen = Path(__file__).resolve().parents[2] / "src" / "spy_z125_sma30" / "fixtures"
    with pytest.raises(RuntimeError):
        assert_cache_writable(frozen)
    assert assert_cache_writable(Path("/tmp/ewma-sma30-cache-ok")).name == "ewma-sma30-cache-ok"


def test_package_source_does_not_import_the_frozen_package():
    src = Path(__file__).resolve().parents[1] / "src"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import spy_z125_sma30" not in text
        assert "from spy_z125_sma30" not in text
        assert "import yfinance" not in text or path.name == "data.py"


def _write_cache(closes: pd.DataFrame, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        closes[name].rename("Close").to_frame().to_csv(cache / f"{name}_max.csv", index_label="Date")


def test_cache_roundtrip_and_duplicate_dates_keep_the_last_print(tmp_path: Path):
    closes = make_closes(30, seed=7, start="2020-01-01")
    _write_cache(closes, tmp_path)
    loaded = load_prices(cache_dir=tmp_path)
    assert list(loaded.columns) == list(NEEDED)
    assert loaded["SPY"].iloc[-1] == pytest.approx(closes["SPY"].iloc[-1])

    path = tmp_path / "SPY_max.csv"
    raw = pd.read_csv(path)
    raw = pd.concat([raw, raw.iloc[[-1]].assign(Close=123.45)], ignore_index=True)
    raw.to_csv(path, index=False)
    reloaded = load_prices(cache_dir=tmp_path)
    assert reloaded["SPY"].iloc[-1] == pytest.approx(123.45)


def test_cli_backtest_and_signal_on_synthetic_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    cache = tmp_path / "cache"
    _write_cache(make_closes(1200, seed=8), cache)
    assert main(["--cache-dir", str(cache), "backtest"]) == 0
    backtest_out = capsys.readouterr().out
    assert "未冻结" in backtest_out
    assert "Sharpe" in backtest_out
    assert "EWMA 规则" in backtest_out

    assert main(["--cache-dir", str(cache), "signal"]) == 0
    signal_out = capsys.readouterr().out
    assert "次日生效仓位" in signal_out
    assert "收盘日期" in signal_out


def test_cli_errors_without_a_cache_and_when_history_is_too_short(tmp_path: Path):
    assert main(["signal"]) == 1
    cache = tmp_path / "short"
    _write_cache(make_closes(30, seed=9), cache)
    assert main(["--cache-dir", str(cache), "backtest"]) == 1


def test_complete_signal_refuses_a_spy_session_missing_from_the_basket():
    closes = make_closes(900, seed=10)
    daily = compute_rule(closes)
    payload = latest_complete_signal(closes, daily)
    assert payload["asof"] == str(closes.index[-1].date())
    assert payload["next_session"] == payload["close_signal"]
    assert payload["params"]["lambda"] == 0.94
    assert payload["params"]["drop"] == -0.70
    assert payload["params"]["sma"] == 30
    assert latest_signal(daily)["asof"] == payload["asof"]

    broken = closes.copy()
    broken.iloc[-1, broken.columns.get_loc("DXY")] = np.nan
    with pytest.raises(RuntimeError, match="four-asset"):
        latest_complete_signal(broken, daily)
