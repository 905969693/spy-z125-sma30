"""Load adjusted closes for SPY, TLT, GLD, DXY, and BIL.

Fresh downloads are written only to a local cache directory. This module
refuses to write into known official research caches or the frozen package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .constants import FORBIDDEN_WRITE_ROOTS, NEEDED, YFINANCE_TICKERS


def strip_tz(index: pd.Index) -> pd.DatetimeIndex:
    """Calendar dates with no timezone. Aware indexes keep the local clock time."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def prepare_closes(closes: pd.DataFrame) -> pd.DataFrame:
    """Numeric SPY, TLT, GLD, DXY, and BIL closes, sorted, with duplicates dropped."""
    missing = [name for name in NEEDED if name not in closes.columns]
    if missing:
        raise ValueError("closes is missing columns: " + ", ".join(missing))
    out = closes.loc[:, list(NEEDED)].apply(pd.to_numeric, errors="coerce")
    out.index = strip_tz(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    bad = out.notna() & (out <= 0)
    if bool(bad.to_numpy().any()):
        names = [name for name in out.columns if bool(bad[name].any())]
        raise ValueError("prices must be strictly positive; non-positive close in " + ", ".join(names))
    if out.empty:
        raise ValueError("price frame is empty")
    return out


def assert_cache_writable(path: Path | str) -> Path:
    """Reject official research caches and the frozen package tree."""
    resolved = Path(path).resolve()
    for raw in FORBIDDEN_WRITE_ROOTS:
        forbidden = Path(raw).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise RuntimeError(f"refusing to write into an official research cache: {resolved}")
    if "spy_z125_sma30" in resolved.parts:
        raise RuntimeError(f"refusing to write into the frozen package: {resolved}")
    return resolved


def _date_col(raw: pd.DataFrame) -> str:
    for col in raw.columns:
        if str(col).lower() in {"date", "datetime"}:
            return str(col)
    return str(raw.columns[0])


def load_close_csv(path: Path | str, name: str) -> pd.Series:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{path} has no rows")
    dcol = _date_col(raw)
    idx = strip_tz(raw[dcol])
    if "Close" in raw.columns:
        values = raw["Close"]
    else:
        values = raw.drop(columns=[dcol]).iloc[:, 0]
    series = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=idx, name=name)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _cache_paths(cache_dir: Path | str) -> dict[str, Path]:
    root = Path(cache_dir)
    found: dict[str, Path] = {}
    for name in NEEDED:
        for candidate in (root / f"{name}_max.csv", root / f"{name}.csv"):
            if candidate.is_file():
                found[name] = candidate
                break
    return found


def download_yfinance(write_cache_dir: Path | str | None = None) -> pd.DataFrame:
    """Download split- and dividend-adjusted closes (yfinance auto_adjust=True)."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required to download prices") from exc

    dest: Path | None = None
    if write_cache_dir is not None:
        dest = assert_cache_writable(write_cache_dir)
        dest.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.Series] = {}
    for name, ticker in YFINANCE_TICKERS.items():
        hist = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if hist is None or len(hist) == 0:
            raise RuntimeError(f"yfinance returned no data for {ticker} ({name})")
        if isinstance(hist.columns, pd.MultiIndex):
            close = hist["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            if "Close" not in hist.columns:
                raise RuntimeError(f"yfinance frame for {ticker} ({name}) has no Close column")
            close = hist["Close"]
        close = pd.to_numeric(close, errors="coerce")
        close.index = strip_tz(close.index)
        close = close[~close.index.duplicated(keep="last")].sort_index().rename(name)
        frames[name] = close
        if dest is not None:
            out_path = dest / f"{name}_max.csv"
            assert_cache_writable(out_path)
            close.rename("Close").to_frame().to_csv(out_path, index_label="Date")
    return prepare_closes(pd.concat(frames, axis=1, sort=False))


def load_prices(
    cache_dir: Path | str | None = None,
    cache_map: Mapping[str, Path | str] | None = None,
    use_yfinance: bool = False,
    write_cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Adjusted closes with columns SPY, TLT, GLD, DXY, and BIL.

    ``use_yfinance`` downloads a fresh sample. Otherwise CSVs are read from
    ``cache_map`` or from ``cache_dir`` (``{NAME}_max.csv`` or ``{NAME}.csv``).
    """
    if use_yfinance:
        return download_yfinance(write_cache_dir=write_cache_dir)
    if cache_map is not None:
        paths = {name: Path(path) for name, path in cache_map.items()}
    elif cache_dir is not None:
        paths = _cache_paths(cache_dir)
    else:
        raise FileNotFoundError(
            "pass a cache directory of SPY/TLT/GLD/DXY/BIL CSV files, or download with yfinance"
        )
    missing = [name for name in NEEDED if name not in paths]
    if missing:
        raise FileNotFoundError(
            "missing price files for "
            + ", ".join(missing)
            + ". Expected {NAME}_max.csv or {NAME}.csv in the cache directory."
        )
    series = [load_close_csv(paths[name], name) for name in NEEDED]
    return prepare_closes(pd.concat(series, axis=1, sort=False))
