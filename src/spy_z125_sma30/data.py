"""Load adjusted closes for SPY, TLT, GLD, DXY, and BIL.

Official read-only caches are used when present. Fresh downloads are written
only to a local cache directory, never over those official files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .constants import BASKET, OFFICIAL_CACHE_PATHS, YFINANCE_TICKERS

NEEDED = (*BASKET, "BIL")


def _date_col(raw: pd.DataFrame) -> str:
    for c in raw.columns:
        if str(c).lower() in {"date", "datetime"}:
            return c
    return raw.columns[0]


def load_close_csv(path: Path | str, name: str) -> pd.Series:
    raw = pd.read_csv(path)
    dcol = _date_col(raw)
    raw[dcol] = pd.to_datetime(raw[dcol]).dt.tz_localize(None)
    if "Close" in raw.columns:
        close = raw.set_index(dcol).sort_index()["Close"]
    else:
        close = raw.set_index(dcol).sort_index().iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").rename(name)


def load_official_prices() -> pd.DataFrame:
    """Frozen adjusted closes used for the official 2007-05-31 to 2026-11-13 sample.

    Live yfinance or later caches can restate adjusted history; use this snapshot
    to reproduce the documented official-sample statistics.
    """
    try:
        from importlib.resources import files
        ref = files("spy_z125_sma30") / "fixtures" / "official_prices.csv"
        with ref.open("r", encoding="utf-8") as fh:
            raw = pd.read_csv(fh)
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        path = Path(__file__).resolve().parent / "fixtures" / "official_prices.csv"
        raw = pd.read_csv(path)
    dcol = _date_col(raw)
    raw[dcol] = pd.to_datetime(raw[dcol]).dt.tz_localize(None)
    return raw.set_index(dcol).sort_index()


def default_cache_map(cache_dir: Path | str | None = None) -> dict[str, Path]:
    """Prefer ``cache_dir`` files, then official research caches if they exist."""
    out: dict[str, Path] = {}
    local = Path(cache_dir) if cache_dir is not None else None
    for name in NEEDED:
        candidates: list[Path] = []
        if local is not None:
            candidates.append(local / f"{name}_max.csv")
            candidates.append(local / f"{name}.csv")
        official = Path(OFFICIAL_CACHE_PATHS[name])
        candidates.append(official)
        found = next((p for p in candidates if p.is_file()), None)
        if found is not None:
            out[name] = found
    return out


def download_yfinance(write_cache_dir: Path | str | None = None) -> pd.DataFrame:
    """Download split- and dividend-adjusted closes (yfinance auto_adjust=True)."""
    import yfinance as yf

    dest = Path(write_cache_dir) if write_cache_dir is not None else None
    if dest is not None:
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
        if hist is None or hist.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker} ({name})")
        if isinstance(hist.columns, pd.MultiIndex):
            close = hist["Close"].iloc[:, 0] if hist["Close"].ndim == 2 else hist["Close"]
        else:
            close = hist["Close"]
        close = pd.to_numeric(close, errors="coerce")
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index().rename(name)
        frames[name] = close
        if dest is not None:
            official = Path(OFFICIAL_CACHE_PATHS[name]).resolve()
            out_path = (dest / f"{name}_max.csv").resolve()
            if out_path == official:
                raise RuntimeError(f"refusing to overwrite official cache: {official}")
            close.rename("Close").to_frame().to_csv(out_path, index_label="Date")
    return pd.concat(frames, axis=1, sort=False).sort_index()


def load_prices(
    cache_dir: Path | str | None = None,
    cache_map: Mapping[str, Path | str] | None = None,
    use_yfinance: bool = False,
    write_cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted closes with columns SPY, TLT, GLD, DXY, BIL.

    If ``use_yfinance`` is true, download fresh series and optionally write them
    to ``write_cache_dir``. Otherwise load CSVs from ``cache_map`` or
    ``default_cache_map(cache_dir)``.
    """
    if use_yfinance:
        return download_yfinance(write_cache_dir=write_cache_dir)

    paths = dict(cache_map) if cache_map is not None else default_cache_map(cache_dir)
    missing = [name for name in NEEDED if name not in paths]
    if missing:
        raise FileNotFoundError(
            "missing price files for "
            + ", ".join(missing)
            + ". Pass --cache-dir, or rerun with --download to fetch via yfinance."
        )
    series = [load_close_csv(paths[name], name) for name in NEEDED]
    return pd.concat(series, axis=1, sort=False).sort_index()
