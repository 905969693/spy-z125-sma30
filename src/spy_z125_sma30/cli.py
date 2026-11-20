"""Command-line entry point: backtest and latest close signal."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .backtest import crisis_table, official_backtest
from .constants import NAV_START, OFFICIAL_SAMPLE_END
from .data import load_official_prices, load_prices
from .rule import compute_rule
from .signal import latest_signal


def _fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def _fmt_dd(x: float) -> str:
    return f"{x * 100:.1f}%"


def _print_stats_table(result: dict) -> None:
    rule, spy = result["rule"], result["spy"]
    rows = [
        ("样本起点（净值=1 的前一日）", str(pd.Timestamp(NAV_START).date()), ""),
        ("第一个收益日", rule["start"], spy["start"]),
        ("样本终点", rule["end"], spy["end"]),
        ("交易日数", str(rule["n"]), str(spy["n"])),
        ("期末净值", f"{rule['end_nav']:.2f}", f"{spy['end_nav']:.2f}"),
        ("年化收益", _fmt_pct(rule["cagr"]), _fmt_pct(spy["cagr"])),
        ("年化波动", _fmt_pct(rule["vol"]), _fmt_pct(spy["vol"])),
        ("Sharpe（rf=0）", f"{rule['sharpe']:.2f}", f"{spy['sharpe']:.2f}"),
        ("最大回撤", _fmt_dd(rule["max_dd"]), _fmt_dd(spy["max_dd"])),
        ("空仓时间占比", _fmt_pct(rule["pct_cash"], 1), "0"),
        ("空仓天数", str(int(rule["n_cash"])), "0"),
        ("仓位切换次数", str(int(rule["n_switches"])), "0"),
    ]
    print("| 项目 | 冻结规则 | 买入持有 SPY |")
    print("| --- | --- | --- |")
    for name, a, b in rows:
        print(f"| {name} | {a} | {b} |")


def _print_crisis(daily: pd.DataFrame) -> None:
    tbl = crisis_table(daily)
    if tbl.empty:
        return
    print()
    print("| 窗口 | 起止 | 规则收益 | 规则最大回撤 | SPY 收益 | SPY 最大回撤 | 空仓 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in tbl.iterrows():
        cash = f"{row['pct_cash']*100:.1f}%（{int(row['n_cash'])} 天）"
        print(
            f"| {row['window']} | {row['start']} 至 {row['end']} | "
            f"{_fmt_pct(row['rule_ret'], 1)} | {_fmt_dd(row['rule_max_dd'])} | "
            f"{_fmt_pct(row['spy_ret'], 1)} | {_fmt_dd(row['spy_max_dd'])} | {cash} |"
        )


def _print_signal(sig: dict) -> None:
    print("| 项目 | 值 |")
    print("| --- | --- |")
    print(f"| 收盘日期 | {sig['asof']} |")
    print(f"| s | {sig['s']:.3f} |")
    print(f"| Z | {sig['z']:.2f} |")
    print(f"| SPY 收盘 | {sig['spy']:.2f} |")
    print(f"| SPY 30 日均线 | {sig['spy_ma30']:.2f} |")
    print(f"| 收盘价是否严格低于均线 | {'是' if sig['below_ma30'] else '否'} |")
    print(f"| Z 是否大于 1.25 | {'是' if sig['z_gt_125'] else '否'} |")
    print(f"| 收盘信号 | {'空仓' if sig['signal_cash'] else '满仓 SPY'} |")
    print(f"| 次日生效仓位 | {'空仓（BIL）' if sig['signal_cash'] else '满仓 SPY'} |")


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.official_sample:
        closes = load_official_prices()
    else:
        closes = load_prices(
            cache_dir=args.cache_dir,
            use_yfinance=args.download,
            write_cache_dir=args.write_cache_dir if args.download else None,
        )
    return compute_rule(closes)


def cmd_backtest(args: argparse.Namespace) -> int:
    daily = _load(args)
    end = args.end
    result = official_backtest(daily, end=end, start=NAV_START)
    print("0 成本。这些数字不是样本外验证。")
    print()
    _print_stats_table(result)
    if args.windows:
        _print_crisis(daily.loc[: pd.Timestamp(end)] if end else daily)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out, float_format="%.10f")
        print(f"\n已写入日序列：{out}")
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    daily = _load(args)
    if args.end:
        daily = daily.loc[: pd.Timestamp(args.end)]
    sig = latest_signal(daily)
    _print_signal(sig)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spy-z125-sma30",
        description=(
            "Frozen SPY cash/invested rule: four-asset correlation speed "
            "Z>1.25 plus SPY 30-day SMA gate. Parameters are not flags."
        ),
    )
    p.add_argument("--cache-dir", type=Path, default=None, help="directory of SPY/TLT/GLD/DXY/BIL CSV caches")
    p.add_argument(
        "--official-sample",
        action="store_true",
        help="use the frozen official-sample price snapshot (reproduces documented 2007-06-01 to 2026-11-13 stats)",
    )
    p.add_argument("--download", action="store_true", help="download prices with yfinance (auto_adjust=True)")
    p.add_argument(
        "--write-cache-dir",
        type=Path,
        default=Path("cache"),
        help="where to write downloaded caches (never official research paths)",
    )
    p.add_argument(
        "--end",
        default=None,
        help=f"inclusive end date (backtest default: official sample end {OFFICIAL_SAMPLE_END}; use 'latest' for the last available date)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="official-sample metrics versus buy-and-hold SPY")
    b.add_argument("--windows", action="store_true", help="also print pre-specified crisis windows")
    b.add_argument("--out", type=Path, default=None, help="write the daily series to this CSV")
    b.set_defaults(func=cmd_backtest)
    s = sub.add_parser("signal", help="latest close signal, effective next session")
    s.set_defaults(func=cmd_signal)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.end is not None and str(args.end).lower() == "latest":
        args.end = None
    elif args.cmd == "backtest" and args.end is None:
        args.end = OFFICIAL_SAMPLE_END
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
