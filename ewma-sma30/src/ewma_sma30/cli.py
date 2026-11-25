"""Command-line entry point for the live EWMA backtest and close signal.

Parameters are not flags. This program does not place or modify orders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .constants import NAV_START
from .data import load_prices
from .rule import compute_rule
from .signal import closes_through_last_session, latest_complete_signal

_BANNER = (
    "当前实盘 EWMA 规则（SMA30，λ=0.94，5 日百分位回落 −0.70）。"
    "未冻结，不是 Z>1.25 + SMA30 Pearson 规则的替代。本程序不下单。"
)


def _fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _fmt_dd(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float | None, digits: int) -> str:
    if value is None or pd.isna(value):
        return "尚未形成"
    return f"{value:.{digits}f}"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "尚未形成"
    return "是" if value else "否"


def _ma_text(relation: str) -> str:
    return {
        "strictly_below": "严格低于",
        "strictly_above": "严格高于",
        "equal": "相等",
        "unknown": "尚未形成",
    }.get(relation, relation)


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.download:
        closes = load_prices(use_yfinance=True, write_cache_dir=args.write_cache_dir)
        print(f"已下载价格并写入 {Path(args.write_cache_dir).resolve()}", file=sys.stderr)
    else:
        closes = load_prices(cache_dir=args.cache_dir)
    closes = closes_through_last_session(closes)
    if args.end:
        closes = closes.loc[: pd.Timestamp(args.end)]
    if closes.empty:
        raise RuntimeError("no prices on or before the last completed US session")
    return closes, compute_rule(closes)


def _print_backtest(result: dict) -> None:
    rule, spy = result["rule"], result["spy"]
    rows = [
        ("净值=1 的日期（该日收益不计入）", str(pd.Timestamp(NAV_START).date()), ""),
        ("第一个计入的收益日", rule["start"], spy["start"]),
        ("样本终点", rule["end"], spy["end"]),
        ("交易日数", str(rule["n"]), str(spy["n"])),
        ("期末净值", f"{rule['end_nav']:.4f}", f"{spy['end_nav']:.4f}"),
        ("年化收益", _fmt_pct(rule["cagr"]), _fmt_pct(spy["cagr"])),
        ("年化波动", _fmt_pct(rule["vol"]), _fmt_pct(spy["vol"])),
        ("Sharpe（rf=0）", f"{rule['sharpe']:.4f}", f"{spy['sharpe']:.4f}"),
        ("最大回撤", _fmt_dd(rule["max_dd"]), _fmt_dd(spy["max_dd"])),
        ("空仓时间占比", _fmt_pct(rule["pct_cash"], 1), "0"),
        ("空仓天数", str(int(rule["n_cash"])), "0"),
        ("仓位切换次数", str(int(rule["n_switches"])), "0"),
    ]
    print("| 项目 | EWMA 规则 | 买入持有 SPY |")
    print("| --- | --- | --- |")
    for name, left, right in rows:
        print(f"| {name} | {left} | {right} |")


def _print_signal(sig: dict) -> None:
    position = "空仓（BIL）" if sig["signal_cash"] else "满仓 SPY"
    print("| 项目 | 值 |")
    print("| --- | --- |")
    print(f"| 收盘日期 | {sig['asof']} |")
    print(f"| S | {_num(sig['S'], 4)} |")
    print(f"| 252 日百分位 | {_num(sig['pct'], 4)} |")
    print(f"| 5 日最高百分位 | {_num(sig['high5'], 4)} |")
    print(f"| 5 日回落 drop5 | {_num(sig['drop5'], 4)} |")
    print(f"| z(SPY,TLT) | {_num(sig['z_spy_tlt'], 4)} |")
    print(f"| z(SPY,DXY) | {_num(sig['z_spy_dxy'], 4)} |")
    print(f"| z(SPY,GLD) | {_num(sig['z_spy_gld'], 4)} |")
    print(f"| z(TLT,DXY) | {_num(sig['z_tlt_dxy'], 4)} |")
    print(f"| SPY 收盘 | {_num(sig['spy'], 2)} |")
    print(f"| SPY 30 日均线 | {_num(sig['spy_sma30'], 2)} |")
    print(f"| 收盘价相对均线 | {_ma_text(sig['ma_relation'])} |")
    print(f"| 规则 1 风险关闭（百分位 > 0.9） | {_yes_no(sig['risk_off_r1'])} |")
    print(f"| 规则 1 均线闩锁 | {_yes_no(sig['latched'])} |")
    print(f"| 规则 1 收盘空仓 | {_yes_no(sig['rule1_triggered'])} |")
    print(f"| 规则 2 收盘空仓 | {_yes_no(sig['rule2_triggered'])} |")
    print(f"| 收盘信号 | {'空仓' if sig['signal_cash'] else '满仓 SPY'} |")
    print(f"| 次日生效仓位 | {position} |")


def cmd_backtest(args: argparse.Namespace) -> int:
    _closes, daily = _load(args)
    result = run_backtest(daily, end=args.end, start=pd.Timestamp(args.start))
    print(_BANNER)
    print("0 成本。下面是这次数据的现算结果，不是样本外验证，也没有登记成正式业绩。")
    print()
    _print_backtest(result)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out, float_format="%.10f")
        print(f"\n已写入日序列：{out}")
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    closes, daily = _load(args)
    sig = latest_complete_signal(closes, daily)
    print(_BANNER)
    print()
    _print_signal(sig)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ewma-sma30",
        description=(
            "Live SPY cash/invested signal from an EWMA signed covariance, "
            "SMA30, lambda 0.94, and a 5-day percentile drop of -0.70. "
            "Not frozen, and not a replacement for the Z>1.25 Pearson rule. "
            "Parameters are not flags. This program does not trade."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="directory of SPY/TLT/GLD/DXY/BIL CSV caches")
    parser.add_argument("--download", action="store_true", help="download prices with yfinance (auto_adjust=True)")
    parser.add_argument(
        "--write-cache-dir",
        type=Path,
        default=Path("cache"),
        help="where to write downloaded caches (never an official research cache)",
    )
    parser.add_argument("--start", default=str(NAV_START.date()), help="NAV=1 date for backtest (that day's return is excluded)")
    parser.add_argument("--end", default=None, help="inclusive end date; default is the last completed common session")
    sub = parser.add_subparsers(dest="cmd", required=True)
    backtest = sub.add_parser("backtest", help="0-cost backtest versus buy-and-hold SPY")
    backtest.add_argument("--out", type=Path, default=None, help="write the daily series to this CSV")
    backtest.set_defaults(func=cmd_backtest)
    signal = sub.add_parser("signal", help="latest completed close signal, effective next session")
    signal.set_defaults(func=cmd_signal)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
