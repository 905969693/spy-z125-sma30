#!/usr/bin/env python3
"""Thin wrapper around the package CLI official backtest."""
from spy_z125_sma30.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["backtest", "--windows"]))
