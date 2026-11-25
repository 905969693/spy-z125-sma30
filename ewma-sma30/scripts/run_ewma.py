#!/usr/bin/env python3
"""Run the live EWMA CLI. Install the package in this directory first.

This is not the frozen Pearson rule, and it does not place orders.
"""
from __future__ import annotations

import sys

from ewma_sma30.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
