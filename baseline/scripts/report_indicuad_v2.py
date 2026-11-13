#!/usr/bin/env python3
"""Render results/indicuad_v2/reports/benchmark_report.md plus figures.

  python scripts/report_indicuad_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.cli import build_parser, cmd_report  # noqa: E402
from benchmark.config import load_config  # noqa: E402

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["report"] + sys.argv[1:])
    sys.exit(cmd_report(load_config(), args))
