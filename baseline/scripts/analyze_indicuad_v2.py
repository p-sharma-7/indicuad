#!/usr/bin/env python3
"""Aggregate scored results into leaderboards, breakdowns, bootstrap CIs,
and pairwise comparisons under results/indicuad_v2/analysis/.

  python scripts/analyze_indicuad_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.cli import build_parser, cmd_analyze  # noqa: E402
from benchmark.config import load_config  # noqa: E402

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["analyze"] + sys.argv[1:])
    sys.exit(cmd_analyze(load_config(), args))
