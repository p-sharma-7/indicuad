#!/usr/bin/env python3
"""Score raw predictions in results/indicuad_v2/raw/ against gold + distractors.

  python scripts/score_indicuad_v2.py [--overwrite]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.cli import build_parser, cmd_score  # noqa: E402
from benchmark.config import load_config  # noqa: E402

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["score"] + sys.argv[1:])
    sys.exit(cmd_score(load_config(), args))
