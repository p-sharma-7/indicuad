#!/usr/bin/env python3
"""Merge row-sharded raw output files (from run_model.py --row-shard-index/
--row-shard-count) into the canonical {context_condition}.jsonl that
scoring/analysis actually read.

  python scripts/merge_shards.py --model MODEL_KEY --context {full_contract,oracle_local} --shard-count N

Refuses to merge if any (qa_id, setting) key is duplicated across shards —
sharding is supposed to be a strict partition, so a duplicate means a shard
was re-run with a different --row-shard-count than the others, or two
shards were pointed at overlapping data. Never overwrites a shard file;
only writes the canonical output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import load_config, read_jsonl, append_jsonl  # noqa: E402
from benchmark.models.base import raw_output_path  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context", choices=["full_contract", "oracle_local"], required=True)
    p.add_argument("--shard-count", type=int, required=True)
    args = p.parse_args()

    cfg = load_config()
    canonical = raw_output_path(cfg, args.model, args.context)

    seen: dict[tuple[str, str], str] = {}
    records = []
    for idx in range(args.shard_count):
        shard_path = raw_output_path(cfg, args.model, args.context, shard_suffix=f"shard{idx}of{args.shard_count}")
        if not shard_path.is_file():
            print(f"MISSING shard {idx}/{args.shard_count}: {shard_path} — not merging")
            return 1
        n = 0
        for rec in read_jsonl(shard_path):
            key = (rec["qa_id"], rec["setting"])
            if key in seen:
                print(f"DUPLICATE key {key} in shard {idx} (already seen in {seen[key]}) — refusing to merge")
                return 1
            seen[key] = f"shard{idx}"
            records.append(rec)
            n += 1
        print(f"shard {idx}/{args.shard_count}: {n} records from {shard_path}")

    if canonical.is_file():
        canonical.unlink()
    for rec in records:
        append_jsonl(canonical, rec)
    print(f"merged {len(records)} records -> {canonical}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
