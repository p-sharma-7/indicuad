"""Convert the JSONL artifacts to plain JSON.

JSONL is what the pipeline writes: it streams, it appends, and a corrupt
line does not destroy the file. Plain JSON is easier to hand to a notebook
or a viewer, so both are shipped.

Every export is round-tripped before it is written — the JSON is parsed
back and compared record-for-record against the source. A conversion that
silently drops or reorders records would be worse than no conversion,
because the two files would disagree and nothing would say so.

    python scripts/export_json.py                # qa.jsonl -> qa/qa.json
    python scripts/export_json.py --all          # every corpus artifact too
    python scripts/export_json.py --compact      # no indentation

Output: <name>.json alongside each source file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    PROJECT_ROOT,
    QA_DIR,
    Stage,
    read_jsonl,
)

PRIMARY = [QA_DIR / "qa.jsonl"]
EXTRA = [
    CORPUS_DIR / "documents.jsonl",
    CORPUS_DIR / "paragraphs.jsonl",
    CORPUS_DIR / "categories.jsonl",
    ANNOTATIONS_DIR / "annotations_reviewed.jsonl",
]


def convert(source: Path, compact: bool, stage: Stage) -> bool:
    if not source.is_file():
        stage.warn(f"{source.name} not found, skipping")
        return True

    records = read_jsonl(source)
    target = source.with_suffix(".json")

    with target.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(records, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)

    # Round-trip: the exported file must parse back to exactly the source.
    with target.open(encoding="utf-8") as handle:
        restored = json.load(handle)

    if restored != records:
        stage.fail(f"{target.name}: round-trip mismatch — export is NOT faithful")
        return False

    size_mb = target.stat().st_size / 1_048_576
    stage.info(
        f"{source.name:<30} -> {target.name:<30} "
        f"{len(records):>6,} records  {size_mb:>6.2f} MB  round-trip OK"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="also export the corpus and annotation artifacts")
    parser.add_argument("--compact", action="store_true",
                        help="no indentation; roughly a third the size")
    args = parser.parse_args()

    stage = Stage("export", "JSONL to JSON")

    sources = PRIMARY + (EXTRA if args.all else [])
    ok = all(convert(s, args.compact, stage) for s in sources)

    if ok:
        stage.info("")
        stage.info("JSONL remains the canonical form; the .json files are a convenience copy.")
        stage.info("Re-run this after any rebuild, or the two will drift apart.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
