"""Return named cells to the unannotated state.

Used when a verdict should not stand — because it was model-derived rather
than independently judged, or because the call is contested and needs the
§7 legal review first.

A reverted cell goes back to `is_impossible: null`, which means "not yet
annotated". It is NOT removed from the workspace: the 780-cell grid is the
benchmark's denominator, and dropping the cells that were hard to call
would bias the gold toward easy ones and inflate every downstream number.
`build_qa.py` refuses to ship a null cell, so a reverted cell simply waits
for a human instead of shipping a judgement nobody stands behind.

The reason is written into `annotator_notes` so the reversion is visible in
the data, not only in this file.

    python scripts/revert_cells.py --reason "..." CELL-ID [CELL-ID ...]
    python scripts/revert_cells.py --reason "..." --from-file cells.txt
"""

from __future__ import annotations

import argparse
import sys

from indicuad_common import (
    ANNOTATIONS_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
    write_jsonl,
)

BLANK = {
    "is_impossible": None,
    "answers": [],
    "gold_paragraph_ids": [],
    "status": "UNANNOTATED",
    "annotation_pass": None,
    "adjudicated": False,
    "jaccard_agreement": None,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="*", help="cell_ids to revert")
    parser.add_argument("--from-file", help="file with one cell_id per line")
    parser.add_argument("--reason", required=True, help="recorded on every reverted cell")
    args = parser.parse_args()

    stage = Stage("A/V", "Revert cells to unannotated")

    targets = list(args.cells)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as handle:
            targets += [line.strip() for line in handle if line.strip()]
    targets = sorted(set(targets))
    if not targets:
        stage.fail("no cell_ids given")
        return 1

    source = current_annotations_path()
    cells = read_jsonl(source)
    if not cells:
        stage.fail(f"no annotation cells at {source}")
        return 1
    stage.info(f"reading {source.name}")

    by_id = {c["cell_id"]: c for c in cells}
    unknown = [t for t in targets if t not in by_id]
    if unknown:
        stage.fail(f"unknown cell_id(s): {unknown[:10]}")
        return 1

    reverted, already = [], []
    for cell_id in targets:
        cell = by_id[cell_id]
        if cell.get("is_impossible") is None:
            already.append(cell_id)
            continue
        cell.update(BLANK)
        cell["annotator_notes"] = f"REVERTED: {args.reason}"
        # The annotator did not make this call stand, so their name comes off it.
        cell["annotator_ids"] = []
        reverted.append(cell_id)

    write_jsonl(source, list(by_id.values()))

    remaining = sum(1 for c in by_id.values() if c.get("is_impossible") is not None)
    positives = sum(1 for c in by_id.values() if c.get("is_impossible") is False)

    stage.info(f"reverted {len(reverted)} cell(s); {len(already)} were already unannotated")
    stage.info(f"cells still carrying a verdict: {remaining}/{len(cells)} ({positives} positive)")
    stage.info(f"reason recorded: {args.reason}")

    emit_report(
        "revert_report.json",
        {
            "status": "PASS",
            "file": source.name,
            "reason": args.reason,
            "reverted": reverted,
            "already_unannotated": already,
            "cells_with_verdict": remaining,
            "positive_cells": positives,
            "note": (
                "Reverted cells remain in the workspace as unannotated. They are not "
                "removed from the benchmark — excluding hard-to-call cells would bias "
                "the gold toward easy ones."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
