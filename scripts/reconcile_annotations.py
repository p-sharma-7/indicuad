"""Sync the annotation chain head with the current cell grid.

`build_annotation_workspace.py` rebuilds `annotations_raw.jsonl` — the
scaffold of every (document, category) cell implied by the current splits.
Verdicts, though, live further up the chain, in
`annotations_reviewed.jsonl`. So after documents are added to or removed
from a split, the scaffold and the chain head disagree: the head is missing
the new documents' cells and still holds cells for documents that have
left.

This reconciles them, and is deliberately loud about the direction of each
change:

    added     a cell in the scaffold that the head lacks     -> is_impossible null
    kept      a cell in both                                 -> the head's record wins
    dropped   a cell in the head that the scaffold lacks     -> removed, and if it
                                                                carried a verdict that
                                                                is reported, never silent

Dropping a cell that holds a real clause is how a corpus quietly loses
gold. The count of dropped positives is printed and written to the report
every time, so a document leaving the corpus is always visible as a number.

    reconcile_annotations.py [--apply]

Without --apply it reports and changes nothing.

Output: annotations/<chain head>, validation/reconcile_report.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from indicuad_common import (
    ANNOTATIONS_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
    write_jsonl,
)

# Fields the scaffold owns: they follow the corpus, not the annotator.
SCAFFOLD_FIELDS = ("split", "doc_id", "category_id", "sweep_hits")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the result")
    args = parser.parse_args()

    stage = Stage("8b", "Reconcile annotation chain with cell grid")

    scaffold = read_jsonl(ANNOTATIONS_DIR / "annotations_raw.jsonl")
    if not scaffold:
        stage.fail("annotations_raw.jsonl is empty — run build_annotation_workspace.py")
        return 1

    head_path = current_annotations_path()
    head = {c["cell_id"]: c for c in read_jsonl(head_path)}
    stage.info(f"chain head: {head_path.name} ({len(head)} cells)")
    stage.info(f"scaffold:   annotations_raw.jsonl ({len(scaffold)} cells)")

    merged = []
    added = kept = 0
    for cell in scaffold:
        previous = head.get(cell["cell_id"])
        if previous is None:
            merged.append(cell)
            added += 1
            continue
        record = dict(previous)
        for field in SCAFFOLD_FIELDS:
            if field in cell:
                record[field] = cell[field]
        merged.append(record)
        kept += 1

    scaffold_ids = {c["cell_id"] for c in scaffold}
    dropped = [c for cid, c in head.items() if cid not in scaffold_ids]
    dropped_positive = [c["cell_id"] for c in dropped if c.get("is_impossible") is False]
    dropped_negative = sum(1 for c in dropped if c.get("is_impossible") is True)
    by_doc = Counter(c["doc_id"] for c in dropped)

    stage.info(f"added   {added:>5}   cells for documents new to the grid")
    stage.info(f"kept    {kept:>5}   cells, verdicts preserved")
    stage.info(f"dropped {len(dropped):>5}   cells for documents no longer in a split")
    if dropped:
        for doc_id, count in sorted(by_doc.items()):
            stage.info(f"    {doc_id}: {count}")
    if dropped_positive:
        stage.info(
            f"  {len(dropped_positive)} of the dropped cells held a real clause — "
            "that gold leaves the corpus with the document"
        )
        stage.info(f"    {dropped_positive[:10]}")

    verdicts = sum(1 for c in merged if c.get("is_impossible") is not None)
    positives = sum(1 for c in merged if c.get("is_impossible") is False)
    stage.info(f"after reconcile: {verdicts} verdicts, {positives} positive, "
               f"{len(merged) - verdicts} open")

    if args.apply:
        write_jsonl(head_path, merged)
        stage.info(f"wrote {head_path.name}")
    else:
        stage.info("dry run — pass --apply to write")

    emit_report(
        "reconcile_report.json",
        {
            "status": "PASS" if args.apply else "DRY_RUN",
            "chain_head": head_path.name,
            "cells_added": added,
            "cells_kept": kept,
            "cells_dropped": len(dropped),
            "dropped_by_document": dict(by_doc),
            "dropped_positive_cells": dropped_positive,
            "dropped_negative_cells": dropped_negative,
            "verdicts_after": verdicts,
            "positive_after": positives,
            "open_after": len(merged) - verdicts,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
