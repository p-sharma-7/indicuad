"""Recompute `sweep_hits` on every annotation cell, in place.

`sweep_hits` is the P2 recall aid stored on each cell. It is derived data,
so when the matcher or the segmentation changes it goes stale — and a
stale recall aid is worse than none, because an annotator trusts it.

Only `sweep_hits` is rewritten. Verdicts, spans, annotator IDs and notes
are untouched, so this is safe to run at any point during annotation.

Output: rewrites the current annotations file in place.
"""

from __future__ import annotations

import sys

from indicuad_common import (
    CORPUS_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
    write_jsonl,
)
from sweep import hits as sweep_hits


def main() -> int:
    stage = Stage("P2-refresh", "Recompute sweep_hits")

    source = current_annotations_path()
    cells = read_jsonl(source)
    if not cells:
        stage.fail(f"no annotation cells at {source}")
        return 1

    categories = {c["category_id"]: c for c in read_jsonl(CORPUS_DIR / "categories.jsonl")}
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")

    by_doc: dict[str, list[dict]] = {}
    for paragraph in paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    before = sum(len(c.get("sweep_hits") or []) for c in cells)
    changed = 0
    for cell in cells:
        category = categories.get(cell["category_id"])
        if category is None:
            stage.fail(f"{cell['cell_id']}: unknown category")
            return 1
        fresh = sweep_hits(by_doc.get(cell["doc_id"], []), category["sweep_terms"])
        if fresh != (cell.get("sweep_hits") or []):
            changed += 1
        cell["sweep_hits"] = fresh
    after = sum(len(c.get("sweep_hits") or []) for c in cells)

    write_jsonl(source, cells)

    verdicts = sum(1 for c in cells if c.get("is_impossible") is not None)
    stage.info(f"rewrote sweep_hits in {source.name}")
    stage.info(f"cells changed: {changed}/{len(cells)}")
    stage.info(f"total hits: {before:,} -> {after:,} ({before - after:+,} removed)")
    stage.info(f"verdicts untouched: {verdicts} cells still carry one")

    emit_report(
        "sweep_refresh_report.json",
        {
            "status": "PASS",
            "file": source.name,
            "cells": len(cells),
            "cells_changed": changed,
            "hits_before": before,
            "hits_after": after,
            "verdicts_preserved": verdicts,
            "note": "Only sweep_hits was rewritten; no verdict, span or note was modified.",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
