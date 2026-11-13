"""Stage 8 — scaffold the empty annotation workspace.

Creates one row per (document, category) cell for every document in the
core and negatives splits, with NO verdict filled in. Claude Code builds
the grid; humans fill it. Every cell starts:

    is_impossible: null      (null means "not yet annotated", not "empty")
    answers: []
    status: "UNANNOTATED"

Tier D documents get no cells at all — the protocol never annotates them.

Each cell carries the P2 keyword sweep result as `sweep_hits`: paragraph
IDs whose text matches the category's sweep terms. These are candidate
locations to check, explicitly NOT proposed labels. Protocol §5 P2 is a
recall aid for a human reader, and the field is named to keep that
distinction visible in the data.

Output: annotations/annotations_raw.jsonl
"""

from __future__ import annotations

import sys

from sweep import hits as sweep_hits

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    Stage,
    emit_report,
    read_jsonl,
    write_jsonl,
)

ANNOTATED_SPLITS = {"core", "negatives"}


def sweep(paragraphs: list[dict], terms: list[str]) -> list[str]:
    """Paragraph IDs containing any sweep term. Candidates, not labels.

    Delegates to scripts/sweep.py, which handles acronyms, short words and
    stems differently — a plain substring match made 'LD' hit "would" and
    'cap' hit "CAPACITY", drowning the real hits. Boilerplate is skipped:
    running headers are not part of the instrument.
    """
    return sweep_hits(paragraphs, terms)


def main() -> int:
    stage = Stage("8", "Annotation workspace scaffold")

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    categories = read_jsonl(CORPUS_DIR / "categories.jsonl")
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")

    if not documents or not categories or not paragraphs:
        stage.fail("corpus incomplete — build documents, paragraphs and categories first.")
        return 1

    untriaged = [d["doc_id"] for d in documents if not d.get("split")]
    if untriaged:
        stage.fail(
            f"{len(untriaged)} document(s) have no split assigned. "
            "Run apply_triage.py after a human reviews triage_report.csv."
        )
        emit_report(
            "annotation_workspace_report.json",
            {"status": "BLOCKED_PENDING_HUMAN_REVIEW",
             "documents_without_split": untriaged},
        )
        return 1

    existing = ANNOTATIONS_DIR / "annotations_raw.jsonl"
    # Merge rather than refuse. The corpus grows during a project, and a new
    # document must be able to add its cells without discarding verdicts
    # already recorded for the documents that were there before. Only
    # sweep_hits is refreshed on an existing cell; nothing else is touched.
    previous = {c["cell_id"]: c for c in read_jsonl(existing)}
    carried = sum(1 for c in previous.values() if c.get("is_impossible") is not None)

    by_doc: dict[str, list[dict]] = {}
    for paragraph in paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    cells = []
    per_split: dict[str, int] = {}
    for document in documents:
        split = document["split"]
        if split not in ANNOTATED_SPLITS:
            continue
        doc_paragraphs = by_doc.get(document["doc_id"], [])
        for category in categories:
            cell_id = f"{document['doc_id']}x{category['category_id']}"
            if cell_id in previous:
                carried_cell = dict(previous[cell_id])
                carried_cell["sweep_hits"] = sweep(doc_paragraphs, category["sweep_terms"])
                cells.append(carried_cell)
                per_split[split] = per_split.get(split, 0) + 1
                continue
            cells.append(
                {
                    "cell_id": f"{document['doc_id']}x{category['category_id']}",
                    "doc_id": document["doc_id"],
                    "category_id": category["category_id"],
                    "category_name": category["category_name"],
                    "attention_tier": category["attention_tier"],
                    "tier": document["tier"],
                    "split": split,
                    # null = not yet annotated. Distinct from false (positive)
                    # and true (confirmed empty).
                    "is_impossible": None,
                    "answers": [],
                    "gold_paragraph_ids": [],
                    "status": "UNANNOTATED",
                    "sweep_hits": sweep(doc_paragraphs, category["sweep_terms"]),
                    "annotator_ids": [],
                    "annotation_pass": None,
                    "model_assisted": False,
                    "adjudicated": False,
                    "jaccard_agreement": None,
                    "annotator_notes": "",
                }
            )
            per_split[split] = per_split.get(split, 0) + 1

    write_jsonl(existing, cells)

    core_docs = [d for d in documents if d["split"] == "core"]
    negative_docs = [d for d in documents if d["split"] == "negatives"]
    high_contest = sum(1 for c in cells if c["attention_tier"] == "high_contest"
                       and c["split"] == "core")

    # Query rows: core cells get 3 registers, negatives get canonical only.
    projected_queries = per_split.get("core", 0) * 3 + per_split.get("negatives", 0)

    stage.info(f"core documents      {len(core_docs):>3}  -> {per_split.get('core', 0)} cells")
    stage.info(f"negatives documents {len(negative_docs):>3}  -> {per_split.get('negatives', 0)} cells")
    stage.info(f"total cells         {len(cells):>3}  (carried {carried} existing verdict(s))")
    stage.info(f"projected query rows {projected_queries:>3} (core x3 registers, negatives x1)")
    stage.info(f"high-contest core cells requiring double annotation: {high_contest}")
    stage.info("all cells written with is_impossible=null, status=UNANNOTATED")

    emit_report(
        "annotation_workspace_report.json",
        {
            "status": "PASS",
            "cells_total": len(cells),
            "cells_by_split": per_split,
            "core_documents": [d["doc_id"] for d in core_docs],
            "negatives_documents": [d["doc_id"] for d in negative_docs],
            "projected_query_rows": projected_queries,
            "high_contest_core_cells": high_contest,
            "gold_labels_written": 0,
            "note": (
                "sweep_hits are P2 keyword-sweep candidates for a human to check, "
                "not proposed labels."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
