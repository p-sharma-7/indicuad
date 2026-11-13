"""Verify the figures quoted in DATASET_METHODOLOGY.html against the live build.

A methodology report is only useful if its numbers are the corpus's numbers.
They drift silently: a document is re-extracted, a cell is re-annotated, and
prose written an hour earlier is quietly wrong. This recomputes every headline
figure and fails if the document disagrees.

    python scripts/check_doc_figures.py

Add new figures here as they are added to the document, otherwise the check
gives false confidence about the parts it does not cover.
"""

from __future__ import annotations

import re
import sys

from indicuad_common import (
    CORPUS_DIR,
    PROJECT_ROOT,
    QA_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
)

DOCUMENT = PROJECT_ROOT / "DATASET_METHODOLOGY.html"


def live_figures() -> dict[str, int]:
    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    cells = read_jsonl(current_annotations_path())
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    categories = read_jsonl(CORPUS_DIR / "categories.jsonl")
    rows = read_jsonl(QA_DIR / "qa.jsonl")

    positives = [c for c in cells if c.get("is_impossible") is False]
    positive_rows = [r for r in rows if not r["is_impossible"]]

    return {
        "documents": len(documents),
        "core_documents": sum(1 for d in documents if d["split"] == "core"),
        "negatives_documents": sum(1 for d in documents if d["split"] == "negatives"),
        "excluded_documents": sum(1 for d in documents if d["split"] == "excluded"),
        "core_chars": sum(d["char_count"] for d in documents if d["split"] == "core"),
        "categories": len(categories),
        "cells": len(cells),
        "paragraphs": len(paragraphs),
        "furniture_paragraphs": sum(1 for p in paragraphs if p.get("is_boilerplate")),
        "superseded_paragraphs": sum(1 for p in paragraphs if p.get("is_superseded")),
        "indexed_paragraphs": sum(
            1 for p in paragraphs
            if not p.get("is_boilerplate") and not p.get("is_superseded")
        ),
        "positive_cells": len(positives),
        "gold_spans": sum(len(c["answers"]) for c in positives),
        "query_rows": len(rows),
        "positive_rows": len(positive_rows),
        "abstention_rows": len(rows) - len(positive_rows),
        "effective_n": len({r["intent_group_id"] for r in positive_rows}),
        "human_cells": sum(
            1 for c in cells
            if c.get("is_impossible") is not None
            and (c.get("verdict_source") or "human") == "human"
        ),
        "model_cells": sum(
            1 for c in cells
            if c.get("verdict_source") == "model_proposed_human_approved"
        ),
        "high_contest_core": sum(
            1 for c in cells
            if c["split"] == "core" and c["attention_tier"] == "high_contest"
        ),
    }


def main() -> int:
    stage = Stage("doc-check", "Verify methodology figures")

    if not DOCUMENT.is_file():
        stage.fail(f"{DOCUMENT.name} not found")
        return 1

    text = DOCUMENT.read_text(encoding="utf-8")
    # Digits are written with thousands separators in prose; compare on the
    # bare integers so 1,742 and 1742 are the same claim.
    quoted = {int(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]*\b", text)}

    figures = live_figures()
    missing = {k: v for k, v in figures.items() if v not in quoted}

    for key, value in sorted(figures.items()):
        mark = " " if value in quoted else "!"
        stage.info(f"  [{mark}] {key:<24} {value:,}")

    if missing:
        stage.fail(
            f"{len(missing)} live figure(s) do not appear anywhere in "
            f"{DOCUMENT.name}: {sorted(missing)}"
        )
        stage.info(
            "Either the document is stale, or the figure is genuinely absent from "
            "it — this check cannot tell those apart, so read the section."
        )

    emit_report(
        "doc_figures_report.json",
        {
            "status": "FAILED" if missing else "PASS",
            "document": DOCUMENT.name,
            "figures": figures,
            "not_found_in_document": missing,
            "note": (
                "Presence-based check: confirms each live figure appears somewhere "
                "in the document. It cannot detect a figure quoted in the wrong "
                "place, only one that is absent."
            ),
        },
    )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
