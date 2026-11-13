"""Stage 10 — unroll annotated cells into qa/qa.jsonl query rows.

One cell becomes:
    core split      3 rows (canonical, semi_formal, naive_layperson)
    negatives split 1 row  (canonical only)

All rows from one cell share doc_id, category_id, intent_group_id, the
gold answer, and gold_paragraph_ids. Only the question text differs. The
three registers are one observation asked three ways, which is why the
intent_group_id is carried into the output — significance tests must
group on it rather than treating the rows as independent.

Refuses to run on an incompletely annotated workspace: a cell with
is_impossible=null has no verdict, and emitting it as a query row would
silently turn "not yet annotated" into "no answer exists".

Output: qa/qa.jsonl
"""

from __future__ import annotations

import sys

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    QA_DIR,
    Stage,
    emit_report,
    read_jsonl,
    write_jsonl,
)

REGISTER_FIELD = {
    "canonical": "canonical_question",
    "semi_formal": "semi_formal_question",
    "naive_layperson": "naive_layperson_question",
}

SPLIT_REGISTERS = {
    "core": ["canonical", "semi_formal", "naive_layperson"],
    "negatives": ["canonical"],
}


def intent_group_id(doc_id: str, category_id: str) -> str:
    return f"IG-{doc_id.replace('-', '')}-{category_id.replace('-', '')}"


def main() -> int:
    stage = Stage("10", "Query row generation")

    source = ANNOTATIONS_DIR / "annotations_adjudicated.jsonl"
    if not source.is_file():
        source = ANNOTATIONS_DIR / "annotations_reviewed.jsonl"
    if not source.is_file():
        source = ANNOTATIONS_DIR / "annotations_raw.jsonl"

    cells = read_jsonl(source)
    if not cells:
        stage.fail("no annotation cells found — run build_annotation_workspace.py first.")
        return 1
    stage.info(f"reading cells from {source.relative_to(source.parent.parent)}")

    categories = {c["category_id"]: c for c in read_jsonl(CORPUS_DIR / "categories.jsonl")}
    documents = {d["doc_id"]: d for d in read_jsonl(CORPUS_DIR / "documents.jsonl")}

    # Cells scoped out by instrument type (§A-4) are not emitted as queries.
    # Asking a power of attorney about warranty duration tests nothing, and
    # counting it as an abstention case inflates the negative class. The
    # cells remain in the annotations file so the decision stays auditable.
    scoped_out = [c for c in cells if c.get("applicable") is False]
    if scoped_out:
        cells = [c for c in cells if c.get("applicable") is not False]
        stage.info(
            f"{len(scoped_out)} cell(s) scoped out by instrument type; "
            f"{len(cells)} remain"
        )

    unannotated = [c["cell_id"] for c in cells if c.get("is_impossible") is None]
    if unannotated:
        stage.fail(
            f"{len(unannotated)} of {len(cells)} cells still have is_impossible=null. "
            "Query generation would misreport them as negatives."
        )
        stage.info(f"first few: {unannotated[:8]}")
        emit_report(
            "qa_build_report.json",
            {"status": "BLOCKED_INCOMPLETE_ANNOTATION",
             "cells_total": len(cells),
             "cells_unannotated": len(unannotated)},
        )
        return 1

    rows = []
    counter = 0
    per_register: dict[str, int] = {}

    for cell in cells:
        document = documents.get(cell["doc_id"])
        category = categories.get(cell["category_id"])
        if document is None or category is None:
            stage.fail(f"{cell['cell_id']}: unknown document or category")
            return 1

        registers = SPLIT_REGISTERS.get(cell["split"])
        if registers is None:
            stage.fail(f"{cell['cell_id']}: unexpected split {cell['split']!r}")
            return 1

        group = intent_group_id(cell["doc_id"], cell["category_id"])

        for register in registers:
            counter += 1
            rows.append(
                {
                    "qa_id": f"IND-Q-{counter:05d}",
                    "doc_id": cell["doc_id"],
                    "category_id": cell["category_id"],
                    "question": category[REGISTER_FIELD[register]],
                    "question_register": register,
                    "intent_group_id": group,
                    "answers": cell["answers"],
                    "is_impossible": cell["is_impossible"],
                    "gold_paragraph_ids": cell["gold_paragraph_ids"],
                    "doc_origin": document["doc_origin"],
                    "tier": document["tier"],
                    "split": cell["split"],
                    "annotator_ids": cell["annotator_ids"],
                    "annotation_pass": cell["annotation_pass"],
                    "model_assisted": cell["model_assisted"],
                    "adjudicated": cell["adjudicated"],
                    "jaccard_agreement": cell["jaccard_agreement"],
                    "annotator_notes": cell["annotator_notes"],
                }
            )
            per_register[register] = per_register.get(register, 0) + 1

    write_jsonl(QA_DIR / "qa.jsonl", rows)

    positives = sum(1 for r in rows if not r["is_impossible"])
    core_cells = sum(1 for c in cells if c["split"] == "core")
    negative_cells = sum(1 for c in cells if c["split"] == "negatives")

    stage.info(f"cells: {len(cells)} ({core_cells} core, {negative_cells} negatives)")
    stage.info(f"query rows: {len(rows)}  " + ", ".join(f"{k}={v}" for k, v in per_register.items()))
    stage.info(f"positive rows: {positives}   abstention rows: {len(rows) - positives}")
    stage.info("wrote qa/qa.jsonl")

    emit_report(
        "qa_build_report.json",
        {
            "status": "PASS",
            "source_file": source.name,
            "cells_total": len(cells),
            "core_cells": core_cells,
            "negative_cells": negative_cells,
            "query_rows": len(rows),
            "rows_per_register": per_register,
            "positive_rows": positives,
            "abstention_rows": len(rows) - positives,
            "intent_groups": len({r["intent_group_id"] for r in rows}),
            "reporting_note": (
                "Report N_documents, N_categories, N_cells and N_queries together. "
                "Significance tests group on intent_group_id, not on query rows."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
