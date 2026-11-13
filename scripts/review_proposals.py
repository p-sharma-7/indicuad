"""Human review of model proposals — the step that turns them into gold.

Two modes:

    review_proposals.py                 emit annotations/review_worksheet.jsonl
    review_proposals.py --promote \\
        --reviewer "Name, Role"         write annotations/annotations_reviewed.jsonl

The worksheet has one row per proposed cell with `verdict: null`. A human
sets each to:

    ACCEPT   the proposed spans are correct -> is_impossible = false
    EMPTY    the cell has no clause          -> is_impossible = true
    EDIT     supply `corrected_spans` (quoted text) -> is_impossible = false

`--promote` refuses while any verdict is null, and refuses a NO_CANDIDATE
row marked ACCEPT — "the model found nothing" is not a finding, and
accepting it would smuggle a model-authored negative into gold against
Appendix A rule 1. That cell needs EMPTY (the human confirming) or EDIT
(the human found what the model missed).

Cells with no proposal are untouched: they stay is_impossible=null in
annotations_raw.jsonl and are annotated by a human from scratch, which is
the protocol default for the moderate and high-contest categories.
"""

from __future__ import annotations

import argparse
import sys

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    Stage,
    emit_report,
    read_jsonl,
    write_jsonl,
)
from resolve_spans import MASTHEAD_CATEGORY, SpanError, build_answer

ACCEPT, EMPTY, EDIT = "ACCEPT", "EMPTY", "EDIT"
VERDICTS = {ACCEPT, EMPTY, EDIT}

WORKSHEET = ANNOTATIONS_DIR / "review_worksheet.jsonl"
PROPOSED = ANNOTATIONS_DIR / "annotations_proposed.jsonl"
REVIEWED = ANNOTATIONS_DIR / "annotations_reviewed.jsonl"


def emit_worksheet(stage: Stage) -> int:
    proposals = read_jsonl(PROPOSED)
    if not proposals:
        stage.fail(f"no proposals at {PROPOSED} — run propose_annotations.py first.")
        return 1

    # Merge rather than overwrite: proposals arrive in passes, and a later
    # pass must not discard verdicts already recorded in an earlier one.
    previous = {row["cell_id"]: row for row in read_jsonl(WORKSHEET)}
    kept = sum(1 for row in previous.values() if row.get("verdict"))

    rows = []
    for proposal in proposals:
        earlier = previous.get(proposal["cell_id"], {})
        rows.append(
            {
                "cell_id": proposal["cell_id"],
                "doc_id": proposal["doc_id"],
                "category_id": proposal["category_id"],
                "category_name": proposal["category_name"],
                "attention_tier": proposal["attention_tier"],
                "proposal_type": proposal["proposal_type"],
                "proposed_spans": [a["text"] for a in proposal["answers"]],
                "rationale": proposal["rationale"],
                # Fill this in. One of ACCEPT / EMPTY / EDIT.
                "verdict": earlier.get("verdict"),
                "corrected_spans": earlier.get("corrected_spans", []),
                "reviewer_notes": earlier.get("reviewer_notes", ""),
            }
        )

    write_jsonl(WORKSHEET, rows)
    pending = sum(1 for row in rows if row["verdict"] is None)
    stage.info(f"wrote {WORKSHEET.name} ({len(rows)} rows)")
    stage.info(f"carried forward {kept} existing verdict(s); {pending} awaiting one")
    stage.info("set 'verdict' on every row, then re-run with --promote --reviewer '...'")
    return 0


def promote(stage: Stage, reviewer: str, partial: bool = False) -> int:
    rows = read_jsonl(WORKSHEET)
    if not rows:
        stage.fail(f"no worksheet at {WORKSHEET} — run without --promote first.")
        return 1

    proposals = {p["cell_id"]: p for p in read_jsonl(PROPOSED)}
    cells = {c["cell_id"]: dict(c) for c in read_jsonl(ANNOTATIONS_DIR / "annotations_raw.jsonl")}
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")

    problems: list[str] = []
    pending = [r["cell_id"] for r in rows if r.get("verdict") is None]
    if pending and not partial:
        stage.fail(f"{len(pending)} of {len(rows)} rows still have verdict=null")
        stage.info(f"first few: {pending[:8]}")
        stage.info("pass --partial to promote the reviewed rows and leave the rest.")
        return 1
    if pending:
        stage.warn(
            f"{len(pending)} row(s) have no verdict and stay unannotated: {pending[:6]}"
            f"{' ...' if len(pending) > 6 else ''}"
        )

    counts = {v: 0 for v in VERDICTS}
    for row in rows:
        if row.get("verdict") is None:
            continue
        cell_id = row["cell_id"]
        verdict = row.get("verdict")
        proposal = proposals.get(cell_id)
        cell = cells.get(cell_id)

        if verdict not in VERDICTS:
            problems.append(f"{cell_id}: verdict {verdict!r} not in {sorted(VERDICTS)}")
            continue
        if proposal is None or cell is None:
            problems.append(f"{cell_id}: no matching proposal or workspace cell")
            continue

        if verdict == ACCEPT and proposal["proposal_type"] == "NO_CANDIDATE_FOUND":
            problems.append(
                f"{cell_id}: cannot ACCEPT a NO_CANDIDATE_FOUND row — the model made no "
                "finding. Use EMPTY to confirm the cell empty yourself (Appendix A rule 1), "
                "or EDIT if you found a clause."
            )
            continue

        if verdict == EMPTY:
            answers, gold = [], []
        elif verdict == ACCEPT:
            answers = list(proposal["answers"])
            gold = list(proposal["gold_paragraph_ids"])
        else:  # EDIT
            quotes = row.get("corrected_spans") or []
            if not quotes:
                problems.append(f"{cell_id}: EDIT requires corrected_spans")
                continue
            answers = []
            failed = False
            for quote in quotes:
                try:
                    resolved = build_answer(
                        cell["doc_id"],
                        quote,
                        paragraphs,
                        allow_furniture=cell["category_id"] == MASTHEAD_CATEGORY,
                    )
                except SpanError as error:
                    problems.append(f"{cell_id}: {error}")
                    failed = True
                    continue
                answers.append({k: v for k, v in resolved.items() if k != "paragraph_ids"})
            if failed:
                continue
            gold = sorted({a["paragraph_id"] for a in answers})

        counts[verdict] += 1
        cell.update(
            {
                "is_impossible": verdict == EMPTY,
                "answers": answers,
                "gold_paragraph_ids": gold,
                "status": "REVIEWED",
                "annotator_ids": sorted(set(cell["annotator_ids"]) | {reviewer}),
                "annotation_pass": "P1-model-assisted",
                # True even when the human overrode the model: the cell was
                # reached with model assistance and the datasheet must say so.
                "model_assisted": True,
                "annotator_notes": row.get("reviewer_notes", ""),
                # Provenance, and the field the blind subsample draws its pool
                # from. A cell the reviewer read one at a time is "human"; a
                # cell approved as part of a bulk pass under amendment A-1 is
                # not, whoever's name is on it. Leaving this unset made every
                # promoted cell invisible to blind_subsample.py, so the sample
                # could only ever contain empties — it could measure the model
                # missing a clause but never whether a span it proposed was
                # wrong.
                "verdict_source": (
                    "human" if row.get("reviewed_individually")
                    else "model_proposed_human_approved"
                ),
            }
        )
        cells[cell_id] = cell

    if problems:
        for problem in problems[:25]:
            stage.fail(problem)
        if len(problems) > 25:
            stage.info(f"... and {len(problems) - 25} more")
        return 1

    # Promotion rebuilds every answer from the worksheet's quoted text, which
    # cannot express "this answer is two pieces because the clause crosses a
    # paragraph boundary". So a re-promotion silently undoes
    # split_straddling_spans.py and puts SPAN_OUTSIDE_PARAGRAPH findings back.
    # That happened once and was caught only by an audit. Re-splitting here
    # closes the loop: the splitter is idempotent, so this is a no-op when
    # there is nothing to split.
    write_jsonl(REVIEWED, list(cells.values()))
    _resplit_straddling_answers(stage)

    reviewed = sum(1 for c in cells.values() if c["status"] == "REVIEWED")
    positives = sum(1 for c in cells.values() if c["is_impossible"] is False)
    stage.info(f"reviewer: {reviewer}")
    stage.info(f"verdicts: " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))
    stage.info(f"cells now carrying a human verdict: {reviewed}/{len(cells)}")
    stage.info(f"wrote {REVIEWED.name} ({positives} positive cells)")
    stage.info(
        "remaining cells stay is_impossible=null; build_qa.py will refuse until "
        "a human annotates them."
    )

    emit_report(
        "review_report.json",
        {
            "status": "PASS",
            "reviewer": reviewer,
            "rows_reviewed": len(rows),
            "verdicts": counts,
            "cells_with_verdict": reviewed,
            "cells_total": len(cells),
            "positive_cells": positives,
            "note": (
                "Every is_impossible=true verdict here was set by a human, per "
                "Appendix A rule 1. model_assisted=true is recorded on every cell "
                "the model touched, including those the human overrode."
            ),
        },
    )
    return 0



def _resplit_straddling_answers(stage: Stage) -> None:
    """Re-cut answers that cross a paragraph boundary (protocol §6 rule 7)."""
    try:
        from split_straddling_spans import main as split_main
    except ImportError:
        stage.info("split_straddling_spans.py not importable — skipping re-split")
        return
    import sys as _sys
    argv = _sys.argv
    _sys.argv = ["split_straddling_spans.py", "--apply"]
    try:
        split_main()
    finally:
        _sys.argv = argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--reviewer", help="Name, Role — recorded in the datasheet")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="promote the rows that have a verdict; leave the rest unannotated",
    )
    args = parser.parse_args()

    stage = Stage("A/R", "Proposal review")
    if not args.promote:
        return emit_worksheet(stage)
    if not args.reviewer:
        stage.fail("--promote requires --reviewer 'Name, Role'")
        return 1
    return promote(stage, args.reviewer, args.partial)


if __name__ == "__main__":
    sys.exit(main())
