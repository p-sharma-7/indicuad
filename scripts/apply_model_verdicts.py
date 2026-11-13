"""Apply model-proposed verdicts in bulk — protocol Appendix A, amendment A-1.

Under the frozen protocol an assisting model could propose spans but never
decide a cell empty. Amendment A-1 relaxes that: the model may propose
`is_impossible: true`, and the project lead approves in bulk.

Every cell written here records `verdict_source:
"model_proposed_human_approved"`, distinct from `"human"`. That separation
is the point — the datasheet has to be able to say how many cells a person
actually judged, and reporting a model-approved corpus as human-annotated
would be false.

Two groups are written:

  negatives   Every cell of a Tier C disclosure. These documents announce a
              contract rather than containing one; prescreen_negatives.py
              found no operative content in any of the 22.
  core        Only cells the model examined and found nothing for
              (NO_CANDIDATE_FOUND). Cells with a proposed span are left
              alone — those go through review_proposals.py, where a human
              can see what is being claimed.

What this cannot fix, and what the blind subsample exists to measure: if
the model missed a clause, this writes that miss into gold. Run
blind_subsample.py before trusting these numbers.

    apply_model_verdicts.py --approver "Name, Role" [--negatives] [--core]
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

SOURCE_MODEL = "claude-opus-5 (Anthropic) via Claude Code"
VERDICT_SOURCE = "model_proposed_human_approved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approver", required=True, help="Name, Role")
    parser.add_argument("--negatives", action="store_true", help="all Tier C cells")
    parser.add_argument("--core", action="store_true",
                        help="core cells the model found nothing for")
    args = parser.parse_args()

    if not (args.negatives or args.core):
        parser.error("choose --negatives, --core, or both")

    stage = Stage("A-1", "Apply model-proposed verdicts")

    path = current_annotations_path()
    cells = read_jsonl(path)
    if not cells:
        stage.fail(f"no annotation cells at {path}")
        return 1

    proposals = {p["cell_id"]: p for p in
                 read_jsonl(ANNOTATIONS_DIR / "annotations_proposed.jsonl")}

    written = {"negatives": 0, "core": 0}
    skipped_has_span = 0
    skipped_no_proposal = 0

    for cell in cells:
        if cell.get("is_impossible") is not None:
            continue

        if cell["split"] == "negatives":
            if not args.negatives:
                continue
            group = "negatives"
        elif cell["split"] == "core":
            if not args.core:
                continue
            proposal = proposals.get(cell["cell_id"])
            if proposal is None:
                skipped_no_proposal += 1
                continue
            if proposal["proposal_type"] != "NO_CANDIDATE_FOUND":
                # A proposed span is a claim about the document. It goes in
                # front of a human via review_proposals.py, not through here.
                skipped_has_span += 1
                continue
            group = "core"
        else:
            continue

        cell.update(
            {
                "is_impossible": True,
                "answers": [],
                "gold_paragraph_ids": [],
                "status": "REVIEWED",
                "annotation_pass": "P4" if group == "negatives" else "P1",
                "model_assisted": True,
                "verdict_source": VERDICT_SOURCE,
                "proposed_by": SOURCE_MODEL,
                "approved_by": args.approver,
                "annotator_ids": [SOURCE_MODEL],
                "annotator_notes": (
                    f"Empty proposed by {SOURCE_MODEL}, approved in bulk by "
                    f"{args.approver} under Appendix A amendment A-1. Not "
                    f"independently judged cell by cell."
                ),
            }
        )
        written[group] += 1

    write_jsonl(path, cells)

    decided = [c for c in cells if c.get("is_impossible") is not None]
    by_source: dict[str, int] = {}
    for cell in decided:
        by_source[cell.get("verdict_source", "human")] = (
            by_source.get(cell.get("verdict_source", "human"), 0) + 1
        )

    stage.info(f"approver: {args.approver}")
    stage.info(f"negatives cells written: {written['negatives']}")
    stage.info(f"core cells written:      {written['core']}")
    if skipped_has_span:
        stage.info(f"core cells left for human review (span proposed): {skipped_has_span}")
    if skipped_no_proposal:
        stage.warn(f"core cells with no proposal, untouched: {skipped_no_proposal}")
    stage.info(f"cells with a verdict: {len(decided)}/{len(cells)}")
    for source, count in sorted(by_source.items()):
        stage.info(f"  {source:<34} {count}")
    stage.warn(
        "Blind subsample is now REQUIRED (amendment A-1). Until it runs, the "
        "error rate on these verdicts is unmeasured."
    )

    emit_report(
        "model_verdicts_report.json",
        {
            "status": "PASS",
            "amendment": "Appendix A, A-1",
            "approver": args.approver,
            "proposed_by": SOURCE_MODEL,
            "negatives_cells": written["negatives"],
            "core_cells": written["core"],
            "left_for_human_review": skipped_has_span,
            "cells_with_verdict": len(decided),
            "cells_total": len(cells),
            "by_verdict_source": by_source,
            "blind_subsample_run": False,
            "note": (
                "These verdicts were proposed by a model and approved in bulk, not "
                "judged cell by cell. The datasheet must report by_verdict_source "
                "rather than describing the corpus as human-annotated. Correlated "
                "blind spots between annotating and evaluated model remain "
                "unmeasured until the blind subsample runs."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
