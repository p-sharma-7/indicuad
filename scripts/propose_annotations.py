"""Model-assisted annotation proposals — protocol Appendix A.

Reads `annotations/proposals.jsonl` (authored by the assisting model),
resolves every quoted span to offsets in the frozen corpus, and writes
`annotations/annotations_proposed.jsonl`.

The output file is deliberately OUTSIDE the chain `build_qa.py` reads
(`annotations_adjudicated` -> `annotations_reviewed` -> `annotations_raw`).
A proposal therefore cannot become gold without a human explicitly
promoting it. As a second guard, every proposal carries
`is_impossible: null`, so even if one were copied into the chain by
mistake, `build_qa.py` would still refuse to ship it.

Appendix A's non-negotiable rules, enforced here rather than left to the
assisting model's judgement:

  Rule 1  No LLM-generated negatives, ever. This script cannot emit
          `is_impossible: true`. A cell where the model found nothing is
          recorded as NO_CANDIDATE_FOUND, which is a recall note, not a
          verdict — a human must still confirm the cell empty. The
          negatives are the instrument for measuring hallucination under
          quantization; a model's guess entering gold would reward the
          evaluated system for the exact failure the benchmark detects.
  Rule 3  Primary labelling is confined to the low-friction categories.
          For moderate and high-contest categories the model may only
          raise RECALL_CANDIDATE — "look here", never "this is the span".
  Rule 4  Everything lands as status=PROPOSED, model_assisted=true.

Output: annotations/annotations_proposed.jsonl
"""

from __future__ import annotations

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

# Recorded in the datasheet per Appendix A rule 2 (annotation model must not
# be from the evaluated system's family — the evaluated system is Llama-3.1).
PROPOSED_BY = "claude-opus-5 (Anthropic) via Claude Code"

PRIMARY_LABEL = "PRIMARY_LABEL"
RECALL_CANDIDATE = "RECALL_CANDIDATE"
NO_CANDIDATE_FOUND = "NO_CANDIDATE_FOUND"
PROPOSAL_TYPES = {PRIMARY_LABEL, RECALL_CANDIDATE, NO_CANDIDATE_FOUND}


def main() -> int:
    stage = Stage("A", "Model-assisted annotation proposals")

    source = ANNOTATIONS_DIR / "proposals.jsonl"
    proposals = read_jsonl(source)
    if not proposals:
        stage.fail(f"no proposals found at {source}")
        return 1

    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    categories = {c["category_id"]: c for c in read_jsonl(CORPUS_DIR / "categories.jsonl")}
    cells = {c["cell_id"]: c for c in read_jsonl(ANNOTATIONS_DIR / "annotations_raw.jsonl")}

    records: list[dict] = []
    errors: list[str] = []
    counts = {t: 0 for t in PROPOSAL_TYPES}

    for proposal in proposals:
        cell_id = proposal.get("cell_id", "<missing>")
        cell = cells.get(cell_id)
        if cell is None:
            errors.append(f"{cell_id}: not a cell in the annotation workspace")
            continue

        kind = proposal.get("proposal_type")
        if kind not in PROPOSAL_TYPES:
            errors.append(f"{cell_id}: proposal_type {kind!r} not in {sorted(PROPOSAL_TYPES)}")
            continue

        if "is_impossible" in proposal:
            errors.append(
                f"{cell_id}: proposals must not carry a verdict (Appendix A rule 1)"
            )
            continue

        tier = categories[cell["category_id"]]["attention_tier"]
        if kind == PRIMARY_LABEL and tier != "low_friction":
            errors.append(
                f"{cell_id}: PRIMARY_LABEL not permitted for a {tier} category "
                "(Appendix A rule 3) — use RECALL_CANDIDATE"
            )
            continue

        answers = []
        failed = False
        for span in proposal.get("spans", []):
            quote = span if isinstance(span, str) else span.get("text", "")
            occurrence = None if isinstance(span, str) else span.get("occurrence")
            try:
                answers.append(
                    build_answer(
                        cell["doc_id"],
                        quote,
                        paragraphs,
                        occurrence,
                        allow_furniture=cell["category_id"] == MASTHEAD_CATEGORY,
                    )
                )
            except SpanError as error:
                errors.append(f"{cell_id}: {error}")
                failed = True
        if failed:
            continue

        if kind != NO_CANDIDATE_FOUND and not answers:
            errors.append(f"{cell_id}: {kind} requires at least one span")
            continue

        counts[kind] += 1
        records.append(
            {
                "cell_id": cell_id,
                "doc_id": cell["doc_id"],
                "category_id": cell["category_id"],
                "category_name": cell["category_name"],
                "attention_tier": tier,
                "split": cell["split"],
                "proposal_type": kind,
                # Never a verdict. A human decides is_impossible.
                "is_impossible": None,
                "answers": [
                    {k: v for k, v in a.items() if k != "paragraph_ids"} for a in answers
                ],
                "gold_paragraph_ids": sorted({a["paragraph_id"] for a in answers}),
                "rationale": proposal.get("rationale", ""),
                "proposed_by": PROPOSED_BY,
                "model_assisted": True,
                "status": "PROPOSED",
                "human_verdict": None,
                "human_reviewer": None,
            }
        )

    if errors:
        for error in errors[:25]:
            stage.fail(error)
        if len(errors) > 25:
            stage.info(f"... and {len(errors) - 25} more")
        emit_report(
            "proposal_report.json",
            {"status": "FAILED", "errors": errors, "proposals": len(proposals)},
        )
        return 1

    write_jsonl(ANNOTATIONS_DIR / "annotations_proposed.jsonl", records)

    covered = {r["cell_id"] for r in records}
    core_cells = {c["cell_id"] for c in cells.values() if c["split"] == "core"}

    stage.info(f"proposals validated: {len(records)}")
    for kind in (PRIMARY_LABEL, RECALL_CANDIDATE, NO_CANDIDATE_FOUND):
        stage.info(f"  {kind:<20} {counts[kind]}")
    stage.info(f"core cells covered: {len(covered & core_cells)}/{len(core_cells)}")
    stage.info("wrote annotations/annotations_proposed.jsonl")
    stage.info("NOT gold — a human must review every record before it counts.")

    emit_report(
        "proposal_report.json",
        {
            "status": "PASS",
            "proposed_by": PROPOSED_BY,
            "records": len(records),
            "by_type": counts,
            "core_cells_covered": len(covered & core_cells),
            "core_cells_total": len(core_cells),
            "spans_resolved": sum(len(r["answers"]) for r in records),
            "verdicts_written": 0,
            "note": (
                "Proposals are not annotations. This file is outside the chain "
                "build_qa.py reads, and every record carries is_impossible=null. "
                "Appendix A rule 1 reserves all is_impossible=true verdicts to a "
                "human; rule 3 confines primary labelling to low-friction "
                "categories. NO_CANDIDATE_FOUND is a recall note, not an empty "
                "verdict."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
