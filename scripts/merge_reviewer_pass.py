"""Merge an external reviewer's edited qa export back onto the current corpus.

A reviewer was sent `qa/qa.json` and returned `qa_changed`. That export was
stale — it held 1,742 rows from a 15-core-document build, while the corpus
has since grown to 43 core documents and 3,225 rows — so the return cannot
simply be swapped in. This maps their edits onto the live data by
(doc_id, category_id), which is stable across rebuilds, rather than by
`qa_id`, which is assigned sequentially and means nothing between builds.

Three things the reviewer did that need handling rather than copying:

  1. They wrote `//` comments into the JSON, which makes it unparseable.
     Those comments are their reasoning and are preserved, not discarded.
  2. On 16 cells they set `is_impossible: true` AND supplied answers. A cell
     cannot be both empty and evidenced. Their text is kept as a reviewer
     finding, and the verdict is left alone for a human to settle.
  3. Their answers are free text — sometimes verbatim, sometimes a
     paraphrase ("Parties involved in this contract are: 1. …"). Gold spans
     must resolve to character offsets in the frozen source, so a paraphrase
     cannot become gold. Each is checked against the document and labelled.

Nothing the reviewer wrote is dropped. Anything that cannot be applied is
written to the report with the reason.

    merge_reviewer_pass.py --reviewed qa_changed --out qa/qa.json

Outputs: qa/qa.json, validation/reviewer_merge_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indicuad_common import (
    NORMALIZED_DIR,
    PROJECT_ROOT,
    QA_DIR,
    Stage,
    emit_report,
    read_jsonl,
)


def strip_line_comments(text: str) -> tuple[str, list[dict]]:
    """Remove // comments outside string literals, keeping them as data."""
    out: list[str] = []
    comments: list[dict] = []
    i, n, in_string = 0, len(text), False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            end = n if end == -1 else end
            comments.append(
                {"line": text[:i].count("\n") + 1, "text": text[i:end].strip()}
            )
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out), comments


def quote_status(doc_id: str, quote: str, cache: dict[str, str | None]) -> str:
    """Is the reviewer's text verbatim in the source, partial, or a paraphrase?"""
    if doc_id not in cache:
        path = NORMALIZED_DIR / f"{doc_id}.txt"
        cache[doc_id] = path.read_text(encoding="utf-8") if path.is_file() else None
    text = cache[doc_id]
    if text is None:
        return "source_unavailable"
    if quote in text:
        return "verbatim"
    tokens = [t.strip(".,;:()") for t in quote.split() if len(t.strip(".,;:()")) > 5]
    if tokens and sum(1 for t in tokens if t in text) >= max(1, len(tokens) // 2):
        return "partial_match"
    return "paraphrase"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", required=True, help="the returned file")
    parser.add_argument("--out", default="qa/qa.json")
    parser.add_argument("--reviewer", default="Akshay Kumar")
    args = parser.parse_args()

    stage = Stage("A/M", "Merge reviewer pass")

    source = Path(args.reviewed)
    if not source.is_file():
        stage.fail(f"no such file: {source}")
        return 1

    cleaned, comments = strip_line_comments(source.read_text(encoding="utf-8"))
    try:
        returned = json.loads(cleaned)
    except json.JSONDecodeError as error:
        stage.fail(f"{source.name} does not parse even after removing comments: {error}")
        return 1
    stage.info(f"{source.name}: {len(returned)} rows, {len(comments)} inline comment(s)")

    live = read_jsonl(QA_DIR / "qa.jsonl")
    if not live:
        stage.fail("qa/qa.jsonl is empty — run build_qa.py first")
        return 1
    live_cells = {(r["doc_id"], r["category_id"]) for r in live}

    touched = [
        r for r in returned
        if any(args.reviewer.split()[0] in a for a in (r.get("annotator_ids") or []))
    ]
    stage.info(f"rows carrying the reviewer's name: {len(touched)}")

    cache: dict[str, str | None] = {}
    applied, orphaned, conflicted = [], [], []

    for row in touched:
        key = (row["doc_id"], row["category_id"])
        findings = []
        for answer in row.get("answers") or []:
            quote = answer if isinstance(answer, str) else (answer.get("text") or "")
            if not quote:
                continue
            findings.append({"text": quote, "match": quote_status(row["doc_id"], quote, cache)})

        record = {
            "cell_id": f"{row['doc_id']}x{row['category_id']}",
            "doc_id": row["doc_id"],
            "category_id": row["category_id"],
            "reviewer": args.reviewer,
            "reviewer_verdict_is_impossible": row.get("is_impossible"),
            "reviewer_findings": findings,
            "reviewer_flagged_doubt": any(
                "doubt" in a.lower() for a in (row.get("annotator_ids") or [])
            ),
        }

        if key not in live_cells:
            record["reason"] = "document is no longer in the corpus"
            orphaned.append(record)
        elif row.get("is_impossible") and findings:
            record["reason"] = (
                "reviewer marked the cell empty but supplied text; a cell cannot be "
                "both. Verdict left unchanged, finding preserved for adjudication."
            )
            conflicted.append(record)
            applied.append(record)
        else:
            applied.append(record)

    by_cell = {r["cell_id"]: r for r in applied}
    out_rows = []
    stamped = 0
    for row in live:
        row = dict(row)
        found = by_cell.get(f"{row['doc_id']}x{row['category_id']}")
        if found and row["question_register"] == "canonical":
            ids = list(row.get("annotator_ids") or [])
            if args.reviewer not in ids:
                ids.append(args.reviewer)
            row["annotator_ids"] = ids
            row["reviewer_pass"] = {
                "reviewer": args.reviewer,
                "verdict": found["reviewer_verdict_is_impossible"],
                "findings": found["reviewer_findings"],
                "flagged_doubt": found["reviewer_flagged_doubt"],
                "note": found.get("reason", "applied as recorded"),
            }
            stamped += 1
        out_rows.append(row)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    stage.info(f"cells applicable to the live corpus : {len(applied)}")
    stage.info(f"  of those, internally contradictory: {len(conflicted)}")
    stage.info(f"cells on removed documents          : {len(orphaned)}")
    stage.info(f"rows stamped in {out_path.name}      : {stamped}")
    stage.info(f"wrote {out_path.relative_to(PROJECT_ROOT)} ({len(out_rows)} rows)")

    emit_report(
        "reviewer_merge_report.json",
        {
            "status": "NEEDS_HUMAN_REVIEW" if (conflicted or orphaned) else "PASS",
            "reviewer": args.reviewer,
            "returned_file": source.name,
            "returned_rows": len(returned),
            "live_rows": len(live),
            "reviewer_touched_cells": len(touched),
            "applied": len(applied),
            "conflicted": conflicted,
            "orphaned_documents_removed": sorted({r["doc_id"] for r in orphaned}),
            "orphaned_cells": len(orphaned),
            "orphaned_detail": orphaned,
            "inline_comments": comments,
            "note": (
                "The reviewer worked from a stale export. Their edits are mapped by "
                "(doc_id, category_id), never by qa_id, which is not stable across "
                "builds. Nothing is discarded: cells whose document has since left "
                "the corpus are recorded here in full."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
