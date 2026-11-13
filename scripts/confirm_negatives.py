"""P4 — negatives-split confirmation (protocol §5 P4).

A Tier C disclosure announces that a contract was signed; it does not
contain the contract. So all 26 cells should be empty. P4 is a human
skimming each document and confirming that. If a cell turns out **not**
empty, the document is misfiled and must be re-tiered to A/B and routed
through P1 instead — which is what happened to DOC-027 and DOC-031.

    confirm_negatives.py --evidence
        Write annotations/P4_EVIDENCE.md — per document, every category
        whose sweep terms hit, with the matching text. Read this next to
        the document; it is where a missed clause would show up.

    confirm_negatives.py --doc DOC-011 --confirmer "Name, Role"
        You have skimmed DOC-011 and confirm all 26 cells empty.

    confirm_negatives.py --all --confirmer "Name, Role"
        Same, for every negatives document at once. Recorded as such,
        because "confirmed 22 documents in one command" is a different
        provenance claim from 22 separate confirmations, and the datasheet
        should be able to tell them apart.

This script never decides anything. It records a decision a human states
they have made — Appendix A rule 1 keeps every `is_impossible: true` in
human hands, because the negatives are the instrument this benchmark
measures hallucination with.
"""

from __future__ import annotations

import argparse
import sys

from sweep import compile_terms, first_match

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
    write_jsonl,
)


def sweep_hits(paragraphs: list[dict], terms: list[str]) -> list[tuple[str, str, str]]:
    """(paragraph_id, matched term, surrounding text) for real paragraphs."""
    patterns = compile_terms(terms)
    found = []
    for paragraph in paragraphs:
        if paragraph.get("is_boilerplate"):
            continue
        match = first_match(paragraph["text"], patterns)
        if match is None:
            continue
        start = max(0, match.start() - 60)
        snippet = " ".join(paragraph["text"][start:match.end() + 80].split())
        found.append((paragraph["paragraph_id"], match.group(0), snippet))
    return found


def write_evidence(stage: Stage) -> int:
    documents = [d for d in read_jsonl(CORPUS_DIR / "documents.jsonl")
                 if d.get("split") == "negatives"]
    categories = read_jsonl(CORPUS_DIR / "categories.jsonl")
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")

    by_doc: dict[str, list[dict]] = {}
    for paragraph in paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    out = ["# P4 — negatives-split evidence\n"]
    out.append(
        f"\n{len(documents)} documents × {len(categories)} categories = "
        f"**{len(documents) * len(categories)} cells**, all expected empty.\n"
    )
    out.append(
        "\nEach section lists only the categories whose sweep terms matched. A category "
        "not listed had no hit at all. A hit is **not** a clause — these disclosures "
        "quote contract values and dates, which trip payment and delivery terms — but a "
        "real clause would appear here, so this is where to look.\n"
    )
    out.append(
        "\nConfirm a document with:\n\n"
        "```bash\n"
        ".venv/Scripts/python.exe scripts/confirm_negatives.py \\\n"
        '    --doc DOC-011 --confirmer "Name, Role"\n'
        "```\n"
    )

    total_hits = 0
    for document in sorted(documents, key=lambda d: d["doc_id"]):
        doc_id = document["doc_id"]
        own = by_doc.get(doc_id, [])
        rows = []
        for category in categories:
            hits = sweep_hits(own, category["sweep_terms"])
            if hits:
                rows.append((category, hits))
        total_hits += len(rows)

        out.append(f"\n---\n\n## {doc_id} — {document['filename']}\n")
        out.append(f"\n*{document['char_count']:,} chars · {len(own)} paragraphs*\n")
        if not rows:
            out.append("\n**No sweep hit in any of the 26 categories.**\n")
            continue
        out.append(f"\n{len(rows)} of 26 categories have a sweep hit:\n")
        for category, hits in rows:
            out.append(f"\n**{category['category_id']} {category['category_name']}**\n")
            for paragraph_id, term, snippet in hits[:2]:
                out.append(
                    f"\n> …{snippet}…\n>\n> — `{paragraph_id}` · matched `{term}`\n"
                )

    path = ANNOTATIONS_DIR / "P4_EVIDENCE.md"
    path.write_text("".join(out), encoding="utf-8")
    stage.info(f"wrote {path.name}")
    stage.info(
        f"{len(documents)} documents, {total_hits} document×category pairs with any hit "
        f"out of {len(documents) * len(categories)} cells"
    )
    return 0


def confirm(stage: Stage, doc_ids: list[str], confirmer: str, bulk: bool) -> int:
    source = current_annotations_path()
    cells = read_jsonl(source)
    if not cells:
        stage.fail(f"no annotation cells at {source}")
        return 1

    negatives = {c["doc_id"] for c in cells if c.get("split") == "negatives"}
    if bulk:
        doc_ids = sorted(negatives)
    unknown = [d for d in doc_ids if d not in negatives]
    if unknown:
        stage.fail(f"not negatives-split document(s): {unknown}")
        return 1

    mode = "bulk (--all, one command for every document)" if bulk else "per document"
    confirmed, conflicts = 0, []
    for cell in cells:
        if cell["doc_id"] not in doc_ids:
            continue
        if cell.get("is_impossible") is False:
            conflicts.append(cell["cell_id"])
            continue
        cell.update(
            {
                "is_impossible": True,
                "answers": [],
                "gold_paragraph_ids": [],
                "status": "REVIEWED",
                "annotation_pass": "P4",
                "model_assisted": False,
                "annotator_ids": sorted(set(cell["annotator_ids"]) | {confirmer}),
                "annotator_notes": f"P4: confirmed empty by {confirmer} ({mode}).",
            }
        )
        confirmed += 1

    if conflicts:
        stage.fail(
            f"{len(conflicts)} cell(s) already carry a positive verdict: {conflicts[:6]}. "
            "A negatives document with a real clause is misfiled — re-tier it rather than "
            "overwriting the finding."
        )
        return 1

    write_jsonl(source, cells)
    done = sum(1 for c in cells if c.get("is_impossible") is not None)

    stage.info(f"confirmer: {confirmer}")
    stage.info(f"documents confirmed: {len(doc_ids)} ({mode})")
    stage.info(f"cells marked empty: {confirmed}")
    stage.info(f"cells now carrying a verdict: {done}/{len(cells)}")

    emit_report(
        "p4_report.json",
        {
            "status": "PASS",
            "confirmer": confirmer,
            "mode": mode,
            "documents": sorted(doc_ids),
            "cells_confirmed_empty": confirmed,
            "cells_with_verdict": done,
            "cells_total": len(cells),
            "note": (
                "P4 verdicts are human assertions recorded by this script, not model "
                "output; model_assisted is false on every one. Appendix A rule 1 "
                "reserves is_impossible=true to a human."
            ),
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="store_true")
    parser.add_argument("--doc", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--confirmer", help="Name, Role — recorded on every cell")
    args = parser.parse_args()

    stage = Stage("P4", "Negatives-split confirmation")
    if args.evidence:
        return write_evidence(stage)
    if not args.confirmer:
        stage.fail("--confirmer 'Name, Role' is required to record a confirmation")
        return 1
    if not args.doc and not args.all:
        stage.fail("give --doc DOC-XXX (repeatable) or --all")
        return 1
    return confirm(stage, args.doc, args.confirmer, args.all)


if __name__ == "__main__":
    sys.exit(main())
