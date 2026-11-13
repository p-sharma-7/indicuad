"""P4 recall aid — find negatives-split documents that may not be empty.

Protocol §5 P4: a human skims each Tier C disclosure and confirms all 26
cells empty. If a cell turns out **not** empty the document is misfiled and
must be re-tiered to A/B and routed through P1 instead. That is the failure
this script helps catch early — it already happened twice, with DOC-027
(a full GeM contract behind a disclosure letter) and DOC-031 (a MoRTH
Letter of Acceptance).

This script only ever says "this document may carry obligations, go look".
It never says a cell is empty. Appendix A rule 1 reserves every
`is_impossible: true` to a human, because the negatives are the instrument
the benchmark measures hallucination with — a model's guess entering gold
would reward the evaluated system for exactly the failure being detected.

Scoring is deliberately crude and recall-oriented: modal obligation verbs,
operative legal nouns, and a density figure per 1,000 characters. A
disclosure letter announcing a contract scores near zero; a disclosure with
the contract itself attached does not.

Output: validation/negatives_prescreen.json
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from indicuad_common import (
    CORPUS_DIR,
    Stage,
    emit_report,
    read_jsonl,
)

# Modal obligation verbs — the load-bearing words of an operative clause.
MODALS = re.compile(r"\b(shall|must|undertakes?|covenants?|hereby agrees?)\b", re.I)

# Operative nouns that rarely appear in a bare intimation letter.
OPERATIVE = re.compile(
    r"\b(indemnif\w*|confidential\w*|arbitrat\w*|terminat\w*|warrant\w*|"
    r"liabilit\w*|governing law|force majeure|breach|penalt\w*|"
    r"liquidated damages|intellectual property)\b",
    re.I,
)

# Above this many operative signals per 1,000 chars, a "disclosure" is
# probably carrying a contract body. Tuned so the 22 known-clean filings sit
# below it and the two known misfiled documents sat well above.
DENSITY_FLAG = 1.0


def main() -> int:
    stage = Stage("P4-pre", "Negatives-split pre-screen")

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    if not documents or not paragraphs:
        stage.fail("corpus incomplete")
        return 1

    by_doc: dict[str, list[dict]] = {}
    for paragraph in paragraphs:
        if not paragraph.get("is_boilerplate"):
            by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    rows = []
    for document in documents:
        if document.get("split") != "negatives":
            continue
        doc_id = document["doc_id"]
        own = by_doc.get(doc_id, [])
        text = "\n".join(p["text"] for p in own)
        chars = max(len(text), 1)

        modals = MODALS.findall(text)
        operative = OPERATIVE.findall(text)
        density = round((len(modals) + len(operative)) * 1000 / chars, 2)

        hot = [
            {
                "paragraph_id": p["paragraph_id"],
                "signals": len(MODALS.findall(p["text"])) + len(OPERATIVE.findall(p["text"])),
                "preview": " ".join(p["text"].split())[:160],
            }
            for p in own
            if MODALS.search(p["text"]) or OPERATIVE.search(p["text"])
        ]
        hot.sort(key=lambda h: -h["signals"])

        rows.append(
            {
                "doc_id": doc_id,
                "filename": document["filename"],
                "chars": chars,
                "modal_hits": len(modals),
                "operative_hits": len(operative),
                "signal_density_per_1k": density,
                "flagged": density > DENSITY_FLAG,
                "distinct_terms": sorted({t.lower() for t in operative})[:8],
                "paragraphs_to_check": hot[:5],
            }
        )

    rows.sort(key=lambda r: -r["signal_density_per_1k"])
    flagged = [r for r in rows if r["flagged"]]

    stage.info(f"screened {len(rows)} negatives-split documents")
    for row in rows:
        mark = "  <-- CHECK" if row["flagged"] else ""
        stage.info(
            f"  {row['doc_id']}  density={row['signal_density_per_1k']:>5}  "
            f"modal={row['modal_hits']:>3}  operative={row['operative_hits']:>3}{mark}"
        )
    stage.info(f"{len(flagged)} document(s) above the {DENSITY_FLAG}/1k threshold")
    stage.info("This is a recall aid. No cell was marked empty; P4 remains human work.")

    emit_report(
        "negatives_prescreen.json",
        {
            "status": "PASS",
            "threshold_per_1k": DENSITY_FLAG,
            "documents_screened": len(rows),
            "documents_flagged": len(flagged),
            "flagged_doc_ids": [r["doc_id"] for r in flagged],
            "documents": rows,
            "verdicts_written": 0,
            "note": (
                "Recall aid for protocol §5 P4 only. Flags documents that may carry "
                "operative content and therefore be misfiled in the negatives split. "
                "Never asserts a cell is empty — Appendix A rule 1 reserves every "
                "is_impossible=true to a human."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
