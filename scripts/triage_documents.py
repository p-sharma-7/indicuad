"""Stage 6/7 — document tiering and core/negatives split (PROPOSAL ONLY).

The protocol requires tier assignment to be human-reviewed before P1, so
this script proposes and evidences; it does not decide. Nothing here is
written into corpus/documents.jsonl until a human approves the triage via
apply_triage.py.

Tiers (protocol §3):
    A  full multi-clause contracts                    -> core split
    B  short single-purpose instruments               -> core split
    C  Reg 30 / order disclosures, no clause body     -> negatives split
    D  statutes, MOA/AOA, Companies Act schedules     -> never annotated

The load-bearing check here is MIS-TIERED_DOCUMENT detection: several NSE
filings attach a full work order or GeM contract behind the disclosure
letter. The protocol (§5 P4) says such a document is misfiled and must be
re-tiered to A/B, so we measure operative-clause density rather than
trusting the filing type.

Outputs:
    triage_report.csv                      human-editable decision sheet
    validation/triage_report.json
"""

from __future__ import annotations

import csv
import re
import sys

from indicuad_common import (
    CORPUS_DIR,
    NORMALIZED_DIR,
    PROJECT_ROOT,
    Stage,
    emit_report,
    read_jsonl,
)

CLAUSE_START = re.compile(
    r"^\s*(?:\(\s*[a-zA-Z0-9ivxIVX]{1,4}\s*\)|[0-9]+(?:\.[0-9]+)*[.)])\s+",
    re.MULTILINE,
)

# Language that only appears in an operative contract body, not in a
# disclosure letter describing a contract.
OPERATIVE_TERMS = [
    "shall", "hereby", "hereinafter", "in witness whereof", "governing law",
    "arbitration", "indemnif", "terminate", "warrant", "liquidated damages",
    "confidential", "force majeure", "undertakes", "covenant",
]

# Tier D: statutes and constitutional company documents.
TIER_D_MARKERS = ["indiacode", "companiesact", "moa", "aoa"]

# Tier B: short single-purpose instruments, named in protocol §3.
TIER_B_TITLE_MARKERS = [
    "bankguarantee", "powerofattorney", "indemnity", "guaranteeindemnity",
    "release",
]

# Mis-tier detection uses ABSOLUTE counts, not density. Density divides by
# document length, which hides exactly the case we care about: a short
# disclosure letter with a long contract stapled behind it. Measured over
# this corpus the separation is stark — genuine Reg-30 disclosures contain
# 0-1 occurrences of "shall", an attached contract body contains dozens.
MISTIER_SHALL_COUNT = 15
MISTIER_DISTINCT_MARKERS = 4

# Weaker signal for the case the strong one misses: a Letter of Acceptance
# or work order carrying a handful of real obligations rather than a full
# clause body. Measured over the 24 Tier C documents here, the separation
# is clean — only the two documents with genuine operative content score
# 2+ on "shall" or 4+ operative phrases; the other 22 score 0-1.
WEAK_SHALL_COUNT = 2
WEAK_OPERATIVE_PHRASES = 4

OPERATIVE_PHRASES = [
    "shall", "hereby", "you are requested", "is requested to", "agrees to",
    "undertakes", "deemed to be cancelled", "performance security",
    "penalty", "liquidated",
]

# Strong markers that essentially only occur in an operative clause body.
STRONG_MARKERS = [
    "governing law", "arbitration", "indemnif", "force majeure",
    "liquidated damages", "in witness whereof", "confidential", "covenant",
]

# A Tier A document this long is probably a compound instrument (scheme
# guidelines with model agreements annexed) rather than one contract.
COMPOUND_PAGE_COUNT = 20


def analyse(document: dict, paragraphs: list[dict]) -> dict:
    text = (NORMALIZED_DIR / f"{document['doc_id']}.txt").read_text(encoding="utf-8")
    lowered = text.lower()
    chars = max(len(text), 1)

    term_hits = {t: lowered.count(t) for t in OPERATIVE_TERMS}
    total_hits = sum(term_hits.values())
    density = round(total_hits / chars * 1000, 2)
    clause_lines = len(CLAUSE_START.findall(text))

    distinct_markers = sorted(m for m in STRONG_MARKERS if m in lowered)
    operative_phrases = sum(lowered.count(p) for p in OPERATIVE_PHRASES)

    # Pages whose text came from OCR, and pages still unreadable. A Tier C
    # document with unreadable pages cannot be confirmed empty from text.
    ocr_recovered = [p for p in document.get("ocr_pages", [])]
    outstanding = [
        p["page"] for p in document.get("page_spans", []) if p.get("needs_ocr")
    ]

    return {
        "chars": len(text),
        "operative_phrases": operative_phrases,
        "ocr_recovered_pages": ocr_recovered,
        "unreadable_pages": outstanding,
        "paragraphs": len(paragraphs),
        "clause_start_lines": clause_lines,
        "operative_hits": total_hits,
        "operative_density_per_1k": density,
        "shall_count": term_hits["shall"],
        "distinct_strong_markers": len(distinct_markers),
        "strong_markers": distinct_markers,
        "top_terms": sorted(
            ((t, n) for t, n in term_hits.items() if n),
            key=lambda kv: -kv[1],
        )[:6],
    }


def propose_tier(document: dict, stats: dict) -> tuple[str, str, list[str]]:
    """Return (tier, rationale, flags). Proposal only."""
    filename = document["filename"].lower().replace("_", "").replace(" ", "")
    entity = document["source_entity"].lower()
    flags: list[str] = []

    if any(marker in filename for marker in TIER_D_MARKERS) or "india code" in entity:
        return "D", "statute / MOA-AOA / Companies Act schedule", flags

    is_nse_filing = "nse" in entity
    if is_nse_filing:
        # A Tier C document is meant to be confirmed empty by a human
        # reading it. If part of it is an unreadable scan, that skim
        # cannot be trusted — the text layer looks empty precisely where
        # the content is.
        if stats["unreadable_pages"]:
            flags.append("UNVERIFIABLE_SCANNED_PAGES")

        looks_operative = (
            stats["shall_count"] >= MISTIER_SHALL_COUNT
            or stats["distinct_strong_markers"] >= MISTIER_DISTINCT_MARKERS
        )
        weakly_operative = (
            stats["shall_count"] >= WEAK_SHALL_COUNT
            or stats["operative_phrases"] >= WEAK_OPERATIVE_PHRASES
        )
        if not looks_operative and weakly_operative:
            flags.append("TIER_C_OPERATIVE_CONTENT")
            return (
                "C",
                (
                    "NSE Reg-30 filing carrying a limited number of real "
                    f"obligations ({stats['shall_count']} 'shall', "
                    f"{stats['operative_phrases']} operative phrases) — typically an "
                    "acceptance letter or work order rather than a full contract. "
                    "Not a clause body, but unlikely to be empty across all 26 "
                    "categories, so protocol §5 P4 probably applies."
                ),
                flags,
            )
        if looks_operative:
            flags.append("MIS-TIERED_DOCUMENT")
            return (
                "C",
                (
                    "NSE Reg-30 filing, but the text carries "
                    f"{stats['shall_count']} occurrences of 'shall' and "
                    f"{stats['distinct_strong_markers']} distinct operative markers "
                    f"{stats['strong_markers']} — an operative contract body appears "
                    "to be attached behind the disclosure letter. Protocol §5 P4: "
                    "re-tier to A/B and route through P1 instead."
                ),
                flags,
            )
        return "C", "NSE Reg-30 order/contract disclosure filing", flags

    if any(marker in filename for marker in TIER_B_TITLE_MARKERS):
        return "B", "short single-purpose instrument (protocol §3 names this type)", flags

    if document["page_count"] >= COMPOUND_PAGE_COUNT:
        flags.append("COMPOUND_DOCUMENT")
        return (
            "A",
            (
                f"{document['page_count']}-page instrument — likely scheme guidelines "
                "with model agreements annexed rather than a single contract. A human "
                "must decide whether to annotate whole or split the annexed agreements "
                "into their own documents before segmentation is frozen."
            ),
            flags,
        )

    return "A", "multi-clause contract template", flags


TIER_TO_SPLIT = {"A": "core_candidate", "B": "core_candidate", "C": "negatives", "D": "excluded"}


def main() -> int:
    stage = Stage("6/7", "Document tiering and split (proposal for human review)")

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    if not documents or not paragraphs:
        stage.fail("corpus not built — run normalize_corpus.py and segment_paragraphs.py.")
        return 1

    by_doc: dict[str, list[dict]] = {}
    for paragraph in paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    # Carry forward any decisions a human already recorded. Regenerating the
    # proposal must never silently discard review work.
    csv_path = PROJECT_ROOT / "triage_report.csv"
    previous: dict[str, dict] = {}
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            previous = {r["doc_id"]: r for r in csv.DictReader(handle)}
    carried = 0

    rows = []
    for document in documents:
        stats = analyse(document, by_doc.get(document["doc_id"], []))
        tier, rationale, flags = propose_tier(document, stats)
        prior = previous.get(document["doc_id"], {})

        if document["ocr_required"]:
            flags.append("UNRESOLVED_OCR_PAGES")

        rows.append(
            {
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "source_entity": document["source_entity"],
                "pages": document["page_count"],
                "chars": stats["chars"],
                "paragraphs": stats["paragraphs"],
                "clause_start_lines": stats["clause_start_lines"],
                "shall_count": stats["shall_count"],
                "distinct_strong_markers": stats["distinct_strong_markers"],
                "operative_phrases": stats["operative_phrases"],
                "ocr_recovered_pages": ";".join(map(str, stats["ocr_recovered_pages"])),
                "operative_density_per_1k": stats["operative_density_per_1k"],
                "proposed_tier": tier,
                "proposed_split": TIER_TO_SPLIT[tier],
                "rationale": rationale,
                "flags": ";".join(flags),
                "human_tier": prior.get("human_tier", ""),
                "human_split": prior.get("human_split", ""),
                "reviewer": prior.get("reviewer", ""),
                "review_status": (
                    "HUMAN_REVIEWED" if (prior.get("human_tier") or "").strip()
                    else "PENDING_HUMAN_REVIEW"
                ),
            }
        )
        if (prior.get("human_tier") or "").strip():
            carried += 1

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tier_counts: dict[str, int] = {}
    for row in rows:
        tier_counts[row["proposed_tier"]] = tier_counts.get(row["proposed_tier"], 0) + 1

    mistiered = [r for r in rows if "MIS-TIERED_DOCUMENT" in r["flags"]]
    core_candidates = [r for r in rows if r["proposed_split"] == "core_candidate"]

    for tier in sorted(tier_counts):
        stage.info(f"proposed Tier {tier}: {tier_counts[tier]} documents")
    stage.info(f"core candidates (Tier A/B): {len(core_candidates)}  — protocol wants 6–8 selected")
    if carried:
        stage.info(f"wrote triage_report.csv ({len(rows)} rows; "
                   f"{carried} existing human decision(s) preserved)")
    else:
        stage.info(f"wrote triage_report.csv ({len(rows)} rows, all PENDING_HUMAN_REVIEW)")

    compound = [r for r in rows if "COMPOUND_DOCUMENT" in r["flags"]]
    weak = [r for r in rows if "TIER_C_OPERATIVE_CONTENT" in r["flags"]]
    unverifiable = [r for r in rows if "UNVERIFIABLE_SCANNED_PAGES" in r["flags"]]

    for row in weak:
        stage.warn(
            f"TIER_C_OPERATIVE_CONTENT {row['doc_id']} (shall={row['shall_count']}, "
            f"phrases={row['operative_phrases']}) — {row['filename']}"
        )
    for row in unverifiable:
        stage.warn(
            f"UNVERIFIABLE_SCANNED_PAGES {row['doc_id']} — cannot confirm empty from text"
        )
    for row in mistiered:
        stage.warn(
            f"MIS-TIERED_DOCUMENT {row['doc_id']} ({row['pages']}p, "
            f"shall={row['shall_count']}, markers={row['distinct_strong_markers']}) "
            f"— {row['filename']}"
        )
    for row in compound:
        stage.warn(
            f"COMPOUND_DOCUMENT {row['doc_id']} ({row['pages']}p) — {row['filename']}"
        )

    emit_report(
        "triage_report.json",
        {
            "status": "PENDING_HUMAN_REVIEW",
            "documents_total": len(rows),
            "proposed_tier_counts": tier_counts,
            "core_candidates": [r["doc_id"] for r in core_candidates],
            "mis_tiered_flagged": [r["doc_id"] for r in mistiered],
            "compound_document_flagged": [r["doc_id"] for r in compound],
            "tier_c_operative_content_flagged": [r["doc_id"] for r in weak],
            "unverifiable_scanned_pages_flagged": [r["doc_id"] for r in unverifiable],
            "mis_tier_thresholds": {
                "shall_count": MISTIER_SHALL_COUNT,
                "distinct_strong_markers": MISTIER_DISTINCT_MARKERS,
                "weak_shall_count": WEAK_SHALL_COUNT,
                "weak_operative_phrases": WEAK_OPERATIVE_PHRASES,
            },
            "protocol_expectation": {
                "core_documents": "6-8 Tier A/B",
                "negatives_documents": "~24 Tier C",
                "tier_d": "never annotated; retrieval index and hard negatives only",
            },
            "note": (
                "Proposal only. corpus/documents.jsonl still carries tier=null "
                "and split=null; run apply_triage.py after a human fills the "
                "human_tier / human_split columns."
            ),
            "rows": rows,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
