"""Stage 17 / §25 — build HUMAN_REVIEW_REQUIRED.md.

Collects everything the pipeline refused to decide on its own. The queue
holds only items that deterministic validation cannot settle safely: if a
check could resolve it mechanically, it belongs in repairs.py instead.

Output: HUMAN_REVIEW_REQUIRED.md
"""

from __future__ import annotations

import sys

from indicuad_common import (
    CORPUS_DIR,
    PROJECT_ROOT,
    Stage,
    find_protocol,
    read_json,
    read_jsonl,
    utc_now,
    VALIDATION_DIR,
)


DECISIONS_PATH = PROJECT_ROOT / "decisions.json"


def load_decisions() -> dict:
    """Recorded resolutions, keyed by review-item key.

    A resolution is a durable statement that a human (or a human-delegated
    run) considered the item and decided it. Items whose key appears here
    move to the Resolved section instead of blocking, and the recorded
    rationale travels with them so the decision stays auditable.
    """
    payload = read_json(DECISIONS_PATH) or {}
    return payload.get("resolved", {})


class Queue:
    def __init__(self, decisions: dict | None = None) -> None:
        self.items: list[dict] = []
        self.decisions = decisions or {}

    def add(self, *, key, title, blocking, docs, issue, reason, action, evidence=None):
        decision = self.decisions.get(key)
        self.items.append(
            {
                "id": f"HR-{len(self.items) + 1:03d}",
                "key": key,
                "title": title,
                # A resolved item is never blocking, whatever it was before.
                "blocking": blocking and decision is None,
                "resolved": decision is not None,
                "decision": decision,
                "documents": docs,
                "issue": issue,
                "reason": reason,
                "action": action,
                "evidence": evidence or [],
            }
        )


def build_queue() -> Queue:
    queue = Queue(load_decisions())

    protocol = find_protocol()
    if protocol and protocol.name != "ANNOTATION_PROTOCOL.md":
        queue.add(
            key="protocol_filename",
            title="Protocol filename does not match the specification",
            blocking=False,
            docs=[],
            issue=f"The authoritative protocol is on disk as {protocol.name!r}, "
                  "but every document and script refers to ANNOTATION_PROTOCOL.md.",
            reason="Renaming the frozen protocol is a human decision, and copying it "
                   "would create two divergent authoritative files.",
            action="Rename the file to ANNOTATION_PROTOCOL.md. The pipeline resolves "
                   "either name, so nothing breaks either way.",
        )

    # --- tiering -----------------------------------------------------------
    triage = read_json(VALIDATION_DIR / "triage_report.json")
    if triage:
        rows = {r["doc_id"]: r for r in triage.get("rows", [])}

        # Judge by the corpus itself, not by the proposal report's status —
        # triage_documents.py always writes its output as a proposal, so its
        # status stays PENDING even after apply_triage.py has run.
        corpus_documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
        untiered = [d["doc_id"] for d in corpus_documents if not d.get("split")]

        if untiered:
            queue.add(
                key="triage_pending",
                title="Tier and split assignment awaiting human review",
                blocking=True,
                docs=[],
                issue=f"{len(untiered)} of {len(corpus_documents)} documents still "
                      "have tier=null and split=null in corpus/documents.jsonl.",
                reason="Protocol §3 requires tier assignment to be human-reviewed "
                       "before P1, and the core split must be chosen before "
                       "annotation begins rather than derived from how documents "
                       "turn out to annotate.",
                action="Review triage_report.csv, fill human_tier / human_split / "
                       "reviewer, then run scripts/apply_triage.py.",
                evidence=[
                    f"proposed tiers: {triage['proposed_tier_counts']}",
                    f"Tier A/B core candidates ({len(triage['core_candidates'])}): "
                    f"{', '.join(triage['core_candidates'])}",
                    "Protocol §2 wants 6-8 of these in the core split; the remainder "
                    "should be marked human_split=excluded.",
                ],
            )

        for doc_id in triage.get("tier_c_operative_content_flagged", []):
            row = rows.get(doc_id, {})
            queue.add(
                key=f"tier_c_operative:{doc_id}",
                title=f"TIER_C_OPERATIVE_CONTENT — {doc_id}",
                blocking=True,
                docs=[doc_id],
                issue=f"{row.get('filename')} is filed as a Tier C disclosure but "
                      "carries a limited number of genuine obligations — typically an "
                      "acceptance letter or work order rather than a full contract.",
                reason="Tier C documents are annotated by confirm-empty skim. A "
                       "document with real obligations will not be empty across all "
                       "26 categories, so marking it empty would inject false "
                       "negatives into the abstention probe (protocol §5 P4).",
                action=f"Decide whether {doc_id} is re-tiered to A/B and routed "
                       "through P1, or excluded from annotation and kept in the "
                       "retrieval index only.",
                evidence=[
                    f"{row.get('shall_count')} occurrences of 'shall', "
                    f"{row.get('operative_phrases')} operative phrases",
                    "The other 22 Tier C documents score 0-1 on both measures.",
                ],
            )

        for doc_id in triage.get("unverifiable_scanned_pages_flagged", []):
            queue.add(
                key=f"unverifiable_scan:{doc_id}",
                title=f"UNVERIFIABLE_SCANNED_PAGES — {doc_id}",
                blocking=True,
                docs=[doc_id],
                issue="This Tier C document still contains pages with no readable "
                      "text.",
                reason="Tier C is confirmed empty by reading the document. Where a "
                       "page is an unreadable scan the text layer looks empty "
                       "precisely where the content is, so the skim cannot be "
                       "trusted.",
                action="OCR the pages (scripts/ocr_pages.py) or inspect them "
                       "visually before confirming the document empty.",
            )

        for doc_id in triage.get("mis_tiered_flagged", []):
            row = rows.get(doc_id, {})
            queue.add(
                key=f"mis_tier:{doc_id}",
                title=f"MIS-TIERED_DOCUMENT — {doc_id}",
                blocking=True,
                docs=[doc_id],
                issue=f"{row.get('filename')} is an NSE Reg-30 filing, which would "
                      "normally be Tier C (negatives, confirm-empty). Its text "
                      "contains a full operative contract body.",
                reason="Protocol §5 P4: a Tier C document that is not actually empty "
                       "is misfiled and must be re-tiered to A/B and routed through "
                       "P1. Leaving it in the negatives split would put real clauses "
                       "into the abstention probe as false negatives.",
                action=f"Decide whether {doc_id} becomes a Tier A core document, or "
                       "whether the disclosure letter and the attached contract are "
                       "split into two documents. Splitting changes segmentation, so "
                       "decide BEFORE paragraph IDs are frozen.",
                evidence=[
                    f"{row.get('shall_count')} occurrences of 'shall' "
                    f"(every other Tier C document has 0-1)",
                    f"{row.get('clause_start_lines')} numbered clause lines",
                    f"{row.get('pages')} pages, {row.get('chars')} characters",
                    "BM25 smoke query 'arbitration seat and governing law' ranks this "
                    "document top of the whole corpus.",
                ],
            )

        for doc_id in triage.get("compound_document_flagged", []):
            row = rows.get(doc_id, {})
            queue.add(
                key=f"compound:{doc_id}",
                title=f"COMPOUND_DOCUMENT — {doc_id}",
                blocking=True,
                docs=[doc_id],
                issue=f"{row.get('filename')} is {row.get('pages')} pages of scheme "
                      "guidelines with model agreements annexed. It is being treated "
                      "as one document.",
                reason="Guidelines text is closer to Tier D (never annotated) while "
                       "the annexed model agreements are Tier A contract material. "
                       "Annotating them as a single document mixes the two, and the "
                       "protocol's own illustrative table in §2 lists a 'PM Surya Ghar "
                       "Vendor Agmt' as a document in its own right.",
                action="Decide whether to (a) split the annexed model agreements into "
                       "separate documents, (b) annotate the whole file as Tier A, or "
                       "(c) exclude it from the core split. Option (a) changes "
                       "segmentation and must happen before paragraph IDs are frozen.",
                evidence=[f"{row.get('paragraphs')} paragraphs, {row.get('chars')} characters"],
            )

    # --- OCR ---------------------------------------------------------------
    ocr = read_json(VALIDATION_DIR / "ocr_report.json")
    if ocr:
        recovered = [e for e in ocr["documents"] if e.get("ocr_pages_recovered")]
        if recovered:
            applied = read_json(VALIDATION_DIR / "ocr_applied_report.json") or {}
            confidences = [
                f"{d['doc_id']} p{p['page']}: confidence {p['confidence']:.3f}, "
                f"{p['chars']} chars"
                for d in applied.get("documents", []) for p in d["pages"]
            ]
            queue.add(
                key="ocr_quality",
                title="OCR text has not been manually corrected",
                blocking=True,
                docs=[e["doc_id"] for e in recovered],
                issue="Pages recovered by OCR are in the corpus at ~0.86 mean "
                      "confidence. Spot-checking shows systematic word-space loss "
                      "('GovernmentofIndia') and occasional digit confusion "
                      "('6o' for '60').",
                reason="Protocol §10's pre-annotation checklist requires scanned "
                       "material to be OCR'd AND manually corrected. Gold spans must "
                       "match the corpus text verbatim, so uncorrected OCR means "
                       "annotators would select malformed text.",
                action="Either manually correct the OCR'd pages and re-run "
                       "normalize/segment, or confirm these documents are excluded "
                       "from annotation and used for retrieval only.",
                evidence=confidences,
            )

    if ocr and ocr.get("documents_needing_ocr"):
        for entry in ocr["documents"]:
            if not entry.get("ocr_outstanding_pages"):
                continue
            queue.add(
                key=f"ocr:{entry['doc_id']}",
                title=f"Unresolved OCR — {entry['doc_id']} page(s) "
                      f"{entry['ocr_outstanding_pages']}",
                blocking=True,
                docs=[entry["doc_id"]],
                issue=f"{entry['filename']}: page(s) "
                      f"{entry['ocr_outstanding_pages']} are images with no "
                      "extractable text, and OCR did not recover any.",
                reason="Spec §8: a scanned page must never be silently emitted as "
                       "empty. Any clause on these pages is currently invisible to "
                       "both annotators and the retriever, which would show up as a "
                       "false negative rather than as missing data.",
                action="Either OCR the pages and re-run extraction, or confirm the "
                       "pages carry no annotatable content (a cover image or a "
                       "signature scan) and record that decision here.",
                evidence=[f"extraction mode: {entry['text_extraction']}",
                          f"{entry['page_count']} pages total"],
            )

    # --- text-layer defects -------------------------------------------------
    normalization = read_json(VALIDATION_DIR / "normalization_report.json")
    if normalization:
        homoglyph_docs = [
            d for d in normalization["documents"] if d.get("homoglyph_word_count")
        ]
        if homoglyph_docs:
            queue.add(
                key="homoglyphs",
                title="Homoglyph contamination in the source text layer",
                blocking=False,
                docs=[d["doc_id"] for d in homoglyph_docs],
                issue="Words mixing Latin with Cyrillic or Greek characters appear in "
                      "the extracted text, e.g. 'SEBI' written with a Cyrillic В and І.",
                reason="These read correctly to a human but will not match a "
                       "plain-Latin keyword sweep or a retrieval query, so the P2 "
                       "sweep can miss clauses in these documents. Rewriting them "
                       "would alter legal text, so the pipeline flags rather than fixes.",
                action="Confirm whether to leave them verbatim (safest, and the "
                       "current behaviour) or to record a documented normalization "
                       "rule. If left as-is, note the limitation in the datasheet.",
                evidence=[
                    f"{d['doc_id']}: {d['homoglyph_word_count']} word(s) "
                    f"e.g. {d['homoglyph_words'][:3]}"
                    for d in homoglyph_docs
                ],
            )

        control_docs = [d for d in normalization["documents"] if d.get("control_chars")]
        for entry in control_docs:
            queue.add(
                key=f"encoding:{entry['doc_id']}",
                title=f"Corrupt character encoding — {entry['doc_id']}",
                blocking=False,
                docs=[entry["doc_id"]],
                issue="The text layer contains NUL and C1 control characters, from a "
                      "broken ToUnicode mapping on the Devanagari portions of a "
                      "bilingual GeM contract.",
                reason="The English text extracted cleanly, but the Hindi text is "
                       "partially garbled. If any operative clause exists only in the "
                       "Hindi column it is unreadable.",
                action="Confirm the English text is sufficient for annotation, or "
                       "re-extract with a Devanagari-aware pipeline.",
                evidence=[f"control characters present: {entry['control_chars']}"],
            )

    # --- categories -----------------------------------------------------------
    categories = read_jsonl(CORPUS_DIR / "categories.jsonl")
    pending = [c for c in categories if c.get("review_status") != "APPROVED"]
    if pending:
        queue.add(
            key="categories_legal_review",
            title="categories.jsonl requires legal review before P1",
            blocking=True,
            docs=[],
            issue=f"{len(pending)} of {len(categories)} categories are still marked "
                  "DRAFT_PENDING_LEGAL_REVIEW.",
            reason="Protocol §7 requires a lawyer or law-faculty reviewer to read the "
                   "26 descriptions and boundary rules BEFORE P1, because a corrected "
                   "guideline propagates to every future cell while a spot-check only "
                   "fixes what it touches. Boundary rules were transcribed from the "
                   "protocol, but descriptions, the three question registers, negative "
                   "examples and sweep terms were drafted by this pipeline.",
            action="Have the reviewer read corpus/categories.jsonl, amend as needed, "
                   "and set review_status to APPROVED with reviewed_by / reviewed_at.",
            evidence=[
                "Category IDs are anchored to the protocol's worked examples: "
                "CAT-008 Non-Compete, CAT-017 Payment Terms & Schedule, "
                "CAT-022 Warranty Duration.",
                "Confirm the remaining 23 ID assignments are acceptable — the protocol "
                "does not publish a full ID table.",
            ],
        )

    # --- validator findings needing judgement --------------------------------
    # Validator findings are grouped by code. One queue item per affected
    # cell would bury the handful of genuine decisions under hundreds of
    # near-identical entries.
    for name in ("span_report.json", "query_consistency_report.json"):
        report = read_json(VALIDATION_DIR / name)
        if not report:
            continue
        grouped: dict[str, list[dict]] = {}
        for finding in report.get("findings", []):
            if finding.get("error_class") != "NEEDS_HUMAN_REVIEW":
                continue
            grouped.setdefault(finding["code"], []).append(finding)

        for code, findings in grouped.items():
            cells = [f.get("cell") for f in findings if f.get("cell")]
            queue.add(
                key=f"validator:{code}",
                title=f"{code} — {len(findings)} cell(s)",
                blocking=True,
                docs=[],
                issue=findings[0]["message"]
                      + (f" (and {len(findings) - 1} more)" if len(findings) > 1 else ""),
                reason="Flagged by deterministic validation as requiring semantic "
                       "judgement; the pipeline does not resolve these.",
                action="Annotator or adjudicator decision required.",
                evidence=[f"affected cells: {', '.join(cells[:15])}"
                          + (" ..." if len(cells) > 15 else "")] if cells else [],
            )

    return queue


def render(queue: Queue) -> str:
    resolved = [i for i in queue.items if i["resolved"]]
    blocking = [i for i in queue.items if i["blocking"]]
    advisory = [i for i in queue.items if not i["blocking"] and not i["resolved"]]

    lines = [
        "# Human Review Required",
        "",
        f"_Generated {utc_now()} by `scripts/generate_review_queue.py`._",
        "",
        f"**{len(blocking)} blocking**, {len(advisory)} advisory, "
        f"{len(resolved)} resolved. "
        "Blocking items must be resolved before annotation (P1) begins.",
        "",
        "These are the decisions the pipeline deliberately declined to make. "
        "Anything resolvable by deterministic rule was repaired automatically and "
        "logged to `validation/repair_log.json` instead. Resolutions are recorded "
        "in `decisions.json`.",
        "",
        "---",
        "",
    ]

    if resolved:
        lines += ["## Resolved", ""]
        for item in resolved:
            decision = item["decision"] or {}
            lines += [
                f"### {item['id']} — {item['title']}  ✅",
                "",
                f"**Decision.** {decision.get('decision', '(no rationale recorded)')}",
                "",
                f"_Decided by {decision.get('decided_by', 'unknown')} "
                f"on {decision.get('decided_at', 'unknown date')}._",
                "",
                "---",
                "",
            ]

    for heading, items in (("Blocking", blocking), ("Advisory", advisory)):
        if not items:
            continue
        lines += [f"## {heading}", ""]
        for item in items:
            lines += [
                f"### {item['id']} — {item['title']}",
                "",
            ]
            if item["documents"]:
                lines += [f"**Documents:** {', '.join(item['documents'])}", ""]
            lines += [
                f"**Issue.** {item['issue']}",
                "",
                f"**Why this needs a human.** {item['reason']}",
                "",
                f"**Action.** {item['action']}",
                "",
            ]
            if item["evidence"]:
                lines += ["**Evidence.**", ""]
                lines += [f"- {e}" for e in item["evidence"]]
                lines += [""]
            lines += ["---", ""]

    if not queue.items:
        lines += ["No outstanding human review items.", ""]

    return "\n".join(lines)


def main() -> int:
    stage = Stage("17", "Human review queue")
    queue = build_queue()

    path = PROJECT_ROOT / "HUMAN_REVIEW_REQUIRED.md"
    path.write_text(render(queue), encoding="utf-8", newline="\n")

    blocking = sum(1 for i in queue.items if i["blocking"])
    resolved = sum(1 for i in queue.items if i["resolved"])
    stage.info(
        f"wrote HUMAN_REVIEW_REQUIRED.md — {len(queue.items)} item(s), "
        f"{blocking} blocking, {resolved} resolved"
    )
    for item in queue.items:
        if item["resolved"]:
            marker = "done "
        elif item["blocking"]:
            marker = "BLOCK"
        else:
            marker = "note "
        stage.info(f"  [{marker}] {item['id']}  {item['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
