"""Document serialization for the two prompting conditions (spec §6).

full_contract: exactly the frozen `work/normalized/{doc_id}.txt` text, with
at most a single deterministic, non-content `[DOCUMENT: ...]` header —
never a gold marker, distractor marker, paragraph ranking, or gold
location. Structural `[PAGE: n]` markers are available but off by default
(see `include_page_markers`) precisely because they are optional per spec
and add a behavioral variable the primary condition should not carry by
default.

oracle_local: gold paragraph(s) + adjacent non-boilerplate paragraph(s) +
resolvable distractor paragraphs, each kept as a clearly labelled, source-
attributed segment (never silently merged into one undifferentiated block,
since distractor paragraphs may come from a different document with its
own offset space). Strictly labelled `oracle_local`; never mixed with
full-contract output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.config import Config, sha256_text

FULL_CONTRACT_HEADER = "[DOCUMENT: {doc_id}]"
FULL_CONTRACT_PAGE_MARKER = "[PAGE: {page}]"


@dataclass
class SerializedDocument:
    kind: str                    # "full_contract" | "oracle_local"
    doc_id: str
    text: str
    sha256: str
    serializer_version: str
    char_count: int
    segments: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_limitations: list[dict[str, Any]] = field(default_factory=list)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "doc_id": self.doc_id, "sha256": self.sha256,
            "serializer_version": self.serializer_version, "char_count": self.char_count,
            "n_segments": len(self.segments),
            "diagnostic_limitations": self.diagnostic_limitations,
        }


def serialize_full_contract(dataset, doc_id: str, cfg: Config, include_page_markers: bool = False,
                             include_document_header: bool = False) -> SerializedDocument:
    """Default (both flags False): the model input is `work/normalized/
    {doc_id}.txt` completely byte-identical, nothing prepended — the most
    literal reading of spec §6 ("use exactly the frozen text ... as
    content"). `include_document_header` was the earlier default and is
    kept available but off by default after a real pilot run
    (qwen3_8b_fp16, IND-Q-00027, category "Document Name") showed the model
    echoing the `[DOCUMENT: DOC-015]` marker back as its answer instead of
    the real title a few lines below — a genuine confound between an
    optional bookkeeping marker and exactly the question category most
    likely to be misled by it. The model never needs the doc_id to answer;
    it is recoverable from the run record without embedding it in the
    prompt."""
    version = cfg.context.get("serializer_version", "fullctx-v1")
    raw_text = dataset.normalized_text(doc_id)

    if not include_page_markers and not include_document_header:
        return SerializedDocument(
            kind="full_contract", doc_id=doc_id, text=raw_text, sha256=sha256_text(raw_text),
            serializer_version=version, char_count=len(raw_text),
            segments=[{"doc_id": doc_id, "char_start": 0, "char_end": len(raw_text), "source": "full_document"}],
        )

    if not include_page_markers:
        text = f"{FULL_CONTRACT_HEADER.format(doc_id=doc_id)}\n\n{raw_text}"
        return SerializedDocument(
            kind="full_contract", doc_id=doc_id, text=text, sha256=sha256_text(text),
            serializer_version=version + "+doc_header", char_count=len(text),
            segments=[{"doc_id": doc_id, "char_start": 0, "char_end": len(raw_text), "source": "full_document"}],
        )

    doc_meta = dataset.documents.get(doc_id, {})
    page_spans = doc_meta.get("page_spans", [])
    if not page_spans:
        return serialize_full_contract(dataset, doc_id, cfg, include_page_markers=False)

    parts = [FULL_CONTRACT_HEADER.format(doc_id=doc_id), ""]
    cursor = 0
    for span in page_spans:
        gap = raw_text[cursor:span["char_start"]]
        if gap:
            parts.append(gap)
        parts.append(FULL_CONTRACT_PAGE_MARKER.format(page=span["page"]))
        parts.append(raw_text[span["char_start"]:span["char_end"]])
        cursor = span["char_end"]
    if cursor < len(raw_text):
        parts.append(raw_text[cursor:])
    text = "\n".join(parts)
    return SerializedDocument(
        kind="full_contract", doc_id=doc_id, text=text, sha256=sha256_text(text),
        serializer_version=version + "+page_markers", char_count=len(text),
        segments=[{"doc_id": doc_id, "char_start": 0, "char_end": len(raw_text), "source": "full_document_with_page_markers"}],
    )


def _adjacent_non_boilerplate(paragraphs_ordered: list[dict[str, Any]], index_in_list: int, direction: int) -> dict[str, Any] | None:
    i = index_in_list + direction
    while 0 <= i < len(paragraphs_ordered):
        p = paragraphs_ordered[i]
        if not p.get("is_boilerplate"):
            return p
        i += direction
    return None


def serialize_oracle_local(dataset, qa_row: dict[str, Any], cfg: Config) -> SerializedDocument:
    version = cfg.context.get("oracle_serializer_version", "oraclelocal-v1")
    doc_id = qa_row["doc_id"]
    gold_paragraph_ids = qa_row.get("gold_paragraph_ids", [])
    paragraphs_ordered = dataset.paragraphs_by_doc.get(doc_id, [])
    id_to_pos = {p["paragraph_id"]: i for i, p in enumerate(paragraphs_ordered)}

    seen_pids: set[str] = set()
    gold_segments: list[dict[str, Any]] = []
    for pid in gold_paragraph_ids:
        pos = id_to_pos.get(pid)
        if pos is None:
            continue
        block_pids = []
        prev_p = _adjacent_non_boilerplate(paragraphs_ordered, pos, -1)
        if prev_p and prev_p["paragraph_id"] not in seen_pids:
            block_pids.append(prev_p)
        gold_p = paragraphs_ordered[pos]
        if gold_p["paragraph_id"] not in seen_pids:
            block_pids.append(gold_p)
        next_p = _adjacent_non_boilerplate(paragraphs_ordered, pos, +1)
        if next_p and next_p["paragraph_id"] not in seen_pids:
            block_pids.append(next_p)
        for p in block_pids:
            if p["paragraph_id"] in seen_pids:
                continue
            seen_pids.add(p["paragraph_id"])
            gold_segments.append({
                "role": "gold_context", "doc_id": doc_id, "paragraph_id": p["paragraph_id"],
                "char_start": p["char_start"], "char_end": p["char_end"], "text": p["text"],
            })

    # Distractor paragraphs, resolved only from distractor metadata.
    distractor_segments: list[dict[str, Any]] = []
    diagnostic_limitations: list[dict[str, Any]] = []
    for answer_index, _ans in enumerate(qa_row.get("answers", [])):
        rec = None
        for r in dataset.distractor_by_qa_id.get(qa_row["qa_id"], []):
            if r.get("answer_index") == answer_index:
                rec = r
                break
        if rec is None:
            continue
        for slot_idx, slot in enumerate(rec.get("distractors", [])):
            from_pid = slot.get("from_paragraph_id")
            if not from_pid:
                # Perturbed distractor: synthetic text with no real
                # document location. Injected anyway, explicitly labeled
                # as fabricated (never claims a real paragraph_id) — per
                # operator decision 2026-08-20, oracle_local tests
                # in-context susceptibility to both distractor types, not
                # mined only.
                synthetic_text = slot.get("text", "")
                distractor_segments.append({
                    "role": "synthetic_distractor", "doc_id": None, "paragraph_id": None,
                    # No real document position exists for fabricated text;
                    # 0/len(text) are self-relative placeholders (consumers
                    # only use these for the informational best_span report,
                    # never to slice a larger document — see
                    # embedding_ranker.predict_span_over_candidates).
                    "char_start": 0, "char_end": len(synthetic_text),
                    "text": synthetic_text,
                })
                continue
            para = dataset.paragraphs.get(from_pid)
            if para is None:
                diagnostic_limitations.append({
                    "answer_index": answer_index, "slot_index": slot_idx,
                    "reason": f"from_paragraph_id={from_pid} not found in core paragraph metadata",
                })
                continue
            if from_pid in seen_pids:
                continue
            seen_pids.add(from_pid)
            distractor_segments.append({
                "role": "distractor", "doc_id": slot.get("from_doc_id", para["doc_id"]),
                "paragraph_id": from_pid, "char_start": para["char_start"], "char_end": para["char_end"],
                "text": para["text"],
            })

    segments = gold_segments + distractor_segments
    parts = []
    for seg in segments:
        if seg["role"] == "gold_context":
            label = f"[GOLD-ADJACENT: {seg['doc_id']} / {seg['paragraph_id']}]"
        elif seg["role"] == "synthetic_distractor":
            label = "[SYNTHETIC-DISTRACTOR]"
        else:
            label = f"[DISTRACTOR-SOURCE: {seg['doc_id']} / {seg['paragraph_id']}]"
        parts.append(f"{label}\n{seg['text']}")
    text = "\n\n".join(parts)

    return SerializedDocument(
        kind="oracle_local", doc_id=doc_id, text=text, sha256=sha256_text(text),
        serializer_version=version, char_count=len(text),
        segments=segments, diagnostic_limitations=diagnostic_limitations,
    )
