"""Deterministic, auditable normalization of raw model output (spec §10).

Raw output is always preserved separately by the caller (the JSONL record
keeps both `raw_output` and this module's `normalized` result). Nothing here
repairs spelling, OCR defects, punctuation, numbers, party names, or legal
terms — only known formatting wrappers and comparison-only normalization.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Prompt markers a completion may re-emit; a free-text completion is cut at the first one.
PROMPT_MARKERS = (
    "\nQuestion:", "\nContract:", "\nContract excerpt:", "\nJSON:",
    "\n[INST]", "[INST]", "\nHuman:", "\nAssistant:", "<s>",
)

ALIGNMENT_EXACT = "exact"
ALIGNMENT_NORMALIZED_EXACT = "normalized_exact"
ALIGNMENT_NOT_FOUND = "not_found"
ALIGNMENT_MULTIPLE_MATCHES = "multiple_matches"
ALIGNMENT_INVALID_JSON = "invalid_json"
ALIGNMENT_EMPTY = "empty_output"


@dataclass
class NormalizedOutput:
    raw_output: str
    setting: str                       # direct_extraction | structured_extraction
    unwrapped_spans: list[str] = field(default_factory=list)   # after wrapper stripping, before doc alignment
    alignment_status: str = ALIGNMENT_EMPTY
    aligned_spans: list[dict[str, Any]] = field(default_factory=list)  # [{text, char_start, char_end}]
    parse_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "unwrapped_spans": self.unwrapped_spans,
            "alignment_status": self.alignment_status,
            "aligned_spans": self.aligned_spans,
            "parse_notes": self.parse_notes,
        }


def cut_at_prompt_marker(text: str) -> str:
    """Truncate a completion at the first re-emitted prompt marker."""
    cut = len(text)
    for marker in PROMPT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def first_balanced_object(text: str) -> str | None:
    """The first brace-balanced `{...}` substring (braces inside JSON strings
    ignored), or None if there is none."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def unwrap_structured(text: str) -> tuple[list[str], list[str]]:
    """structured_extraction: parse the first balanced JSON object and take its
    `answer` field. A parse failure, missing key or non-string value yields no
    span (scored as an empty prediction). Returns (spans, notes)."""
    candidate = first_balanced_object(text)
    if candidate is None:
        return [], ["invalid_json"]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return [], ["invalid_json"]
    if not isinstance(obj, dict) or "answer" not in obj:
        return [], ["json_missing_answer_field"]
    if not isinstance(obj["answer"], str):
        return [], ["answer_field_not_string"]
    return [obj["answer"]], []


def unwrap_direct(text: str) -> tuple[list[str], list[str]]:
    """direct_extraction: cut at the first re-emitted prompt marker and score
    the remainder as-is (one span)."""
    candidate = cut_at_prompt_marker(text).strip()
    if not candidate:
        return [], ["empty_after_unwrap"]
    return [candidate], []


def nfkc_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def whitespace_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def comparison_form(text: str) -> str:
    """NFKC + whitespace-collapse + lowercase — comparison-only, never used
    for anything that gets written back as a 'cleaned' quotation."""
    return whitespace_normalize(nfkc_normalize(text)).lower()


def align_span_to_document(span_text: str, document_text: str) -> list[dict[str, Any]]:
    """Attempt exact substring alignment of a cleaned span back to the
    document. Returns all exact match locations (empty if none). Tries raw
    exact match first, then NFKC+whitespace-normalized match on a
    whitespace-normalized view of the document (offsets rebased back to the
    raw document via a monotonic index map) — never a fuzzy/edit-distance
    match, and never repairs the span text itself."""
    matches = []
    start = 0
    while True:
        idx = document_text.find(span_text, start)
        if idx == -1:
            break
        matches.append({"text": span_text, "char_start": idx, "char_end": idx + len(span_text), "match_kind": ALIGNMENT_EXACT})
        start = idx + 1
    return matches


def _build_normalized_index_map(document_text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed, NFKC'd view of document_text plus a map from
    each character in that view back to its offset in document_text, so a
    normalized-exact match can be rebased to real document coordinates."""
    norm_chars: list[str] = []
    index_map: list[int] = []
    prev_was_space = True  # treat leading whitespace as already-collapsed
    for i, ch in enumerate(document_text):
        nfkc_ch = unicodedata.normalize("NFKC", ch)
        for c in nfkc_ch:
            if c.isspace():
                if not prev_was_space:
                    norm_chars.append(" ")
                    index_map.append(i)
                prev_was_space = True
            else:
                norm_chars.append(c.lower())
                index_map.append(i)
                prev_was_space = False
    return "".join(norm_chars), index_map


def align_span_normalized(span_text: str, document_text: str, doc_norm_cache: dict[str, Any]) -> list[dict[str, Any]]:
    key = id(document_text)
    if key not in doc_norm_cache:
        doc_norm_cache[key] = _build_normalized_index_map(document_text)
    norm_doc, index_map = doc_norm_cache[key]
    norm_span = comparison_form(span_text)
    if not norm_span:
        return []
    matches = []
    start = 0
    while True:
        idx = norm_doc.find(norm_span, start)
        if idx == -1:
            break
        raw_start = index_map[idx]
        raw_end = index_map[idx + len(norm_span) - 1] + 1
        matches.append({
            "text": document_text[raw_start:raw_end],
            "char_start": raw_start, "char_end": raw_end,
            "match_kind": ALIGNMENT_NORMALIZED_EXACT,
        })
        start = idx + 1
    return matches


def normalize_output(raw_output: str, setting: str, document_text: str | None,
                      doc_norm_cache: dict[str, Any] | None = None) -> NormalizedOutput:
    result = NormalizedOutput(raw_output=raw_output, setting=setting)
    if raw_output is None or not raw_output.strip():
        result.alignment_status = ALIGNMENT_EMPTY
        return result

    if setting == "structured_extraction":
        spans, notes = unwrap_structured(raw_output)
        if not spans and "invalid_json" in notes:
            result.alignment_status = ALIGNMENT_INVALID_JSON
            result.parse_notes = notes
            return result
    else:
        spans, notes = unwrap_direct(raw_output)

    result.unwrapped_spans = spans
    result.parse_notes = notes

    if not spans:
        result.alignment_status = ALIGNMENT_EMPTY
        return result

    if document_text is None:
        # No document available to align against (e.g. offline unit test of
        # unwrapping logic only).
        result.alignment_status = ALIGNMENT_NOT_FOUND
        return result

    doc_norm_cache = doc_norm_cache if doc_norm_cache is not None else {}
    all_aligned: list[dict[str, Any]] = []
    statuses = []
    for span in spans:
        exact = align_span_to_document(span, document_text)
        if exact:
            all_aligned.extend(exact)
            statuses.append(ALIGNMENT_MULTIPLE_MATCHES if len(exact) > 1 else ALIGNMENT_EXACT)
            continue
        norm_matches = align_span_normalized(span, document_text, doc_norm_cache)
        if norm_matches:
            all_aligned.extend(norm_matches)
            statuses.append(ALIGNMENT_MULTIPLE_MATCHES if len(norm_matches) > 1 else ALIGNMENT_NORMALIZED_EXACT)
        else:
            statuses.append(ALIGNMENT_NOT_FOUND)

    result.aligned_spans = all_aligned
    if all(s == ALIGNMENT_NOT_FOUND for s in statuses):
        result.alignment_status = ALIGNMENT_NOT_FOUND
    elif any(s == ALIGNMENT_MULTIPLE_MATCHES for s in statuses):
        result.alignment_status = ALIGNMENT_MULTIPLE_MATCHES
    elif all(s == ALIGNMENT_EXACT for s in statuses):
        result.alignment_status = ALIGNMENT_EXACT
    else:
        result.alignment_status = ALIGNMENT_NORMALIZED_EXACT
    return result
