"""Stage 3 — document normalization.

Produces one normalized text file per source PDF. This text is the frozen
reference that every gold span offset indexes into, so the rules here are
deliberately conservative.

Applied (objectively mechanical, counted and logged per document):
    - line endings CRLF/CR -> LF
    - soft hyphens and zero-width characters removed
    - non-breaking space variants -> ordinary space
    - trailing whitespace stripped per line
    - runs of 3+ blank lines collapsed to one blank line

Detected but never repaired (substantive text; flagged for humans):
    - Cyrillic/Greek homoglyphs inside Latin words (bad source text layer)
    - U+FFFD replacement characters
    - unexpected control characters

Outputs:
    work/normalized/<doc_id>.txt      the frozen document text
    corpus/documents.jsonl
    validation/normalization_report.json
"""

from __future__ import annotations

import re
import sys
import unicodedata

from indicuad_common import (
    CORPUS_DIR,
    EXTRACTED_DIR,
    NORMALIZED_DIR,
    Stage,
    doc_id_for_filename,
    emit_report,
    ensure_dirs,
    read_json,
    read_manifest,
    sha256_text,
    write_jsonl,
)

ZERO_WIDTH = dict.fromkeys(map(ord, "­​‌‍﻿"), None)
NBSP_VARIANTS = {ord(c): " " for c in "      "}

CYRILLIC_OR_GREEK = re.compile(r"[Ѐ-ӿͰ-Ͽ]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_homoglyphs(text: str) -> list[str]:
    """Find words mixing Latin with Cyrillic/Greek — a corrupt text layer.

    'SEВІ' (Cyrillic В, І) reads as 'SEBI' but will never match a search
    for 'SEBI'. We surface these rather than rewriting legal text.
    """
    hits = []
    for match in WORD.finditer(text):
        word = match.group()
        if CYRILLIC_OR_GREEK.search(word) and LATIN.search(word):
            hits.append(word)
    # Preserve first-seen order, deduplicated.
    return list(dict.fromkeys(hits))


def detect_control_chars(text: str) -> list[str]:
    bad = {
        ch
        for ch in set(text)
        if unicodedata.category(ch) == "Cc" and ch not in "\n\t"
    }
    return sorted(f"U+{ord(c):04X}" for c in bad)


def normalize_page(text: str) -> tuple[str, dict[str, int]]:
    ops = {}

    before = len(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    ops["line_endings_normalized"] = int(len(text) != before)

    stripped = text.translate(ZERO_WIDTH)
    ops["zero_width_removed"] = len(text) - len(stripped)
    text = stripped

    nbsp_count = sum(text.count(chr(c)) for c in NBSP_VARIANTS)
    text = text.translate(NBSP_VARIANTS)
    ops["nbsp_replaced"] = nbsp_count

    lines = text.split("\n")
    trimmed = [line.rstrip() for line in lines]
    ops["lines_right_trimmed"] = sum(1 for a, b in zip(lines, trimmed) if a != b)
    text = "\n".join(trimmed)

    collapsed = re.sub(r"\n{3,}", "\n\n", text)
    ops["blank_runs_collapsed"] = int(collapsed != text)
    text = collapsed

    return text.strip("\n"), ops


def main() -> int:
    stage = Stage("3", "Document normalization")

    ocr_report = read_json(
        NORMALIZED_DIR.parent.parent / "validation" / "ocr_report.json"
    )
    if not ocr_report:
        stage.fail("validation/ocr_report.json missing — run extract_pdf_text.py first.")
        return 1

    ensure_dirs(NORMALIZED_DIR, CORPUS_DIR)
    rows = read_manifest()

    documents = []
    report_entries = []
    flagged = []

    for row in rows:
        filename = row["filename"]
        doc_id = doc_id_for_filename(filename)
        extracted = read_json(EXTRACTED_DIR / f"{doc_id}.json")
        if extracted is None:
            stage.fail(f"{doc_id}: no extraction found; rerun stage 2.")
            return 1

        parts = []
        page_spans = []
        totals: dict[str, int] = {}
        cursor = 0

        for page in extracted["pages"]:
            norm, ops = normalize_page(page["text"])
            for key, value in ops.items():
                totals[key] = totals.get(key, 0) + value

            start = cursor
            parts.append(norm)
            cursor += len(norm)
            page_spans.append(
                {
                    "page": page["page"],
                    "char_start": start,
                    "char_end": cursor,
                    "needs_ocr": page["needs_ocr"],
                }
            )
            # Pages are joined by a blank line; the separator belongs to
            # no page, which keeps page attribution unambiguous.
            if page["page"] != extracted["pages"][-1]["page"]:
                parts.append("\n\n")
                cursor += 2

        text = "".join(parts)

        homoglyphs = detect_homoglyphs(text)
        controls = detect_control_chars(text)
        replacement_chars = text.count("�")

        out_path = NORMALIZED_DIR / f"{doc_id}.txt"
        out_path.write_text(text, encoding="utf-8", newline="\n")

        record = {
            "doc_id": doc_id,
            "filename": filename,
            "source_url": row["source_url"],
            "source_entity": row["source_entity"],
            "license_notes": row["license_notes"],
            "doc_origin": "collected_real",
            "tier": None,
            "split": None,
            "text_path": f"work/normalized/{doc_id}.txt",
            "text_sha256": sha256_text(text),
            "char_count": len(text),
            "page_count": extracted["page_count"],
            "page_spans": page_spans,
            "text_extraction": extracted["text_extraction"],
            "ocr_pages": extracted["ocr_pages"],
            "ocr_required": extracted["ocr_required"],
            "ocr_performed": extracted["ocr_performed"],
            "manual_review_required": extracted["manual_review_required"],
        }
        documents.append(record)

        entry = {
            "doc_id": doc_id,
            "char_count": len(text),
            "normalization_ops": totals,
            "homoglyph_words": homoglyphs[:20],
            "homoglyph_word_count": len(homoglyphs),
            "control_chars": controls,
            "replacement_chars": replacement_chars,
        }
        report_entries.append(entry)

        notes = []
        if homoglyphs:
            notes.append(f"homoglyphs={len(homoglyphs)}")
        if replacement_chars:
            notes.append(f"U+FFFD={replacement_chars}")
        if controls:
            notes.append(f"control={controls}")
        if notes:
            flagged.append(entry)
            stage.warn(f"{doc_id}: " + ", ".join(notes))

    write_jsonl(CORPUS_DIR / "documents.jsonl", documents)
    stage.info(f"wrote corpus/documents.jsonl ({len(documents)} documents)")
    stage.info(f"total normalized characters: {sum(d['char_count'] for d in documents):,}")

    emit_report(
        "normalization_report.json",
        {
            "status": "PASS" if not flagged else "PASS_WITH_FLAGS",
            "documents_total": len(documents),
            "documents_flagged": len(flagged),
            "note": (
                "Flagged text-layer defects are reported, never rewritten. "
                "Homoglyph words will not match plain-Latin keyword sweeps."
            ),
            "documents": report_entries,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
