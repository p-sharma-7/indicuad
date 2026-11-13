"""Stage 2b — OCR the pages that native extraction could not read.

Runs only on pages stage 2 flagged as scanned, and writes the recovered
text back into work/extracted/<doc_id>.json with provenance intact:
ocr_performed, the engine used, and a per-page mean confidence.

Two rules from the spec are load-bearing here:
  - OCR text is never silently mixed with native text. Every OCR'd page
    keeps `source: "ocr"` so downstream consumers and the datasheet can
    tell them apart.
  - Low-confidence output is flagged for human review rather than
    accepted. OCR uncertainty is a NEEDS_HUMAN_REVIEW class, not a
    deterministic repair.

Re-run stage 2 afterwards and it will keep these results: extraction
skips any document whose source PDF hash is unchanged.

    python scripts/ocr_pages.py
"""

from __future__ import annotations

import re
import sys

import pymupdf

from indicuad_common import (
    EXTRACTED_DIR,
    PDF_DIR,
    Stage,
    emit_report,
    read_json,
    write_json,
)

RENDER_DPI = 300

# Mean per-line confidence below which the page is not trusted.
CONFIDENCE_THRESHOLD = 0.80

# A page that OCRs to less than this is almost certainly not a text page
# (a photograph, a logo, a blank scan) and should not be reported as one.
MIN_CHARS_FOR_TEXT_PAGE = 40


# This engine loses inter-word spaces on justified text, producing
# "OfficeMemorandum" for "Office Memorandum". Measure it rather than
# describe it: the rate belongs in the datasheet, and a keyword sweep or
# retrieval query will not match a glued token.
GLUED_WORD = re.compile(r"[a-z][A-Z]")
ALPHA_TOKEN = re.compile(r"[A-Za-z]{2,}")


def glued_word_rate(text: str) -> tuple[float, int, int]:
    tokens = ALPHA_TOKEN.findall(text)
    if not tokens:
        return 0.0, 0, 0
    glued = sum(1 for t in tokens if GLUED_WORD.search(t))
    return round(glued / len(tokens), 4), glued, len(tokens)


def ocr_page(engine, pdf_path, page_number: int) -> dict:
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(dpi=RENDER_DPI)
        image_bytes = pixmap.tobytes("png")

    result, _ = engine(image_bytes)
    if not result:
        return {"text": "", "confidence": 0.0, "lines": 0}

    lines = [str(item[1]) for item in result]
    scores = [float(item[2]) for item in result]
    return {
        "text": "\n".join(lines),
        "confidence": round(sum(scores) / len(scores), 4),
        "lines": len(lines),
    }


def main() -> int:
    stage = Stage("2b", "OCR of scanned pages")

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        stage.fail(
            "rapidocr-onnxruntime is not installed. "
            "Install it, or resolve the scanned pages by hand."
        )
        return 1

    report = read_json(EXTRACTED_DIR.parent.parent / "validation" / "ocr_report.json")
    if not report:
        stage.fail("validation/ocr_report.json missing — run extract_pdf_text.py first.")
        return 1

    pending = [d for d in report["documents"] if d.get("ocr_required")]
    if not pending:
        stage.info("no pages require OCR")
        return 0

    engine = RapidOCR()
    results = []
    flagged = []

    for entry in pending:
        doc_id = entry["doc_id"]
        record = read_json(EXTRACTED_DIR / f"{doc_id}.json")
        if record is None:
            stage.fail(f"{doc_id}: extraction record missing")
            return 1

        pdf_path = PDF_DIR / record["filename"]
        pages_by_number = {p["page"]: p for p in record["pages"]}
        page_results = []

        for page_number in entry["ocr_pages"]:
            outcome = ocr_page(engine, pdf_path, page_number)
            page = pages_by_number[page_number]

            text = outcome["text"]
            confidence = outcome["confidence"]
            low_confidence = confidence < CONFIDENCE_THRESHOLD
            too_short = len(text.strip()) < MIN_CHARS_FOR_TEXT_PAGE
            glued_rate, glued, tokens = glued_word_rate(text)

            page.update(
                {
                    "text": text,
                    "char_count": len(text.strip()),
                    "source": "ocr",
                    "ocr_confidence": confidence,
                    "ocr_lines": outcome["lines"],
                    "needs_ocr": False,
                    "ocr_low_confidence": low_confidence,
                    "ocr_glued_word_rate": glued_rate,
                }
            )

            status = "ok"
            if too_short:
                status = "no_text_recovered"
            elif low_confidence:
                status = "low_confidence"

            if status != "ok":
                flagged.append(
                    {"doc_id": doc_id, "page": page_number, "status": status,
                     "confidence": confidence, "chars": len(text.strip())}
                )

            page_results.append(
                {
                    "page": page_number,
                    "chars": len(text.strip()),
                    "lines": outcome["lines"],
                    "confidence": confidence,
                    "glued_word_rate": glued_rate,
                    "glued_words": glued,
                    "tokens": tokens,
                    "status": status,
                }
            )
            stage.info(
                f"{doc_id} p{page_number}: {len(text.strip()):>5} chars, "
                f"{outcome['lines']:>3} lines, confidence {confidence:.3f}, "
                f"glued {glued}/{tokens} ({glued_rate:.1%})  [{status}]"
            )

        recovered = [p for p in page_results if p["status"] != "no_text_recovered"]
        record["ocr_performed"] = bool(recovered)
        record["ocr_engine"] = "rapidocr-onnxruntime" if recovered else None
        record["ocr_render_dpi"] = RENDER_DPI
        record["char_count"] = sum(p["char_count"] for p in record["pages"])
        # Pages that recovered no text stay flagged for a human.
        still_blank = [p["page"] for p in page_results if p["status"] == "no_text_recovered"]
        record["ocr_pages_unrecovered"] = still_blank
        record["manual_review_required"] = bool(
            still_blank or any(p["status"] == "low_confidence" for p in page_results)
        )
        if record["ocr_performed"]:
            record["text_extraction"] = "native_plus_ocr"

        write_json(EXTRACTED_DIR / f"{doc_id}.json", record)
        results.append({"doc_id": doc_id, "pages": page_results,
                        "manual_review_required": record["manual_review_required"]})

    emit_report(
        "ocr_applied_report.json",
        {
            "status": "PASS" if not flagged else "PASS_WITH_FLAGS",
            "engine": "rapidocr-onnxruntime",
            "render_dpi": RENDER_DPI,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "documents": results,
            "flagged_pages": flagged,
            "glued_word_rate_overall": round(
                sum(p["glued_words"] for d in results for p in d["pages"])
                / max(sum(p["tokens"] for d in results for p in d["pages"]), 1), 4
            ),
            "glued_word_note": (
                "Fraction of alphabetic tokens containing a lowercase-uppercase "
                "transition, i.e. two words run together by lost inter-word spacing "
                "('OfficeMemorandum'). These tokens will not match a plain keyword "
                "sweep or retrieval query. Report in the datasheet."
            ),
            "note": (
                "OCR'd pages carry source='ocr' in work/extracted/. Re-run "
                "normalize_corpus.py and segment_paragraphs.py to fold the "
                "recovered text into the corpus."
            ),
        },
    )

    stage.info("")
    stage.warn("re-run normalize_corpus.py and segment_paragraphs.py to fold OCR text in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
