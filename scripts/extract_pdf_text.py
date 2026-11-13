"""Stage 2 — PDF text extraction with scanned-page detection.

Extracts native text page by page, preserving page boundaries, and flags
pages whose extraction looks suspiciously empty relative to their visual
content (the signature of a scanned image page).

This script never invents text. If a page needs OCR it is recorded as
needing OCR and the document is marked manual_review_required; it is not
silently emitted as an empty page.

Outputs:
    work/extracted/<doc_id>.json      per-page text + provenance
    validation/ocr_report.json
"""

from __future__ import annotations

import sys

import pymupdf

from extract_word_text import extract as extract_word

from indicuad_common import (
    EXTRACTED_DIR,
    PDF_DIR,
    Stage,
    doc_id_for_filename,
    emit_report,
    ensure_dirs,
    read_json,
    read_manifest,
    sha256_file,
    write_json,
)

# A page with fewer than this many extractable characters is a candidate
# for being a scan rather than a text page.
LOW_TEXT_CHAR_THRESHOLD = 100

# ...but only if it actually has visual content covering the page. A
# genuinely near-blank page (a separator or signature page) is not a scan.
IMAGE_COVERAGE_THRESHOLD = 0.40


def page_image_coverage(page) -> float:
    """Fraction of the page area covered by raster images."""
    page_area = abs(page.rect.get_area())
    if page_area <= 0:
        return 0.0
    covered = 0.0
    try:
        for info in page.get_image_info():
            bbox = pymupdf.Rect(info["bbox"])
            covered += abs(bbox.get_area())
    except Exception:  # noqa: BLE001 - malformed image dict shouldn't kill the run
        return 0.0
    return min(covered / page_area, 1.0)


def extract_word_document(filename: str, path) -> dict:
    """A Word source has no pages to extract, so it is recorded as one page.

    Several Startup India model contracts are published only as .doc/.docx.
    Their text layer is native and complete, so there is nothing to OCR and
    nothing to detect as scanned — the page-level machinery below simply
    does not apply. Page numbers in the output are therefore synthetic.
    """
    text, fmt = extract_word(path)
    return {
        "doc_id": doc_id_for_filename(filename),
        "filename": filename,
        "pdf_sha256": sha256_file(path),
        "page_count": 1,
        "char_count": len(text.strip()),
        "text_extraction": f"native ({fmt})",
        "ocr_pages": [],
        "ocr_required": False,
        "ocr_performed": False,
        "ocr_engine": None,
        "manual_review_required": False,
        "pages": [
            {
                "page": 1,
                "char_count": len(text.strip()),
                "image_coverage": 0.0,
                "needs_ocr": False,
                "text": text,
            }
        ],
    }


def extract_document(filename: str) -> dict:
    doc_id = doc_id_for_filename(filename)
    path = PDF_DIR / filename

    if path.suffix.lower() in {".doc", ".docx"}:
        return extract_word_document(filename, path)

    pages = []
    ocr_pages = []
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            # "text" mode preserves reading order and paragraph breaks
            # better than raw extraction for these layouts.
            text = page.get_text("text")
            stripped = text.strip()
            coverage = page_image_coverage(page)
            is_scanned = (
                len(stripped) < LOW_TEXT_CHAR_THRESHOLD
                and coverage >= IMAGE_COVERAGE_THRESHOLD
            )
            if is_scanned:
                ocr_pages.append(page_number)
            pages.append(
                {
                    "page": page_number,
                    "char_count": len(stripped),
                    "image_coverage": round(coverage, 3),
                    "needs_ocr": is_scanned,
                    "text": text,
                }
            )

    native_chars = sum(p["char_count"] for p in pages)
    if not ocr_pages:
        extraction_mode = "native"
    elif len(ocr_pages) == len(pages):
        extraction_mode = "scanned_no_native_text"
    else:
        extraction_mode = "mixed_native_and_scanned"

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pdf_sha256": sha256_file(path),
        "page_count": len(pages),
        "char_count": native_chars,
        "text_extraction": extraction_mode,
        "ocr_pages": ocr_pages,
        "ocr_required": bool(ocr_pages),
        "ocr_performed": False,
        "ocr_engine": None,
        # Any document with an un-OCR'd scanned page cannot be annotated
        # until a human resolves it: gold spans would point into a gap.
        "manual_review_required": bool(ocr_pages),
        "pages": pages,
    }


def main() -> int:
    stage = Stage("2", "PDF text extraction and OCR triage")

    inventory = read_json(
        (EXTRACTED_DIR.parent.parent / "validation" / "source_inventory_report.json")
    )
    if not inventory or inventory.get("status") != "PASS":
        stage.fail("stage 0/1 has not passed — run inspect_sources.py first.")
        return 1

    ensure_dirs(EXTRACTED_DIR)
    rows = read_manifest()

    summaries = []
    for row in rows:
        filename = row["filename"]
        out_path = EXTRACTED_DIR / f"{doc_id_for_filename(filename)}.json"

        # Restartable: skip work already done for an unchanged source PDF.
        existing = read_json(out_path)
        if existing and existing.get("pdf_sha256") == sha256_file(PDF_DIR / filename):
            record = existing
        else:
            record = extract_document(filename)
            write_json(out_path, record)

        summary = {k: v for k, v in record.items() if k != "pages"}
        summary["empty_pages"] = [
            p["page"] for p in record["pages"] if p["char_count"] == 0
        ]
        # Derive outstanding OCR from actual page state rather than the
        # original ocr_required flag: ocr_pages.py may have since resolved
        # these pages, and ocr_required records history, not status.
        summary["ocr_outstanding_pages"] = [
            p["page"] for p in record["pages"] if p.get("needs_ocr")
        ]
        summary["ocr_pages_recovered"] = [
            p["page"] for p in record["pages"] if p.get("source") == "ocr"
        ]
        summaries.append(summary)

        flag = ""
        if summary["ocr_outstanding_pages"]:
            flag = f"  <-- OCR NEEDED on pages {summary['ocr_outstanding_pages']}"
        elif summary["ocr_pages_recovered"]:
            flag = f"  <-- OCR applied to pages {summary['ocr_pages_recovered']}"
        stage.info(
            f"{record['doc_id']}  pages={record['page_count']:>3}  "
            f"chars={record['char_count']:>7}  {record['text_extraction']}{flag}"
        )

    needing_ocr = [s for s in summaries if s["ocr_outstanding_pages"]]
    recovered = [s for s in summaries if s["ocr_pages_recovered"]]
    total_ocr_pages = sum(len(s["ocr_outstanding_pages"]) for s in summaries)

    status = "PASS" if not needing_ocr else "NEEDS_HUMAN_REVIEW"
    emit_report(
        "ocr_report.json",
        {
            "status": status,
            "documents_total": len(summaries),
            "documents_native_only": len(summaries) - len(needing_ocr) - len(recovered),
            "documents_needing_ocr": len(needing_ocr),
            "documents_ocr_recovered": len(recovered),
            "total_pages_needing_ocr": total_ocr_pages,
            "total_pages_ocr_recovered": sum(len(s["ocr_pages_recovered"]) for s in summaries),
            "ocr_engine_available": True,
            "low_text_char_threshold": LOW_TEXT_CHAR_THRESHOLD,
            "image_coverage_threshold": IMAGE_COVERAGE_THRESHOLD,
            "documents": summaries,
        },
    )

    if needing_ocr:
        stage.warn(
            f"{len(needing_ocr)} document(s), {total_ocr_pages} page(s) need OCR; "
            "no OCR engine installed. These are flagged, not guessed."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
