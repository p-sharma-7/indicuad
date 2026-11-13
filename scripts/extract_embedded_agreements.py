"""Split agreements that are buried inside excluded documents into their own PDFs.

Three documents in the corpus are excluded from annotation because they are
compound: a covering letter or a set of scheme guidelines with a real,
fully-drafted agreement attached behind it. The agreement is annotatable;
the wrapper is not. Excluding the whole file throws away the contract.

    DOC-001  MNRE PM Surya Ghar guidelines -> Annexure 4, Model Net Metering Agreement
    DOC-027  NSE Reg-30 letter             -> attached GeM contract (85 x "shall")
    DOC-031  NSE Reg-30 letter             -> MoRTH Letter of Acceptance

This writes the relevant page range of each source PDF out as a new PDF in
pdfs/, so the new documents enter the pipeline exactly like any other
source file — same extraction, same normalization, same segmentation. It
does not copy text: the page images and text layer come straight from the
original, so provenance is unbroken and the licence is inherited.

Page ranges are found by operative-language density rather than hardcoded,
so the split is reproducible and visible in the report.

Output: new PDFs in pdfs/, plus rows appended to the source manifest.
"""

from __future__ import annotations

import csv
import re
import sys

import fitz  # pymupdf

from indicuad_common import (
    CORPUS_DIR,
    MANIFEST_CSV,
    NORMALIZED_DIR,
    PDF_DIR,
    Stage,
    emit_report,
    read_jsonl,
)

SHALL = re.compile(r"\bshall\b", re.I)

# Which source document, what the extracted instrument is, and the filename
# the new document takes. Numbering continues the existing manifest.
PLAN = [
    {
        "source": "DOC-001",
        "filename": "44_MNRE_ModelNetMeteringAgreement.pdf",
        "title": "Model Net Metering Agreement (Annexure 4)",
        "anchor": "Model Net Metering Agreement",
    },
    {
        "source": "DOC-027",
        "filename": "45_GeM_WorkOrderContract_AbhishekIntegrations.pdf",
        "title": "GeM contract attached to a Reg-30 disclosure",
        "anchor": None,
    },
]

# DOC-031's MoRTH Letter of Acceptance was extracted and then dropped. Its
# single page is a scan with no text layer, and decisions.json requires
# scanned pages to be manually OCR-corrected before entering the core split
# — one page carrying 2 occurrences of "shall" does not justify that risk.
# Left here as a record so it is not silently re-added.

MIN_SHALL_PER_PAGE = 2


def page_text(document: dict, text: str, page: int) -> str:
    for span in document["page_spans"]:
        if span["page"] == page:
            return text[span["char_start"]:span["char_end"]]
    return ""


def find_range(document: dict, text: str, anchor: str | None) -> tuple[int, int, list[int]]:
    """Pages carrying the agreement, by operative-language density."""
    scores = {
        span["page"]: len(SHALL.findall(page_text(document, text, span["page"])))
        for span in document["page_spans"]
    }
    hot = sorted(p for p, n in scores.items() if n >= MIN_SHALL_PER_PAGE)

    anchor_page = None
    if anchor:
        position = text.find(anchor)
        if position != -1:
            anchor_page = next(
                (s["page"] for s in document["page_spans"]
                 if s["char_start"] <= position < s["char_end"]),
                None,
            )
            if anchor_page is not None:
                hot = [p for p in hot if p >= anchor_page] or hot

    if not hot:
        raise ValueError("no page carries operative language")
    # Start at the title page when we have one. An agreement's first page
    # carries the title, the parties and the dating clause and is usually
    # thin on "shall", so density alone starts one page too late and loses
    # exactly the material CAT-001 to CAT-003 need.
    first = anchor_page if anchor_page is not None else hot[0]
    last = first
    for page in sorted(scores):
        if page <= last:
            continue
        if page - last <= 1 and scores[page] >= 1:
            last = page
    return first, last, [scores[p] for p in range(first, last + 1)]


def main() -> int:
    stage = Stage("1b", "Split embedded agreements into their own PDFs")

    documents = {d["doc_id"]: d for d in read_jsonl(CORPUS_DIR / "documents.jsonl")}
    manifest_rows = []
    report = []

    for entry in PLAN:
        source = documents.get(entry["source"])
        if source is None:
            stage.fail(f"{entry['source']} not in the corpus")
            return 1

        text = (NORMALIZED_DIR / f"{entry['source']}.txt").read_text(encoding="utf-8")
        try:
            first, last, scores = find_range(source, text, entry["anchor"])
        except ValueError as error:
            stage.fail(f"{entry['source']}: {error}")
            return 1

        target = PDF_DIR / entry["filename"]
        if target.exists():
            stage.info(f"{entry['filename']} already exists, skipping")
            continue

        with fitz.open(PDF_DIR / source["filename"]) as src:
            out = fitz.open()
            out.insert_pdf(src, from_page=first - 1, to_page=last - 1)
            out.save(target)
            out.close()

        stage.info(
            f"{entry['source']} pages {first}-{last} -> {entry['filename']} "
            f"({last - first + 1} pages, shall/page {scores})"
        )
        manifest_rows.append(
            {
                "filename": entry["filename"],
                "source_url": source["source_url"],
                "source_entity": source["source_entity"],
                "license_notes": (
                    f"{source['license_notes']} | DERIVED: pages {first}-{last} of "
                    f"{source['filename']} extracted as a standalone instrument "
                    f"({entry['title']}). Same source and licence as the parent."
                ),
            }
        )
        report.append(
            {
                "new_file": entry["filename"],
                "parent": entry["source"],
                "parent_file": source["filename"],
                "pages": [first, last],
                "page_count": last - first + 1,
                "shall_per_page": scores,
                "instrument": entry["title"],
            }
        )

    if manifest_rows:
        with MANIFEST_CSV.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["filename", "source_url", "source_entity", "license_notes"],
            )
            for row in manifest_rows:
                writer.writerow(row)
        stage.info(f"appended {len(manifest_rows)} row(s) to {MANIFEST_CSV.name}")

    emit_report(
        "embedded_agreements_report.json",
        {
            "status": "PASS",
            "extracted": report,
            "note": (
                "Derived documents: page ranges of existing source PDFs, not new "
                "acquisitions. Text layer and licence are inherited from the parent. "
                "The parent documents remain in the corpus as excluded/retrieval-only."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
