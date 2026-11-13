"""Stage 0/1 — repository inspection and manifest/PDF consistency.

Verifies that every manifest row has a matching PDF in pdfs/ by exact
filename, that no PDF is unaccounted for, and that each PDF opens. Writes
validation/source_inventory_report.json.

A FAIL here is FATAL per the preparation spec: nothing downstream may run,
and no substitute PDF is ever chosen for a missing one.
"""

from __future__ import annotations

import sys
from collections import Counter

import pymupdf

from extract_word_text import extract as extract_word

from indicuad_common import (
    MANIFEST_CSV,
    PDF_DIR,
    PROJECT_ROOT,
    Stage,
    doc_id_for_filename,
    emit_report,
    find_protocol,
    sha256_file,
)

REQUIRED_COLUMNS = ["filename", "source_url", "source_entity", "license_notes"]


def probe_pdf(path):
    """Open a source and report page count / encryption without extracting text.

    Word sources have no page structure to probe, so readability is decided
    by whether the text extractor can read them at all — the same check that
    stage 2 will apply.
    """
    if path.suffix.lower() in {".doc", ".docx"}:
        try:
            text, _ = extract_word(path)
        except Exception as exc:  # noqa: BLE001
            return {"readable": False, "pages": None, "error": f"{type(exc).__name__}: {exc}"}
        return {"readable": bool(text.strip()), "pages": 1, "error": None}
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                return {"readable": False, "pages": None, "error": "password protected"}
            return {"readable": True, "pages": doc.page_count, "error": None}
    except Exception as exc:  # noqa: BLE001 - any failure means unreadable source
        return {"readable": False, "pages": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    stage = Stage("0/1", "Repository inspection and manifest consistency")

    protocol = find_protocol()
    if protocol is None:
        stage.fail("ANNOTATION_PROTOCOL.md not found — FATAL, cannot proceed.")
        emit_report(
            "source_inventory_report.json",
            {"status": "FATAL", "reason": "annotation protocol missing"},
        )
        return 1
    stage.info(f"protocol found: {protocol.name}")

    if not MANIFEST_CSV.is_file():
        stage.fail("document_source_metadata.csv not found — FATAL.")
        emit_report(
            "source_inventory_report.json",
            {"status": "FATAL", "reason": "source manifest missing"},
        )
        return 1

    from indicuad_common import read_manifest

    rows = read_manifest()
    stage.info(f"manifest rows: {len(rows)}")

    missing_columns = [c for c in REQUIRED_COLUMNS if rows and c not in rows[0]]
    if missing_columns:
        stage.fail(f"manifest missing required columns: {missing_columns}")
        emit_report(
            "source_inventory_report.json",
            {"status": "FATAL", "missing_columns": missing_columns},
        )
        return 1

    manifest_names = [r["filename"] for r in rows]
    duplicate_names = sorted(n for n, c in Counter(manifest_names).items() if c > 1)
    duplicate_urls = sorted(
        u for u, c in Counter(r["source_url"] for r in rows).items() if c > 1 and u
    )

    # Sources arrive as PDFs and as Word files; several Startup India model
    # contracts are published only as .doc/.docx.
    disk_pdfs = sorted(
        p.name for p in PDF_DIR.iterdir()
        if p.suffix.lower() in {".pdf", ".doc", ".docx"}
    ) if PDF_DIR.is_dir() else []
    missing_files = [n for n in manifest_names if n not in set(disk_pdfs)]
    extra_files = [n for n in disk_pdfs if n not in set(manifest_names)]

    # doc_id derivation must be total and injective over the manifest.
    doc_ids: dict[str, str] = {}
    id_errors: list[str] = []
    for name in manifest_names:
        try:
            did = doc_id_for_filename(name)
        except ValueError as exc:
            id_errors.append(str(exc))
            continue
        if did in doc_ids:
            id_errors.append(f"doc_id collision {did}: {doc_ids[did]!r} vs {name!r}")
        doc_ids[did] = name

    documents = []
    unreadable = []
    for name in manifest_names:
        path = PDF_DIR / name
        entry = {"filename": name, "doc_id": doc_ids.get(name.split("_")[0], None)}
        try:
            entry["doc_id"] = doc_id_for_filename(name)
        except ValueError:
            entry["doc_id"] = None
        if not path.is_file():
            entry.update({"present": False, "readable": False, "pages": None})
            documents.append(entry)
            continue
        probe = probe_pdf(path)
        entry.update(
            {
                "present": True,
                "readable": probe["readable"],
                "pages": probe["pages"],
                "bytes": path.stat().st_size,
                "pdf_sha256": sha256_file(path),
            }
        )
        if not probe["readable"]:
            entry["error"] = probe["error"]
            unreadable.append(name)
        documents.append(entry)

    problems = {
        "missing_files": missing_files,
        "extra_files": extra_files,
        "duplicate_manifest_filenames": duplicate_names,
        "duplicate_source_urls": duplicate_urls,
        "unreadable_pdfs": unreadable,
        "doc_id_errors": id_errors,
    }

    # Duplicate source URLs are suspicious but not fatal: two rows may
    # legitimately cite one archive page. Flag, do not block.
    blocking = bool(missing_files or duplicate_names or unreadable or id_errors)
    status = "FAIL" if blocking else "PASS"

    for name in missing_files:
        stage.fail(f"manifest row has no PDF on disk: {name}")
    for name in extra_files:
        stage.warn(f"PDF on disk not in manifest: {name}")
    for url in duplicate_urls:
        stage.warn(f"duplicate source_url in manifest: {url}")
    for name in unreadable:
        stage.fail(f"unreadable PDF: {name}")

    total_pages = sum(d["pages"] or 0 for d in documents)
    stage.info(f"pdf files found: {len(disk_pdfs)}   total pages: {total_pages}")

    emit_report(
        "source_inventory_report.json",
        {
            "status": status,
            "protocol_file": protocol.name,
            "project_root": str(PROJECT_ROOT),
            "manifest_rows": len(rows),
            "pdf_files_found": len(disk_pdfs),
            "total_pages": total_pages,
            **problems,
            "documents": documents,
        },
    )

    if status != "PASS":
        stage.fail("source inventory did not pass — stopping before corpus preparation.")
        return 1
    stage.info("source inventory PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
