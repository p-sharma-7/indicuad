"""Stage 18 / §26 — freeze readiness check and MANIFEST.txt.

Evaluates every condition in the spec's freeze checklist against actual
artifacts rather than against anybody's recollection, and writes
MANIFEST.txt only when all of them hold.

The default mode reports status and stops. Pass --write-manifest to emit
MANIFEST.txt, which is refused unless every condition passes.
"""

from __future__ import annotations

import sys

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    NORMALIZED_DIR,
    PROJECT_ROOT,
    QA_DIR,
    RETRIEVAL_DIR,
    Stage,
    VALIDATION_DIR,
    current_annotations_path,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
)


def condition(name: str, ok: bool, detail: str) -> dict:
    return {"condition": name, "passed": bool(ok), "detail": detail}


def evaluate() -> list[dict]:
    checks = []

    inventory = read_json(VALIDATION_DIR / "source_inventory_report.json") or {}
    checks.append(condition(
        "Source manifest validated",
        inventory.get("status") == "PASS",
        f"status={inventory.get('status')}, {inventory.get('manifest_rows')} rows",
    ))
    checks.append(condition(
        "All required PDFs accounted for",
        not inventory.get("missing_files") and not inventory.get("unreadable_pdfs"),
        f"{inventory.get('pdf_files_found')} PDFs, "
        f"{len(inventory.get('missing_files') or [])} missing",
    ))

    ocr = read_json(VALIDATION_DIR / "ocr_report.json") or {}
    unresolved = ocr.get("total_pages_needing_ocr", 0)
    checks.append(condition(
        "OCR / extraction checked",
        unresolved == 0,
        f"{unresolved} page(s) still need OCR" if unresolved else "no pages need OCR",
    ))

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    categories = read_jsonl(CORPUS_DIR / "categories.jsonl")

    checks.append(condition(
        "Corpus normalized",
        bool(documents) and all((NORMALIZED_DIR / f"{d['doc_id']}.txt").is_file()
                                for d in documents),
        f"{len(documents)} documents",
    ))

    paragraph_report = read_json(VALIDATION_DIR / "paragraph_integrity_report.json") or {}
    checks.append(condition(
        "Paragraph segmentation frozen and offset-exact",
        paragraph_report.get("status") == "PASS" and bool(paragraphs),
        f"{len(paragraphs)} paragraphs, status={paragraph_report.get('status')}",
    ))

    approved_categories = [c for c in categories if c.get("review_status") == "APPROVED"]
    checks.append(condition(
        "Categories validated and legally reviewed",
        len(categories) == 26 and len(approved_categories) == 26,
        f"{len(approved_categories)}/{len(categories)} approved",
    ))

    tiered = [d for d in documents if d.get("tier")]
    checks.append(condition(
        "Tier assignments reviewed",
        bool(documents) and len(tiered) == len(documents),
        f"{len(tiered)}/{len(documents)} documents tiered",
    ))

    core = [d for d in documents if d.get("split") == "core"]
    checks.append(condition(
        "Core split selected (6-32 documents, amendment A-6)",
        6 <= len(core) <= 32,  # amendment A-2
        f"{len(core)} core documents",
    ))

    annotations_path = current_annotations_path()
    cells = read_jsonl(annotations_path)
    annotated = [c for c in cells if c.get("is_impossible") is not None]
    checks.append(condition(
        "Annotation completed",
        bool(cells) and len(annotated) == len(cells),
        f"{len(annotated)}/{len(cells)} cells annotated (from {annotations_path.name})",
    ))

    high_contest = [c for c in cells
                    if c.get("split") == "core" and c.get("attention_tier") == "high_contest"]
    doubled = [c for c in high_contest if len(c.get("annotator_ids") or []) >= 2]
    checks.append(condition(
        "High-contest cells double annotated",
        bool(high_contest) and len(doubled) == len(high_contest),
        f"{len(doubled)}/{len(high_contest)} high-contest core cells have 2+ annotators",
    ))

    for label, report_name in (
        ("Span validation passed", "span_report.json"),
        ("Query consistency passed", "query_consistency_report.json"),
        ("Split leakage check passed", "split_leakage_report.json"),
        ("Duplicate check passed", "duplicate_report.json"),
    ):
        report = read_json(VALIDATION_DIR / report_name) or {}
        checks.append(condition(
            label, report.get("status") == "PASS", f"status={report.get('status')}",
        ))

    retrieval = read_json(VALIDATION_DIR / "retrieval_report.json") or {}
    checks.append(condition(
        "Retrieval sanity check passed",
        retrieval.get("status") == "PASS",
        f"status={retrieval.get('status')}, "
        f"recall@10={(retrieval.get('positive') or {}).get('recall@10')}",
    ))

    review_path = PROJECT_ROOT / "HUMAN_REVIEW_REQUIRED.md"
    blocking_open = review_path.is_file() and "**0 blocking**" not in review_path.read_text(
        encoding="utf-8"
    )
    checks.append(condition(
        "Human review queue resolved",
        not blocking_open,
        "blocking items remain in HUMAN_REVIEW_REQUIRED.md" if blocking_open else "clear",
    ))

    return checks


FROZEN_ARTIFACTS = [
    CORPUS_DIR / "documents.jsonl",
    CORPUS_DIR / "paragraphs.jsonl",
    CORPUS_DIR / "categories.jsonl",
    ANNOTATIONS_DIR / "annotations_raw.jsonl",
    ANNOTATIONS_DIR / "annotations_adjudicated.jsonl",
    QA_DIR / "qa.jsonl",
    RETRIEVAL_DIR / "retrieval_config.json",
]


def write_manifest() -> int:
    lines = [
        "IndiCUAD frozen corpus manifest",
        f"generated: {utc_now()}",
        "",
        "SHA-256 of each frozen artifact:",
        "",
    ]
    count = 0
    for path in FROZEN_ARTIFACTS:
        if path.is_file():
            lines.append(f"{sha256_file(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}")
            count += 1
    lines.append("")
    lines.append("Normalized document texts (the reference for all gold offsets):")
    lines.append("")
    for path in sorted(NORMALIZED_DIR.glob("*.txt")):
        lines.append(f"{sha256_file(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}")
        count += 1

    (PROJECT_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def main() -> int:
    stage = Stage("18", "Freeze readiness")
    checks = evaluate()

    passed = sum(1 for c in checks if c["passed"])
    for check in checks:
        mark = "x" if check["passed"] else " "
        stage.info(f"  [{mark}] {check['condition']:<48} {check['detail']}")

    ready = passed == len(checks)
    stage.info("")
    stage.info(f"  {passed}/{len(checks)} freeze conditions met")

    write_json(
        VALIDATION_DIR / "freeze_readiness_report.json",
        {
            "generated_at": utc_now(),
            "status": "READY_TO_FREEZE" if ready else "NOT_READY",
            "conditions_met": passed,
            "conditions_total": len(checks),
            "checks": checks,
        },
    )

    if "--write-manifest" in sys.argv:
        if not ready:
            stage.fail("refusing to write MANIFEST.txt — freeze conditions not met.")
            return 1
        count = write_manifest()
        stage.info(f"wrote MANIFEST.txt ({count} hashed artifacts)")
        return 0

    if not ready:
        stage.warn("not ready to freeze; see the unchecked conditions above.")
    else:
        stage.info("all conditions met — rerun with --write-manifest to freeze.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
