"""Deterministic repairs (spec §5.3 / §22 AUTO_FIX).

Deliberately narrow. A repair belongs here only if it has exactly one
correct outcome that requires no judgement — copying source metadata back
from the manifest, for example. Anything involving what a clause *means*
belongs in the human review queue instead.

Two rules hold without exception:
  1. Gold annotation content is never rewritten. Offsets and answer text
     are inputs to repair decisions, never outputs of them.
  2. Every change is appended to validation/repair_log.json with the
     before and after values.

Structural defects in generated artifacts (bad offsets, duplicate
paragraph IDs) are NOT patched in place: the correct fix is to re-run the
generating stage deterministically, which repairs the whole artifact
consistently rather than one record at a time.
"""

from __future__ import annotations

from indicuad_common import (
    CORPUS_DIR,
    VALIDATION_DIR,
    read_json,
    read_jsonl,
    read_manifest,
    utc_now,
    write_json,
    write_jsonl,
)

# Findings we know how to repair mechanically.
REPAIRABLE_CODES = {"METADATA_DRIFT"}

# Findings that are deterministic but must be fixed by re-running a stage.
REGENERATE_CODES = {
    "INVALID_OFFSETS": "segment_paragraphs.py",
    "OVERLAPPING_PARAGRAPHS": "segment_paragraphs.py",
    "NON_CONTIGUOUS_INDEX": "segment_paragraphs.py",
    "DUPLICATE_ID": "the stage that generated the artifact",
    "MISSING_FIELD": "the stage that generated the artifact",
    "INVALID_TYPE": "the stage that generated the artifact",
}


def _log(entries: list[dict]) -> None:
    path = VALIDATION_DIR / "repair_log.json"
    existing = read_json(path) or {"repairs": []}
    existing["repairs"].extend(entries)
    existing["last_updated"] = utc_now()
    existing["total_repairs"] = len(existing["repairs"])
    write_json(path, existing)


def repair_metadata_drift(findings: list[dict]) -> list[dict]:
    """Restore source metadata verbatim from the manifest."""
    if not findings:
        return []

    manifest = {r["filename"]: r for r in read_manifest()}
    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    by_id = {d["doc_id"]: d for d in documents}

    entries = []
    for finding in findings:
        document = by_id.get(finding.get("doc_id"))
        if document is None:
            continue
        row = manifest.get(document["filename"])
        if row is None:
            continue
        field = finding["field"]
        before = document.get(field)
        after = row[field]
        if before == after:
            continue
        document[field] = after
        entries.append(
            {
                "timestamp": utc_now(),
                "code": "METADATA_DRIFT",
                "artifact": "corpus/documents.jsonl",
                "doc_id": document["doc_id"],
                "field": field,
                "before": before,
                "after": after,
                "rationale": "source metadata restored verbatim from the manifest",
            }
        )

    if entries:
        write_jsonl(CORPUS_DIR / "documents.jsonl", documents)
    return entries


def apply_repairs(reports: dict[str, dict]) -> int:
    """Apply every deterministic repair we know how to make. Returns count."""
    findings: list[dict] = []
    for report in reports.values():
        findings.extend(report.get("findings") or [])

    auto = [f for f in findings if f.get("error_class") == "AUTO_FIX"]
    by_code: dict[str, list[dict]] = {}
    for finding in auto:
        by_code.setdefault(finding["code"], []).append(finding)

    entries = repair_metadata_drift(by_code.get("METADATA_DRIFT", []))

    # Record what we deliberately declined to patch in place.
    for code, findings_for_code in by_code.items():
        if code in REPAIRABLE_CODES:
            continue
        entries.append(
            {
                "timestamp": utc_now(),
                "code": code,
                "action": "NOT_REPAIRED_IN_PLACE",
                "count": len(findings_for_code),
                "remedy": REGENERATE_CODES.get(code, "manual investigation required"),
                "rationale": (
                    "patching generated records individually risks an inconsistent "
                    "artifact; re-run the generating stage instead"
                ),
            }
        )

    if entries:
        _log(entries)
    return len([e for e in entries if e.get("action") != "NOT_REPAIRED_IN_PLACE"])
