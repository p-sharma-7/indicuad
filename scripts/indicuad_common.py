"""Shared helpers for the IndiCUAD dataset pipeline.

Every stage imports from here so that IDs, hashing, and JSONL I/O are
identical across the pipeline. ID construction in particular must never
diverge between stages: paragraph IDs are frozen once annotation begins.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = PROJECT_ROOT / "pdfs"
WORK_DIR = PROJECT_ROOT / "work"
EXTRACTED_DIR = WORK_DIR / "extracted"
NORMALIZED_DIR = WORK_DIR / "normalized"
INTERMEDIATE_DIR = WORK_DIR / "intermediate"
CORPUS_DIR = PROJECT_ROOT / "corpus"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
QA_DIR = PROJECT_ROOT / "qa"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
VALIDATION_DIR = PROJECT_ROOT / "validation"

MANIFEST_CSV = PROJECT_ROOT / "document_source_metadata.csv"

# The protocol is authoritative. It has been seen in the wild under a
# suffixed filename, so resolve rather than assume.
PROTOCOL_CANDIDATES = [
    "ANNOTATION_PROTOCOL.md",
    "ANNOTATION_PROTOCOL (1).md",
]


def find_protocol() -> Path | None:
    for name in PROTOCOL_CANDIDATES:
        candidate = PROJECT_ROOT / name
        if candidate.is_file():
            return candidate
    matches = sorted(PROJECT_ROOT.glob("ANNOTATION_PROTOCOL*.md"))
    return matches[0] if matches else None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Stable identifiers
# --------------------------------------------------------------------------

_DOC_NUM_RE = re.compile(r"^(\d+)_")


def doc_id_for_filename(filename: str) -> str:
    """Derive a stable doc_id from the manifest filename.

    Filenames carry a numeric prefix ("14_NSEEmerge_...") which is the
    corpus's own ordering. Deriving the ID from it keeps doc_ids stable
    across reruns and traceable back to the source row by eye.
    """
    match = _DOC_NUM_RE.match(filename)
    if not match:
        raise ValueError(f"filename does not start with a numeric prefix: {filename!r}")
    return f"DOC-{int(match.group(1)):03d}"


def paragraph_id_for(doc_id: str, index: int) -> str:
    """PAR-DOC001-014 style. Zero-padded so lexical sort == numeric sort."""
    return f"PAR-{doc_id.replace('-', '')}-{index:04d}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# JSON / JSONL I/O
# --------------------------------------------------------------------------


# Annotation files, most-authoritative first. Adjudication supersedes review,
# review supersedes the empty scaffold. Every stage that reads annotations must
# use this order — a stage that reads annotations_raw.jsonl directly validates
# the scaffold instead of the gold and reports a vacuous PASS.
ANNOTATION_CHAIN = [
    "annotations_adjudicated.jsonl",
    "annotations_reviewed.jsonl",
    "annotations_raw.jsonl",
]


def current_annotations_path() -> Path:
    """The authoritative annotation file that currently exists."""
    for name in ANNOTATION_CHAIN:
        candidate = ANNOTATIONS_DIR / name
        if candidate.is_file():
            return candidate
    return ANNOTATIONS_DIR / ANNOTATION_CHAIN[-1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} is not valid JSON: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest() -> list[dict]:
    """Read the source manifest verbatim. Values are never rewritten here."""
    with MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def emit_report(name: str, payload: dict) -> Path:
    """Write a validation report and echo its headline status."""
    payload = {"generated_at": utc_now(), **payload}
    path = VALIDATION_DIR / name
    write_json(path, payload)
    print(f"  wrote validation/{name}  status={payload.get('status', 'n/a')}")
    return path


class Stage:
    """Tiny console formatter so pipeline runs are readable end to end."""

    def __init__(self, number: str, title: str) -> None:
        self.number = number
        self.title = title
        print(f"\n=== Stage {number} — {title} ===")

    @staticmethod
    def info(message: str) -> None:
        print(f"  {message}")

    @staticmethod
    def warn(message: str) -> None:
        print(f"  [WARN] {message}")

    @staticmethod
    def fail(message: str) -> None:
        print(f"  [FAIL] {message}")
