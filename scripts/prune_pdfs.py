"""Keep only the source files listed in document_source_metadata.csv.

The licence log is the provenance record of record: one row per file the corpus
actually ships, carrying the retrieval URL, issuing entity and licence position.
It now lists 43 files — the core corpus. Anything in pdfs/ that is not in that
log is a negatives-split or excluded-split source that no longer contributes to
the dataset.

Archives rather than deletes, in the manner of drop_excluded.py. Two of the
files being moved are PROVENANCE PARENTS of documents that ship:

    DOC-001  01_PM_SuryaGhar_Guidelines_ModelAgreements.pdf
             -> DOC-044 (Model Net Metering Agreement, core, ships)
    DOC-027  27_NSE_AbhishekIntegrations_WorkOrder_Feb2026.pdf
             -> DOC-045 (GeM work-order contract, core, ships)

Those two core documents were carved out of their parents' page ranges. Deleting
a parent outright severs the source chain for a document in the release, which
is the exact thing drop_excluded.py refused to do and which the redistribution
plan in PROJECT_CONTEXT §4.4 depends on. Archiving keeps the chain intact while
leaving pdfs/ holding exactly the logged 43.

    python scripts/prune_pdfs.py --dry-run
    python scripts/prune_pdfs.py --apply
    python scripts/prune_pdfs.py --status
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDFS = ROOT / "pdfs"
ARCHIVE = PDFS / "excluded"
LOG = ROOT / "document_source_metadata.csv"
MANIFEST = ARCHIVE / "_archived_manifest.json"

# Recorded in drop_excluded.py; re-stated here so the parent link survives even
# if someone reads only this script.
PARENTS = {
    "DOC-001": "DOC-044",
    "DOC-027": "DOC-045",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logged_filenames() -> set[str]:
    with open(LOG, encoding="utf-8-sig") as fh:
        return {row["filename"].strip() for row in csv.DictReader(fh)
                if row["filename"].strip()}


def status() -> int:
    keep = logged_filenames()
    live = sorted(p.name for p in PDFS.iterdir() if p.is_file())
    archived = sorted(p.name for p in ARCHIVE.iterdir() if p.is_file()
                      and p.name != MANIFEST.name) if ARCHIVE.is_dir() else []
    print(f"licence log lists   {len(keep):>3} file(s)")
    print(f"pdfs/ holds         {len(live):>3} file(s)")
    print(f"pdfs/excluded holds {len(archived):>3} file(s)")
    stray = [n for n in live if n not in keep]
    missing = sorted(keep - set(live))
    print()
    print(f"unlogged files still in pdfs/: {len(stray)}")
    for n in stray[:10]:
        print(f"    {n}")
    print(f"logged files NOT on disk:      {len(missing)}")
    for n in missing[:10]:
        print(f"    {n}")
    return 1 if (stray or missing) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if not LOG.is_file():
        print(f"ABORT: {LOG.name} not found — it is the keep-list.")
        return 1
    if not PDFS.is_dir():
        print("ABORT: pdfs/ not found.")
        return 1

    if args.status:
        return status()

    keep = logged_filenames()
    live = [p for p in sorted(PDFS.iterdir()) if p.is_file()]

    # A logged file that is not on disk means the log and the tree disagree.
    # Refuse rather than silently archive against an incomplete keep-list.
    missing = sorted(keep - {p.name for p in live})
    if missing:
        print(f"ABORT: {len(missing)} file(s) in the licence log are not in pdfs/:")
        for n in missing[:10]:
            print(f"    {n}")
        print("The log is the keep-list; archiving against an incomplete tree "
              "could move a file the log says ships.")
        return 1

    by_filename = {d["filename"]: d
                   for d in read_jsonl(ROOT / "corpus" / "documents.jsonl")
                   + read_jsonl(ROOT / "corpus" / "excluded_documents.jsonl")
                   if d.get("filename")}

    move = [p for p in live if p.name not in keep]
    stay = [p for p in live if p.name in keep]

    print(f"licence log lists  {len(keep):>3} file(s) — these stay")
    print(f"pdfs/ holds        {len(live):>3} file(s)")
    print(f"to archive         {len(move):>3} file(s) -> {ARCHIVE.relative_to(ROOT)}/")
    print(f"remaining in pdfs/ {len(stay):>3} file(s)")

    records = []
    parents_moved = []
    for path in move:
        doc = by_filename.get(path.name, {})
        doc_id = doc.get("doc_id")
        rec = {
            "filename": path.name,
            "doc_id": doc_id,
            "split": doc.get("split"),
            "source_url": doc.get("source_url"),
            "source_entity": doc.get("source_entity"),
            "license_notes": doc.get("license_notes"),
            "size_bytes": path.stat().st_size,
        }
        if doc_id in PARENTS:
            rec["provenance_parent_of"] = PARENTS[doc_id]
            rec["retain_reason"] = (
                f"source chain for {PARENTS[doc_id]}, which ships in the core "
                "split; do not delete")
            parents_moved.append(rec)
        records.append(rec)

    unknown = [r["filename"] for r in records if not r["doc_id"]]
    splits: dict[str, int] = {}
    for r in records:
        splits[r["split"] or "unknown"] = splits.get(r["split"] or "unknown", 0) + 1
    print(f"\narchived by split: {splits}")
    if unknown:
        print(f"{len(unknown)} file(s) have no documents.jsonl record: "
              + ", ".join(unknown[:5]))

    if parents_moved:
        print("\n!! PROVENANCE PARENTS in the archive set — archived, never deleted:")
        for r in parents_moved:
            print(f"   {r['doc_id']}  {r['filename']}")
            print(f"       parent of {r['provenance_parent_of']} (core, ships)")

    if args.dry_run:
        print("\ndry run — nothing moved. Re-run with --apply.")
        return 0

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for path in move:
        dest = ARCHIVE / path.name
        if dest.exists():
            print(f"  skip (already archived): {path.name}")
            continue
        shutil.move(str(path), str(dest))

    for rec in records:
        dest = ARCHIVE / rec["filename"]
        if dest.is_file():
            rec["sha256"] = sha256(dest)

    MANIFEST.write_text(json.dumps({
        "note": "Archived from pdfs/ because they are absent from "
                "document_source_metadata.csv, which lists the 43 core "
                "sources. Nothing here is deleted; two entries are provenance "
                "parents of shipped core documents and must be retained.",
        "keep_list": LOG.name,
        "archived": len(records),
        "files": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    remaining = sorted(p.name for p in PDFS.iterdir() if p.is_file())
    print(f"\nmoved {len(move)} file(s) to {ARCHIVE.relative_to(ROOT)}/")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(f"pdfs/ now holds {len(remaining)} file(s); licence log lists {len(keep)}")
    print(f"exact match: {set(remaining) == keep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
