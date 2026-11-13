"""Remove split=="excluded" documents from the live corpus indices.

Archives rather than destroys. Three of the excluded documents are provenance
parents — DOC-044 and DOC-045 (both shipped, both core) were carved out of
DOC-001 and DOC-027 — and that link currently exists only as a comment in
extract_embedded_agreements.py. Deleting the parents outright would sever the
source chain for two documents that ship, which the licence position in
PROJECT_CONTEXT §4.4 depends on. So:

  * excluded records move to corpus/excluded_{documents,paragraphs}.jsonl
  * each child gains a derived_from field naming its parent
  * .bak copies of every rewritten file are left beside the original

    python scripts/drop_excluded.py --dry-run
    python scripts/drop_excluded.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

# Recorded in extract_embedded_agreements.py PLAN; re-stated here so the link
# survives in the data rather than only in code.
DERIVED = {
    "DOC-044": {"derived_from": "DOC-001",
                "derived_note": "Model Net Metering Agreement (Annexure 4) extracted from the "
                                "MNRE PM Surya Ghar guidelines circular."},
    "DOC-045": {"derived_from": "DOC-027",
                "derived_note": "GeM work-order contract extracted from an NSE Reg-30 "
                                "disclosure letter."},
}


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rows, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    docs = read_jsonl(CORPUS / "documents.jsonl")
    paras = read_jsonl(CORPUS / "paragraphs.jsonl")
    qa_docs = {r["doc_id"] for r in json.load(open(ROOT / "qa" / "qa.json", encoding="utf-8"))}

    excluded = {d["doc_id"] for d in docs if d["split"] == "excluded"}

    # Refuse to drop anything the dataset actually uses.
    shipping = excluded & qa_docs
    if shipping:
        print(f"ABORT: these excluded docs appear in qa.json: {sorted(shipping)}")
        return 1

    keep_docs = [d for d in docs if d["doc_id"] not in excluded]
    drop_docs = [d for d in docs if d["doc_id"] in excluded]
    keep_paras = [p for p in paras if p["doc_id"] not in excluded]
    drop_paras = [p for p in paras if p["doc_id"] in excluded]

    for d in keep_docs:
        if d["doc_id"] in DERIVED:
            d.update(DERIVED[d["doc_id"]])

    print(f"documents   {len(docs):>6,} -> {len(keep_docs):>6,}  ({len(drop_docs)} archived)")
    print(f"paragraphs  {len(paras):>6,} -> {len(keep_paras):>6,}  ({len(drop_paras):,} archived)")
    print(f"kept docs match qa.json: "
          f"{ {d['doc_id'] for d in keep_docs} == qa_docs }")
    print("\nprovenance preserved:")
    for cid, meta in DERIVED.items():
        print(f"  {cid}.derived_from = {meta['derived_from']} (archived)")

    if args.dry_run:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    for name in ("documents.jsonl", "documents.json", "paragraphs.jsonl", "paragraphs.json"):
        src = CORPUS / name
        if src.exists():
            shutil.copyfile(src, src.with_suffix(src.suffix + ".pre-drop.bak"))

    write_jsonl(CORPUS / "excluded_documents.jsonl", drop_docs)
    write_jsonl(CORPUS / "excluded_paragraphs.jsonl", drop_paras)
    write_jsonl(CORPUS / "documents.jsonl", keep_docs)
    write_jsonl(CORPUS / "paragraphs.jsonl", keep_paras)
    write_json(CORPUS / "documents.json", keep_docs)
    write_json(CORPUS / "paragraphs.json", keep_paras)

    print("\nwrote corpus/excluded_documents.jsonl, corpus/excluded_paragraphs.jsonl")
    print("rewrote documents.{jsonl,json} and paragraphs.{jsonl,json} (.pre-drop.bak beside each)")
    print("\nNow stale and needing a rebuild:")
    print("  SNAPSHOT.txt          — hashes 50 files; four just changed")
    print("  retrieval/bm25/index.pkl — built over the full corpus incl. excluded docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
