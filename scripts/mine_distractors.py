"""Build qa/distractors.json — three wrong-but-plausible answers per gold span.

Slot 1 is mined here, deterministically, from the corpus itself: a real span
of the SAME category from a DIFFERENT document. Real contract language, no
generation, reproducible from corpus/ alone. Slots 2 and 3 are left empty for
model-written perturbations of the gold (see merge_perturbations.py), which is
what makes a distractor hard — a mined span from another contract can often be
rejected on party names alone, without reading the clause.

Three gates a mined candidate must pass, in order:

  * different document, same category, non-identical text. 25 gold texts recur
    across cells (shared templates), so identity is checked on text, not ids.
  * the candidate's source document must not be a near-duplicate of the target.
    DOC-049/DOC-050 sit at 0.926 containment — Startup India's employment and
    founder-employment templates — so a "wrong" answer mined from one is very
    often the right answer in the other.
  * the candidate text must not appear in the target document. If it does, it
    is not a distractor; it is a second correct answer the annotators missed.

Among survivors, the closest in length to gold wins. Length is the one surface
cue that gives a distractor away for free, and it costs nothing to remove.

    python scripts/mine_distractors.py --dry-run
    python scripts/mine_distractors.py --apply
    python scripts/mine_distractors.py --status
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
QA_FILE = ROOT / "qa" / "qa.json"
OUT = ROOT / "qa" / "distractors.json"

# Pairs at or above this 8-word-shingle containment are treated as the same
# template for mining purposes. 0.5 is the near_duplicate_screen threshold.
DUP_CONTAINMENT = 0.5


def norm(text: str) -> str:
    """Whitespace-insensitive form, for substring checks against frozen text."""
    return re.sub(r"\s+", " ", text).strip().lower()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rows, ensure_ascii=False, indent=2))


def status() -> int:
    if not OUT.is_file():
        print(f"{OUT.relative_to(ROOT)} does not exist — run --apply first.")
        return 1
    records = read_json(OUT)
    slots = Counter()
    for r in records:
        for d in r["distractors"]:
            slots[d["source"]] += 1
    total = len(records) * 3
    filled = sum(v for k, v in slots.items() if k != "pending")
    print(f"records          {len(records):>6,}  (one per gold span)")
    print(f"distractor slots {total:>6,}   filled {filled:,} ({filled / total:.0%})")
    for src, n in sorted(slots.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<24} {n:>6,}")
    pending = [r["distractor_id"] for r in records
               if any(d["source"] == "pending" for d in r["distractors"])]
    if pending:
        print(f"\n{len(pending):,} record(s) still need perturbations; first few:")
        print("  " + ", ".join(pending[:8]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        return status()

    qa = read_json(QA_FILE)
    dup = read_json(ROOT / "validation" / "near_duplicate_report.json")

    # Documents too similar to mine across, both directions.
    twins: dict[str, set[str]] = defaultdict(set)
    for pair in dup.get("top_pairs", []):
        if pair["containment"] >= DUP_CONTAINMENT:
            twins[pair["a"]].add(pair["b"])
            twins[pair["b"]].add(pair["a"])

    frozen = {p.stem: norm(p.read_text(encoding="utf-8"))
              for p in (ROOT / "work" / "normalized").glob("DOC-*.txt")}

    # One cell per (doc, category); its answers are shared by all three
    # register rows, so distractors are generated per cell, not per row.
    cells: dict[tuple[str, str], dict] = {}
    qa_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in qa:
        key = (row["doc_id"], row["category_id"])
        cells.setdefault(key, row)
        qa_ids[key].append(row["qa_id"])

    # Candidate pool per category: every gold span, tagged with its home doc.
    pool: dict[str, list[dict]] = defaultdict(list)
    for (doc_id, cat_id), row in cells.items():
        for a in row["answers"]:
            pool[cat_id].append({
                "text": a["text"],
                "doc_id": doc_id,
                "paragraph_id": a["paragraph_id"],
            })

    records = []
    counter = 0
    reasons = Counter()
    no_candidate = []

    for (doc_id, cat_id), row in sorted(cells.items()):
        for idx, ans in enumerate(row["answers"]):
            counter += 1
            gold = ans["text"]

            candidates = []
            for cand in pool[cat_id]:
                if cand["doc_id"] == doc_id:
                    reasons["same document"] += 1
                    continue
                if cand["doc_id"] in twins[doc_id]:
                    reasons["near-duplicate document"] += 1
                    continue
                if norm(cand["text"]) == norm(gold):
                    reasons["identical to gold"] += 1
                    continue
                if norm(cand["text"]) in frozen.get(doc_id, ""):
                    reasons["present in target document"] += 1
                    continue
                candidates.append(cand)

            mined = None
            if candidates:
                # Closest length to gold — length is a free giveaway otherwise.
                best = min(candidates, key=lambda c: abs(len(c["text"]) - len(gold)))
                mined = {
                    "text": best["text"],
                    "source": "mined",
                    "from_doc_id": best["doc_id"],
                    "from_paragraph_id": best["paragraph_id"],
                }
            else:
                no_candidate.append(f"{doc_id}/{cat_id}")

            distractors = [mined] if mined else []
            while len(distractors) < 3:
                distractors.append({"text": None, "source": "pending"})

            records.append({
                "distractor_id": f"IND-D-{counter:05d}",
                "qa_ids": sorted(qa_ids[(doc_id, cat_id)]),
                "doc_id": doc_id,
                "category_id": cat_id,
                "answer_index": idx,
                "gold_text": gold,
                "gold_paragraph_id": ans["paragraph_id"],
                "distractors": distractors,
            })

    mined_n = sum(1 for r in records if r["distractors"][0]["source"] == "mined")
    print(f"gold spans        {len(records):>6,}")
    print(f"slot 1 mined      {mined_n:>6,}  ({mined_n / len(records):.1%})")
    print(f"slots 2-3 pending {len(records) * 2:>6,}  (model-written perturbations)")
    print("\ncandidates rejected by gate:")
    for reason, n in reasons.most_common():
        print(f"  {reason:<28} {n:>7,}")
    if no_candidate:
        print(f"\n{len(no_candidate)} span(s) had no minable candidate at all:")
        print("  " + ", ".join(sorted(set(no_candidate))[:10]))

    if args.dry_run:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    write_json(OUT, records)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("Next: fill slots 2-3, then scripts/merge_perturbations.py --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
