"""Remove is_impossible rows from qa/qa.json, leaving an answers-only set.

Archives rather than destroys, in the manner of drop_excluded.py — the
dropped rows carry five human passes of annotation each (P5-verified,
reviewer verdicts, adjudication notes) and that work is not reproducible
by re-running anything. They move to qa/qa_unanswerable.json intact.

qa_id is NOT renumbered. IND-Q-00027 must stay IND-Q-00027: the reviewer
files at the repo root (qa_akshay_kumar, qa_kanishka) and
qa/reviewer_answers_pending_alignment.csv key on these ids, and resequencing
would silently repoint every one of them at a different question. The
surviving ids are therefore sparse, which is correct.

What this changes, beyond the row count:

  * the negatives split empties completely (390 rows) — every remaining row
    is split=core
  * tier C empties completely; tier B falls 174 -> 33
  * 15 documents keep zero questions. They are deliberately left in
    corpus/ — an unanswerable document is still a legitimate retrieval
    distractor, and removing them would make retrieval look better than it is
  * 1,335 intent groups -> 507, each with a full canonical / semi_formal /
    naive_layperson triple (507 x 3 = 1,521). Significance tests still group
    on intent_group_id; the effective N is 507, not 1,521

    python scripts/drop_unanswerable.py --dry-run
    python scripts/drop_unanswerable.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_FILE = ROOT / "qa" / "qa.json"
ARCHIVE = ROOT / "qa" / "qa_unanswerable.json"


def write_json(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rows, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    qa = json.load(open(QA_FILE, encoding="utf-8"))

    keep = [r for r in qa if not r["is_impossible"]]
    drop = [r for r in qa if r["is_impossible"]]

    if not drop:
        print("nothing to drop — no is_impossible rows remain.")
        return 0

    # A kept row with no answers, or a dropped row with some, would mean
    # is_impossible disagrees with the payload. Neither should exist.
    bad_keep = [r["qa_id"] for r in keep if not r["answers"]]
    bad_drop = [r["qa_id"] for r in drop if r["answers"]]
    if bad_keep or bad_drop:
        print(f"ABORT: {len(bad_keep)} kept row(s) have no answers, "
              f"{len(bad_drop)} dropped row(s) have answers")
        for q in (bad_keep + bad_drop)[:10]:
            print(f"  {q}")
        return 1

    if len({r["qa_id"] for r in keep}) != len(keep):
        print("ABORT: duplicate qa_id among kept rows")
        return 1

    docs_before = {r["doc_id"] for r in qa}
    docs_after = {r["doc_id"] for r in keep}
    groups = Counter(r["intent_group_id"] for r in keep)

    print(f"query rows   {len(qa):>6,} -> {len(keep):>6,}   "
          f"({len(drop):,} archived, {len(drop) / len(qa) * 100:.1f}%)")
    print(f"documents    {len(docs_before):>6} -> {len(docs_after):>6}   "
          f"({len(docs_before - docs_after)} keep zero questions, left in corpus/ as distractors)")
    print(f"categories   {len({r['category_id'] for r in qa}):>6} -> "
          f"{len({r['category_id'] for r in keep}):>6}")
    print(f"intent grps  {len({r['intent_group_id'] for r in qa}):>6,} -> {len(groups):>6,}"
          f"   (effective N for significance tests)")
    print()
    print(f"  split    {dict(Counter(r['split'] for r in keep))}")
    print(f"  tier     {dict(sorted(Counter(r['tier'] for r in keep).items()))}")
    print(f"  register {dict(Counter(r['question_register'] for r in keep))}")
    print(f"  group sizes {dict(sorted(Counter(groups.values()).items()))}")
    print()
    print("  abstention rate  52.8% -> 0.0%  (no negatives remain; the "
          "majority-class baseline and every abstention metric are gone)")
    print(f"  emptied docs     {', '.join(sorted(docs_before - docs_after))}")

    if args.dry_run:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    shutil.copyfile(QA_FILE, QA_FILE.with_suffix(".json.pre-negdrop.bak"))
    write_json(ARCHIVE, drop)
    write_json(QA_FILE, keep)

    print(f"\nwrote {ARCHIVE.relative_to(ROOT)} ({len(drop):,} rows, annotation intact)")
    print(f"rewrote {QA_FILE.relative_to(ROOT)} (.pre-negdrop.bak beside it)")
    print("\nNow stale and needing a rebuild or an edit:")
    print("  SNAPSHOT.txt      — hashes qa.json")
    print("  README.md         — leads with '3,225 query rows, 52.8% abstention'")
    print("  METHODOLOGY.md    — §on abstention as the majority-class baseline")
    print("  PROJECT_CONTEXT.md— row counts and abstention figures")
    print("  scripts/drop_excluded.py — guards excluded docs against qa.json "
          "doc ids; 15 fewer docs now appear there, so it would permit dropping "
          "documents this dataset previously protected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
