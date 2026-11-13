"""Fold model-written perturbations into qa/distractors.json, with gates.

Slots 2 and 3 of each record are perturbations of the gold span — the same
clause with its operative detail changed (a different jurisdiction, a different
notice period, a different liability cap). They are written by a model
(prompts/perturb_distractors.v1.md) rather than mined because a mined span from another contract can usually be rejected on
party names alone; a perturbation matches the gold in style, register and
length, so only the substance separates it from the answer.

Input is work/perturbations.jsonl, append-only, one object per line:

    {"distractor_id": "IND-D-00001", "slot": 2,
     "text": "(1) Basic Employment Agreement",
     "perturbation": "instrument type changed from hiring contract"}

Append-only matters: generating 1,302 perturbations takes many passes, and a
crashed or interrupted pass must not cost the work already done. Re-running is
safe — the last line wins for a given (distractor_id, slot).

Four gates, all of which have caught real mistakes:

  * the text must not appear in the source document. A perturbation that
    happens to quote the contract is not a wrong answer, it is a right one —
    this is the gate that matters, and it is why perturbations are checked
    against the frozen text rather than trusted.
  * it must differ from the gold span. A paraphrase that preserves the
    operative meaning is not a distractor.
  * it must differ from the other distractors on the same record.
  * the record and slot must exist, and the slot must be 2 or 3 — slot 1 is
    mined by mine_distractors.py and is not writable here.

    python scripts/merge_perturbations.py --dry-run
    python scripts/merge_perturbations.py --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIDECAR = ROOT / "qa" / "distractors.json"
INBOX = ROOT / "work" / "perturbations.jsonl"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not SIDECAR.is_file():
        print(f"ABORT: {SIDECAR.relative_to(ROOT)} missing — run mine_distractors.py first.")
        return 1
    if not INBOX.is_file():
        print(f"ABORT: {INBOX.relative_to(ROOT)} missing — nothing to merge.")
        return 1

    records = json.loads(SIDECAR.read_text(encoding="utf-8"))
    by_id = {r["distractor_id"]: r for r in records}

    frozen = {p.stem: norm(p.read_text(encoding="utf-8"))
              for p in (ROOT / "work" / "normalized").glob("DOC-*.txt")}

    # Last line wins, so a re-written perturbation supersedes its earlier form.
    incoming: dict[tuple[str, int], dict] = {}
    malformed = []
    for line_no, line in enumerate(INBOX.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            incoming[(row["distractor_id"], int(row["slot"]))] = row
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            malformed.append(f"line {line_no}: {exc}")

    if malformed:
        print(f"ABORT: {len(malformed)} malformed line(s)")
        for m in malformed[:10]:
            print(f"  {m}")
        return 1

    accepted, rejected = [], []
    reasons = Counter()

    for (did, slot), row in sorted(incoming.items()):
        rec = by_id.get(did)
        if rec is None:
            rejected.append((did, slot, "unknown distractor_id"))
            reasons["unknown distractor_id"] += 1
            continue
        if slot not in (2, 3):
            rejected.append((did, slot, "slot must be 2 or 3"))
            reasons["slot must be 2 or 3"] += 1
            continue

        text = (row.get("text") or "").strip()
        if not text:
            rejected.append((did, slot, "empty text"))
            reasons["empty text"] += 1
            continue

        n = norm(text)
        if n in frozen.get(rec["doc_id"], ""):
            rejected.append((did, slot, "appears in source document"))
            reasons["appears in source document"] += 1
            continue
        if n == norm(rec["gold_text"]):
            rejected.append((did, slot, "identical to gold"))
            reasons["identical to gold"] += 1
            continue

        siblings = {norm(d["text"]) for i, d in enumerate(rec["distractors"], 1)
                    if d.get("text") and i != slot}
        if n in siblings:
            rejected.append((did, slot, "duplicate of another distractor"))
            reasons["duplicate of another distractor"] += 1
            continue

        accepted.append((rec, slot, text, row.get("perturbation", "")))

    print(f"perturbations read     {len(incoming):>6,}")
    print(f"  accepted             {len(accepted):>6,}")
    print(f"  rejected             {len(rejected):>6,}")
    for reason, n in reasons.most_common():
        print(f"    {reason:<32} {n:>5,}")
    if rejected:
        print("\n  first rejections:")
        for did, slot, why in rejected[:10]:
            print(f"    {did} slot {slot}: {why}")

    for rec, slot, text, why in accepted:
        rec["distractors"][slot - 1] = {
            "text": text,
            "source": "perturbed",
            "perturbation": why,
        }

    filled = sum(1 for r in records for d in r["distractors"] if d["source"] != "pending")
    total = len(records) * 3
    print(f"\nslots filled           {filled:>6,} / {total:,}  ({filled / total:.0%})")

    if args.dry_run:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    shutil.copyfile(SIDECAR, SIDECAR.with_suffix(".json.bak"))
    with open(SIDECAR, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\nwrote {SIDECAR.relative_to(ROOT)} (.bak beside it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
