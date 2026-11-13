"""Attach a readable context window to every answerable query row.

A gold paragraph on its own is often too thin to interpret. The median
paragraph in the corpus is 232 characters — frequently a bare sub-clause
whose subject, defined terms and governing heading all live in the
paragraph above it. A model handed only the gold paragraph is being asked
to read a sentence fragment; a model handed the whole contract is being
asked to read 622k characters.

So each positive row gains the paragraphs surrounding its gold span, grown
outward until the block reaches a character floor:

    context_paragraph_ids   ordered by paragraph_index, may be discontinuous
    context_char_spans      [[start, end], ...] one pair per contiguous run

Two design points that the data forced:

*Gold paragraphs are not always adjacent.* 354 rows cite two or more, and
they can sit 633 paragraphs apart (IND-Q-02835, indices 548 and 1181) —
a definition near the front and the operative clause near the back. An
earlier version of this script covered gold with one contiguous block,
which turned that row into a 277k-character "window". Gold paragraphs
within CLUSTER_GAP of each other are treated as one passage and the small
gap is filled; beyond that they become separate segments. Hence a list of
spans rather than a single char_start/char_end.

*Offsets matter as much as ids.* `answers[].answer_start` is an offset into
the whole normalized document, so a consumer that slices the context out
needs the segment bases to rebase them; without those the spans point into
the wrong text. Paragraphs are contiguous in document coordinates, so
work/normalized/{doc_id}.txt[start:end] reproduces each segment verbatim —
checked per row before anything is written.

Negatives are left exactly as they are. An impossible question asserts that
a clause appears nowhere in the contract; pairing it with a few-paragraph
window would turn a genuine absence into an artefact of truncation, and a
model could score it right for the wrong reason. Their scope is the whole
document, which doc_id already names.

    python scripts/add_context_paragraphs.py --dry-run
    python scripts/add_context_paragraphs.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
QA_FILE = ROOT / "qa" / "qa.json"

# Grow each segment until it holds this many characters. Chosen against the
# corpus: the median paragraph is 232 chars, so this lands at roughly six
# paragraphs of surrounding clause text.
MIN_CHARS = 1500

# Gold paragraphs this close are one passage with a hole in it — fill the
# hole. Further apart they are distinct provisions and get their own segment.
CLUSTER_GAP = 3

# Backstop for pathological segmentation — one paragraph in the corpus is a
# single character. Without a cap, a run of those would walk half a document.
MAX_PARAGRAPHS = 25


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rows, ensure_ascii=False, indent=2))


def cluster(gold_idx: list[int]) -> list[tuple[int, int]]:
    """Group gold paragraph indices into [lo, hi] runs, merging small gaps."""
    runs: list[tuple[int, int]] = []
    for i in sorted(set(gold_idx)):
        if runs and i - runs[-1][1] <= CLUSTER_GAP:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def grow(paras: list[dict], lo: int, hi: int) -> tuple[int, int]:
    """Widen [lo, hi] until it holds MIN_CHARS or hits a limit.

    Expands left first, then alternates. Left leads because a clause heading
    or defined term normally sits above the operative sentence in a contract.
    """

    def size() -> int:
        return paras[hi]["char_end"] - paras[lo]["char_start"]

    take_left = True
    while size() < MIN_CHARS and (hi - lo + 1) < MAX_PARAGRAPHS:
        can_left, can_right = lo > 0, hi < len(paras) - 1
        if not can_left and not can_right:
            break
        if take_left and can_left:
            lo -= 1
        elif can_right:
            hi += 1
        elif can_left:
            lo -= 1
        take_left = not take_left

    return lo, hi


def merge(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce runs that overlap or touch after growing."""
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(runs):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    paras = read_jsonl(CORPUS / "paragraphs.jsonl")
    qa = json.load(open(QA_FILE, encoding="utf-8"))

    by_doc: dict[str, list[dict]] = {}
    for p in paras:
        by_doc.setdefault(p["doc_id"], []).append(p)
    for v in by_doc.values():
        v.sort(key=lambda p: p["paragraph_index"])

    by_id = {p["paragraph_id"]: p for p in paras}

    # Frozen document text, for verifying that the emitted offsets really do
    # slice out the paragraphs we claim they do.
    frozen: dict[str, str] = {}
    for doc_id in by_doc:
        path = ROOT / "work" / "normalized" / f"{doc_id}.txt"
        if path.exists():
            frozen[doc_id] = path.read_text(encoding="utf-8")

    stats = {"positives": 0, "negatives": 0, "no_growth": 0, "gap_filled": 0,
             "hit_cap": 0, "short_doc": 0, "multi_segment": 0}
    sizes: list[int] = []
    added: list[int] = []
    problems: list[str] = []

    for row in qa:
        if row["is_impossible"]:
            stats["negatives"] += 1
            continue
        stats["positives"] += 1

        doc_paras = by_doc[row["doc_id"]]
        gold_idx = [by_id[pid]["paragraph_index"] for pid in row["gold_paragraph_ids"]]

        runs = cluster(gold_idx)
        if any(hi > lo for lo, hi in runs):
            stats["gap_filled"] += 1
        runs = merge([grow(doc_paras, lo, hi) for lo, hi in runs])
        if len(runs) > 1:
            stats["multi_segment"] += 1

        ids: list[str] = []
        spans: list[list[int]] = []
        total = 0
        for lo, hi in runs:
            block = doc_paras[lo : hi + 1]
            ids.extend(p["paragraph_id"] for p in block)
            spans.append([block[0]["char_start"], block[-1]["char_end"]])
            total += block[-1]["char_end"] - block[0]["char_start"]

            if hi - lo + 1 >= MAX_PARAGRAPHS:
                stats["hit_cap"] += 1
            elif block[-1]["char_end"] - block[0]["char_start"] < MIN_CHARS:
                stats["short_doc"] += 1

            # Each segment must slice back out of the frozen document intact.
            text = frozen.get(row["doc_id"])
            if text is not None:
                want = "".join(p["text"] for p in block)
                got = text[block[0]["char_start"] : block[-1]["char_end"]]
                if "".join(got.split()) != "".join(want.split()):
                    problems.append(f"{row['qa_id']}: segment does not match frozen text")

        # Every gold paragraph must survive inside the window built from it.
        if not set(row["gold_paragraph_ids"]) <= set(ids):
            problems.append(f"{row['qa_id']}: window dropped a gold paragraph")
            continue

        # Answer offsets must land inside some segment, or the context is useless.
        for a in row["answers"]:
            if not any(s <= a["answer_start"] and a["answer_end"] <= e for s, e in spans):
                problems.append(f"{row['qa_id']}: answer span falls outside context")

        row["context_paragraph_ids"] = ids
        row["context_char_spans"] = spans

        sizes.append(total)
        added.append(len(ids) - len(set(gold_idx)))
        if len(ids) == len(set(gold_idx)):
            stats["no_growth"] += 1

    if problems:
        print(f"ABORT: {len(problems)} integrity failure(s)")
        for p in problems[:20]:
            print(f"  {p}")
        return 1

    sizes.sort()
    print(f"rows              {len(qa):>6,}  ({stats['positives']:,} positive, "
          f"{stats['negatives']:,} negative — negatives untouched)")
    print(f"context blocks    {len(sizes):>6,}")
    print(f"block chars       min={sizes[0]:,}  median={sizes[len(sizes) // 2]:,}  max={sizes[-1]:,}")
    print(f"paragraphs added  min={min(added)}  mean={sum(added) / len(added):.1f}  max={max(added)}")
    print(f"  gold already >= {MIN_CHARS} chars, no growth : {stats['no_growth']:,}")
    print(f"  small gap between gold filled             : {stats['gap_filled']:,}")
    print(f"  split into 2+ segments (distant gold)     : {stats['multi_segment']:,}")
    print(f"  a segment stopped at the {MAX_PARAGRAPHS}-paragraph cap : {stats['hit_cap']:,}")
    print(f"  a segment stopped at the document edge    : {stats['short_doc']:,}")
    print(f"\noffsets verified against work/normalized/ for {len(frozen)}/{len(by_doc)} documents")

    if args.dry_run:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    shutil.copyfile(QA_FILE, QA_FILE.with_suffix(".json.pre-context.bak"))
    write_json(QA_FILE, qa)
    print(f"\nwrote {QA_FILE.relative_to(ROOT)} (.pre-context.bak beside it)")
    print("\nNow stale and needing a rebuild:")
    print("  SNAPSHOT.txt          — hashes qa.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
