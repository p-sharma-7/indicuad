"""Near-duplicate screen for the core split.

A benchmark gets no information from two documents that say the same
thing. Effective N counts positive intent groups, and twenty of them drawn
from a document that is a verbatim copy of one already in the corpus are
twenty groups a system can answer by having answered the first — the
number goes up and the measurement does not.

This is a live risk for the source class the core split now draws on.
Indian PSUs publish General Conditions of Contract built from a shared
template, so two tenders from different ministries can be word-for-word
identical over forty pages. Measured here, GAIL Gas's GCC for Services
and BCPL's GCC for Services share 92.7% of their 8-word shingles.

Method: 8-word shingles over non-boilerplate paragraph text, punctuation
and case folded away. Two figures per pair —

    Jaccard      |A n B| / |A u B|     symmetric overlap
    containment  |A n B| / min(|A|,|B|)  how much of the smaller is inside
                                         the larger

Containment is the operative one. A short document wholly contained in a
long one has low Jaccard and is still redundant.

Thresholds are advisory, not automatic: a pair over the limit is reported
for a human to decide, because sharing statutory language is normal in
this genre and is not the same thing as being a copy.

    near_duplicate_screen.py [--split core] [--threshold 0.60]

Output: validation/near_duplicate_report.json
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys

from indicuad_common import (
    CORPUS_DIR,
    Stage,
    emit_report,
    read_jsonl,
)

SHINGLE = 8

# Above this containment a pair is reported as a candidate duplicate. Set
# from the observed gap: the BCPL/GAIL copy sits at 0.93, while the highest
# pair that is merely same-genre (the PWD and NIT Meghalaya works
# conditions, which share statutory and safety language) sits at 0.18.
# Anything in between is a judgement call and should be looked at.
DEFAULT_THRESHOLD = 0.60

_WORDS = re.compile(r"[^a-z0-9 ]")


def shingles(text: str, size: int = SHINGLE) -> set[str]:
    words = _WORDS.sub(" ", text.lower()).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="core")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    stage = Stage("4b", "Near-duplicate screen")

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    wanted = {d["doc_id"] for d in documents if d.get("split") == args.split}
    if len(wanted) < 2:
        stage.info(f"fewer than two {args.split} documents — nothing to compare")
        return 0

    grams: dict[str, set[str]] = {doc_id: set() for doc_id in wanted}
    for paragraph in read_jsonl(CORPUS_DIR / "paragraphs.jsonl"):
        if paragraph["doc_id"] in wanted and not paragraph.get("is_boilerplate"):
            grams[paragraph["doc_id"]] |= shingles(paragraph["text"])

    empty = sorted(d for d, g in grams.items() if not g)
    if empty:
        stage.info(f"no indexable text, skipped: {empty}")

    pairs = []
    for a, b in itertools.combinations(sorted(d for d in wanted if grams[d]), 2):
        intersection = len(grams[a] & grams[b])
        if not intersection:
            continue
        union = len(grams[a] | grams[b])
        smaller = min(len(grams[a]), len(grams[b]))
        pairs.append(
            {
                "a": a,
                "b": b,
                "jaccard": round(intersection / union, 4),
                "containment": round(intersection / smaller, 4),
                "shared_shingles": intersection,
            }
        )

    pairs.sort(key=lambda p: p["containment"], reverse=True)
    flagged = [p for p in pairs if p["containment"] >= args.threshold]

    stage.info(f"{args.split} documents compared: {len(wanted)}")
    stage.info(f"pairs sharing any 8-gram: {len(pairs)}")
    for pair in pairs[:8]:
        mark = "  <-- DUPLICATE" if pair["containment"] >= args.threshold else ""
        stage.info(
            f"  {pair['a']} / {pair['b']}  containment={pair['containment']:.3f} "
            f"jaccard={pair['jaccard']:.3f}{mark}"
        )

    if flagged:
        stage.info(
            f"{len(flagged)} pair(s) at or above containment {args.threshold} — "
            "a human decides whether to drop one side."
        )

    emit_report(
        "near_duplicate_report.json",
        {
            "status": "NEEDS_HUMAN_REVIEW" if flagged else "PASS",
            "split": args.split,
            "shingle_size": SHINGLE,
            "threshold": args.threshold,
            "documents_compared": len(wanted),
            "flagged_pairs": flagged,
            "top_pairs": pairs[:20],
            "note": (
                "Containment is |A n B| / min(|A|,|B|) over 8-word shingles of "
                "non-boilerplate text. Flagged pairs are advisory: PSU general "
                "conditions legitimately share statutory language, and only a "
                "reader can say whether a pair is a copy or a genre."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
