"""Record legal review of corpus/categories.jsonl (protocol §7).

Protocol §7 requires a lawyer or law-faculty reviewer to read the 26
category descriptions and boundary rules BEFORE P1. This script records
that it happened; it does not perform it. Run it only after a qualified
person has actually read the file.

    python scripts/approve_categories.py --reviewer "Name, Role" --date 2026-08-12

Approve a subset while the rest is still under review:

    python scripts/approve_categories.py --reviewer "..." --only CAT-008,CAT-011

The datasheet must describe the review accurately. Recording an approval
that did not happen would make the §8 datasheet text false, which is the
one thing this whole pipeline is built to avoid.
"""

from __future__ import annotations

import argparse
import sys

from indicuad_common import CORPUS_DIR, Stage, read_jsonl, utc_now, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True,
                        help='Name and role, e.g. "A. Sharma, Advocate"')
    parser.add_argument("--date", default=None,
                        help="Review date (YYYY-MM-DD). Defaults to now.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated category IDs to approve. Default: all.")
    parser.add_argument("--notes", default="",
                        help="Optional reviewer notes recorded on each category.")
    parser.add_argument("--waive", action="store_true",
                        help="Record that review was DECLINED, not performed. Sets "
                             "review_status=WAIVED_NOT_REVIEWED. The §7 freeze "
                             "condition stays unmet, because a waiver is not a review.")
    args = parser.parse_args()

    if args.waive and args.notes == "":
        parser.error("--waive requires --notes explaining who declined and why; "
                     "the datasheet has to state this accurately")

    stage = Stage("5-review", "Record legal review of categories.jsonl")

    path = CORPUS_DIR / "categories.jsonl"
    categories = read_jsonl(path)
    if not categories:
        stage.fail("corpus/categories.jsonl not found — run build_categories.py first.")
        return 1

    targets = None
    if args.only:
        targets = {c.strip().upper() for c in args.only.split(",") if c.strip()}
        known = {c["category_id"] for c in categories}
        unknown = targets - known
        if unknown:
            stage.fail(f"unknown category id(s): {sorted(unknown)}")
            return 1

    when = args.date or utc_now()
    status = "WAIVED_NOT_REVIEWED" if args.waive else "APPROVED"
    approved = 0
    for category in categories:
        if targets is not None and category["category_id"] not in targets:
            continue
        category["review_status"] = status
        category["reviewed_by"] = args.reviewer
        category["reviewed_at"] = when
        if args.notes:
            category["reviewer_notes"] = args.notes
        approved += 1

    write_jsonl(path, categories)

    remaining = [c["category_id"] for c in categories if c["review_status"] != "APPROVED"]
    verb = "recorded WAIVER of review for" if args.waive else "recorded approval of"
    stage.info(f"{verb} {approved} categor(ies) — {args.reviewer!r} on {when}")
    if args.waive:
        stage.warn(
            "§7 legal review was NOT performed. The freeze condition stays unmet and "
            "the §8 datasheet must state that the category schema was drafted by this "
            "pipeline and never independently reviewed by counsel."
        )
    elif remaining:
        stage.warn(f"{len(remaining)} still pending: {', '.join(remaining)}")
    else:
        stage.info("all 26 categories approved — protocol §7 legal review satisfied")
    stage.info("re-run scripts/generate_review_queue.py to refresh the queue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
