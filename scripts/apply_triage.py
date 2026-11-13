"""Apply human-reviewed tier and split decisions to corpus/documents.jsonl.

This is the gate between automated triage and annotation. It refuses to
run until a human has filled the human_tier column in triage_report.csv,
because the protocol requires tier assignment to be human-reviewed before
P1 and because the split determines how many query registers each cell
produces.

Fill in triage_report.csv:
    human_tier   A | B | C | D
    human_split  core | negatives | excluded
    reviewer     your initials

human_split may be left blank; it then defaults from the tier
(A/B -> core, C -> negatives, D -> excluded). Tier A/B documents you do
NOT want in the 6-8 document core split must be given human_split
explicitly, since the protocol caps the core split by hand.
"""

from __future__ import annotations

import csv
import sys

from indicuad_common import (
    CORPUS_DIR,
    PROJECT_ROOT,
    Stage,
    emit_report,
    read_jsonl,
    utc_now,
    write_jsonl,
)

VALID_TIERS = {"A", "B", "C", "D"}
VALID_SPLITS = {"core", "negatives", "excluded"}
DEFAULT_SPLIT = {"A": "core", "B": "core", "C": "negatives", "D": "excluded"}

CORE_MIN, CORE_MAX = 6, 48  # A-7  # upper bound raised from 8 by amendment A-2


def main() -> int:
    stage = Stage("6/7-apply", "Apply human triage decisions")

    csv_path = PROJECT_ROOT / "triage_report.csv"
    if not csv_path.is_file():
        stage.fail("triage_report.csv not found — run triage_documents.py first.")
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    decided = [r for r in rows if (r.get("human_tier") or "").strip()]
    if not decided:
        stage.fail(
            "no human_tier values filled in triage_report.csv — "
            "tier assignment must be human-reviewed before P1 (protocol §3)."
        )
        stage.info(f"{len(rows)} rows awaiting review at {csv_path}")
        emit_report(
            "triage_applied_report.json",
            {"status": "BLOCKED_PENDING_HUMAN_REVIEW", "rows_awaiting_review": len(rows)},
        )
        return 1

    if len(decided) != len(rows):
        missing = [r["doc_id"] for r in rows if not (r.get("human_tier") or "").strip()]
        stage.fail(f"{len(missing)} document(s) still have no human_tier: {missing[:10]}")
        return 1

    problems = []
    decisions = {}
    for row in rows:
        tier = row["human_tier"].strip().upper()
        if tier not in VALID_TIERS:
            problems.append(f"{row['doc_id']}: invalid human_tier {tier!r}")
            continue
        split = (row.get("human_split") or "").strip().lower() or DEFAULT_SPLIT[tier]
        if split not in VALID_SPLITS:
            problems.append(f"{row['doc_id']}: invalid human_split {split!r}")
            continue
        if tier == "D" and split != "excluded":
            problems.append(
                f"{row['doc_id']}: Tier D is never annotated, split must be 'excluded'"
            )
            continue
        if tier in ("C",) and split == "core":
            problems.append(
                f"{row['doc_id']}: Tier C belongs to the negatives split (protocol §3)"
            )
            continue
        decisions[row["doc_id"]] = {
            "tier": tier, "split": split, "reviewer": (row.get("reviewer") or "").strip(),
        }

    core = [d for d, v in decisions.items() if v["split"] == "core"]
    if not (CORE_MIN <= len(core) <= CORE_MAX):
        problems.append(
            f"core split has {len(core)} documents; protocol §2 specifies {CORE_MIN}-{CORE_MAX}. "
            "Set human_split explicitly on the Tier A/B documents you are excluding."
        )

    if problems:
        for problem in problems:
            stage.fail(problem)
        emit_report("triage_applied_report.json", {"status": "FAIL", "problems": problems})
        return 1

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    changes = []
    for document in documents:
        decision = decisions.get(document["doc_id"])
        if decision is None:
            continue
        if document.get("tier") != decision["tier"] or document.get("split") != decision["split"]:
            changes.append(
                {
                    "doc_id": document["doc_id"],
                    "before": {"tier": document.get("tier"), "split": document.get("split")},
                    "after": {"tier": decision["tier"], "split": decision["split"]},
                    "reviewer": decision["reviewer"],
                }
            )
        document["tier"] = decision["tier"]
        document["split"] = decision["split"]
        document["tier_reviewed_by"] = decision["reviewer"]
        document["tier_reviewed_at"] = utc_now()

    write_jsonl(CORPUS_DIR / "documents.jsonl", documents)

    counts: dict[str, int] = {}
    for value in decisions.values():
        key = f"{value['tier']}/{value['split']}"
        counts[key] = counts.get(key, 0) + 1

    for key in sorted(counts):
        stage.info(f"  tier/split {key}: {counts[key]}")
    stage.info(f"core split: {len(core)} documents -> {sorted(core)}")
    stage.info(f"applied {len(changes)} change(s) to corpus/documents.jsonl")

    emit_report(
        "triage_applied_report.json",
        {
            "status": "PASS",
            "core_documents": sorted(core),
            "core_document_count": len(core),
            "tier_split_counts": counts,
            "changes": changes,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
