"""Stage 14 / §21 — run every deterministic validator, bounded repair loop.

Implements the spec's recursive self-check with MAX_VALIDATION_ITERATIONS,
and only ever repairs deterministic errors. Gold annotations are never
modified by this script; the forbidden loop

    while not perfect: modify_gold_annotations(); validate()

is precisely what the iteration bound and the AUTO_FIX-only repair filter
exist to prevent.

Repairs are applied by repairs.py, which handles a deliberately narrow set
of mechanical defects and logs every change to validation/repair_log.json.
"""

from __future__ import annotations

import sys

from indicuad_common import Stage, emit_report, utc_now, write_json, VALIDATION_DIR
from validators import ALL_VALIDATORS, _load

MAX_VALIDATION_ITERATIONS = 3


def run_all() -> dict[str, dict]:
    data = _load()
    return {name: validator(data) for name, validator in ALL_VALIDATORS.items()}


def summarise(reports: dict[str, dict]) -> dict:
    counts = {"FATAL": 0, "AUTO_FIX": 0, "NEEDS_HUMAN_REVIEW": 0}
    for report in reports.values():
        for key, value in (report.get("finding_counts") or {}).items():
            counts[key] = counts.get(key, 0) + value
    return counts


def main() -> int:
    stage = Stage("14", "Deterministic validation suite")

    try:
        from repairs import apply_repairs
    except ImportError:
        apply_repairs = None

    history = []
    reports: dict[str, dict] = {}

    for iteration in range(1, MAX_VALIDATION_ITERATIONS + 1):
        reports = run_all()
        counts = summarise(reports)
        history.append({"iteration": iteration, "counts": dict(counts)})

        stage.info(
            f"iteration {iteration}: "
            f"FATAL={counts['FATAL']}  AUTO_FIX={counts['AUTO_FIX']}  "
            f"HUMAN={counts['NEEDS_HUMAN_REVIEW']}"
        )

        if counts["FATAL"]:
            stage.fail("fatal findings present — repair loop halted, no auto-fix attempted.")
            break

        if not counts["AUTO_FIX"]:
            break

        if apply_repairs is None or iteration == MAX_VALIDATION_ITERATIONS:
            break

        repaired = apply_repairs(reports)
        stage.info(f"applied {repaired} deterministic repair(s); revalidating")
        if repaired == 0:
            stage.warn("no repair rule matched the remaining AUTO_FIX findings.")
            break

    for name, report in reports.items():
        emit_report(name, report)

    counts = summarise(reports)
    statuses = {name: report.get("status") for name, report in reports.items()}

    if counts["FATAL"]:
        overall = "FATAL"
    elif counts["AUTO_FIX"]:
        overall = "FAIL_UNREPAIRED"
    elif counts["NEEDS_HUMAN_REVIEW"]:
        overall = "NEEDS_HUMAN_REVIEW"
    else:
        overall = "PASS"

    write_json(
        VALIDATION_DIR / "final_qc_report.json",
        {
            "generated_at": utc_now(),
            "status": overall,
            "max_validation_iterations": MAX_VALIDATION_ITERATIONS,
            "iterations_run": len(history),
            "iteration_history": history,
            "finding_counts": counts,
            "validator_status": statuses,
            "gold_annotations_modified": False,
        },
    )

    stage.info("")
    for name, status in statuses.items():
        stage.info(f"  {status:<20} {name}")
    stage.info(f"\n  OVERALL: {overall}")

    return 0 if overall in ("PASS", "NEEDS_HUMAN_REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
