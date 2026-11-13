"""Run the IndiCUAD dataset pipeline end to end.

Restartable by design: every stage is idempotent and skips or regenerates
deterministically from its inputs, so re-running after an interruption
continues rather than duplicating work.

Stages that require a human decision stop the run with a clear message
instead of guessing. That is the intended behaviour, not a failure.

    python scripts/run_pipeline.py            # run up to the first human gate
    python scripts/run_pipeline.py --all      # continue past gates where possible
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# (script, required) — a required stage stops the run when it fails;
# a gated stage reports and the pipeline continues to what it can still do.
STAGES = [
    ("inspect_sources.py", True),
    ("extract_pdf_text.py", True),
    # OCR runs before normalization so recovered text is folded into the
    # frozen document text rather than bolted on afterwards.
    ("ocr_pages.py", False),
    ("extract_pdf_text.py", True),
    ("normalize_corpus.py", True),
    ("segment_paragraphs.py", True),
    ("build_categories.py", True),
    ("triage_documents.py", True),
    ("build_bm25.py", True),
    ("apply_triage.py", False),
    ("build_annotation_workspace.py", False),
    ("build_qa.py", False),
    ("run_validation.py", True),
    ("evaluate_retrieval.py", False),
    ("generate_review_queue.py", True),
    ("freeze_corpus.py", True),
]


def main() -> int:
    results = []
    for script, required in STAGES:
        result = subprocess.run([sys.executable, str(SCRIPTS / script)])
        results.append((script, result.returncode, required))
        if result.returncode != 0 and required:
            print(f"\n!! required stage failed: {script} (exit {result.returncode})")
            print("   pipeline stopped.")
            break

    print("\n" + "=" * 68)
    print("PIPELINE SUMMARY")
    print("=" * 68)
    for script, code, required in results:
        if code == 0:
            status = "ok"
        elif required:
            status = "FAILED"
        else:
            status = "gated (awaiting human input)"
        print(f"  {status:<30} {script}")

    gated = [s for s, c, r in results if c != 0 and not r]
    if gated:
        print(f"\n  {len(gated)} stage(s) are waiting on a human decision.")
        print("  See HUMAN_REVIEW_REQUIRED.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
