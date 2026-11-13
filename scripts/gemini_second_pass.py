"""Second-annotator pass through the Gemini chat UI — emit, then ingest.

The gold in this corpus was proposed by Claude Opus 5 and approved in bulk.
That is defensible only if someone measures how often it was wrong, and the
strongest available measurement is an INDEPENDENT annotator working from
the same documents and the same boundary rules without seeing the first
model's answer. Using a different model family for that is deliberate:
amendment A-2 already bars the annotator from the evaluated system's family
(Llama-3.1), and two annotators drawn from one family would share blind
spots, which is the failure this measurement exists to detect.

This script does the two mechanical halves of that pass and nothing else.
It does not talk to any API — the run happens in the chat UI, by hand, and
the point is that a person can see exactly what was sent and what came
back.

    gemini_second_pass.py --emit
        Write annotations/gemini/<DOC>.md — one paste-ready prompt per
        document, carrying the full document text and every sampled cell
        for it. 66 cells across 41 documents.

    gemini_second_pass.py --ingest
        Read annotations/gemini/replies/<DOC>.json — Gemini's answers — and
        write them into annotations/BLIND_SUBSAMPLE.jsonl so that
        blind_subsample.py --score can compare the two annotators.

What is deliberately NOT in the emitted prompt: Claude's verdict, the
sweep hits, the proposal rationale, and the counts of how many cells are
expected to be positive. The blindness is the measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indicuad_common import (
    ANNOTATIONS_DIR,
    NORMALIZED_DIR,
    Stage,
    emit_report,
    read_jsonl,
    write_jsonl,
)

SAMPLE = ANNOTATIONS_DIR / "BLIND_SUBSAMPLE.jsonl"
# Answers go to their own file so the second annotator cannot overwrite a
# human annotator working the same sample, and so the sample file stays blind.
ANSWER_DIR = ANNOTATIONS_DIR / "blind"
ANSWER_FILE = ANSWER_DIR / "gemini-2.5-pro.jsonl"
PROMPT_DIR = ANNOTATIONS_DIR / "gemini"
REPLY_DIR = PROMPT_DIR / "replies"

# Recorded on every cell this pass touches, so the datasheet can report the
# two annotators separately rather than as one undifferentiated "model".
ANNOTATOR = "gemini-2.5-pro (Google) via Gemini web UI"

SYSTEM_PROMPT = """\
You are annotating contracts for a legal-NLP benchmark called IndiCUAD. It
follows the CUAD formulation: for each (document, clause-category) pair —
one "cell" — you decide whether the document contains text relevant to that
category, and if it does, you quote every passage a reviewing lawyer would
want highlighted.

Five rules govern the task. Follow them literally.

1. "No such clause" is a real answer, not a failure. Most cells are empty.
   Say so plainly and give your reason. Do not stretch a nearby clause to
   fill a category.

2. Recall beats precision. The reader is scanning a contract for things
   they must not miss. When you are genuinely torn, include the passage and
   say in your note that it is a borderline call.

3. Quote VERBATIM. Copy the passage character for character from the
   document text as given, including any typos, doubled words, run-together
   words, placeholder dots and unusual punctuation. Do not clean it up, do
   not paraphrase, do not add ellipses. Your quote is matched against the
   source automatically and a quote that has drifted by one word is
   rejected.

4. Obey the boundary rule for the category, which is supplied with each
   question. It tells you where the category stops. If the boundary rule
   says a licence grant does not count as an assignment, then a licence
   grant does not count, however close it looks.

5. A quoted passage must lie inside a single paragraph. Paragraphs are
   separated by blank lines in the document text. If a clause continues
   past a blank line, give it as two separate quotes.

Output STRICT JSON and nothing else — no prose before or after, no
markdown fence. One object per cell you were asked about:

{"answers": [
  {"cell_id": "<exactly as given>",
   "is_impossible": true | false,
   "spans": ["<verbatim quote>", "<verbatim quote>"],
   "notes": "<one or two sentences: why this is the answer, and any
              boundary call you had to make>"}
]}

is_impossible = true means the clause is NOT in this document, and spans
must then be []. is_impossible = false means it IS, and spans must hold at
least one verbatim quote."""


def emit(stage: Stage) -> int:
    rows = read_jsonl(SAMPLE)
    if not rows:
        stage.fail(f"no {SAMPLE.name} — run blind_subsample.py --emit first")
        return 1

    by_doc: dict[str, list[dict]] = {}
    for row in rows:
        by_doc.setdefault(row["doc_id"], []).append(row)

    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    REPLY_DIR.mkdir(parents=True, exist_ok=True)

    oversize = []
    for doc_id, cells in sorted(by_doc.items()):
        text = (NORMALIZED_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
        if len(text) > 150_000:
            oversize.append((doc_id, len(text)))

        questions = []
        for i, cell in enumerate(cells, start=1):
            negatives = "\n".join(
                f"      - {n}" for n in cell.get("negative_examples") or []
            ) or "      (none given)"
            questions.append(
                f"""### Cell {i} of {len(cells)} — `{cell['cell_id']}`

**Category:** {cell['category_name']} (`{cell['category_id']}`)

**Question:** {cell['question']}

**Boundary rule — this decides the edge cases:** {cell['boundary_rule']}

**Things that are NOT this category:**
{negatives}
"""
            )

        body = f"""<!-- IndiCUAD second-annotator pass · {doc_id} · {len(cells)} cell(s)
     Paste SYSTEM PROMPT first, then this whole file, into one Gemini chat.
     Save the JSON reply to annotations/gemini/replies/{doc_id}.json -->

# Annotate {doc_id} — {len(cells)} cell(s)

Below is the full text of one contract, followed by {len(cells)} question(s)
about it. Answer every question. Return one JSON object as specified.

---

## Contract text — {doc_id}

Quote from this text and nothing else. Paragraphs are separated by blank
lines; a quote may not cross one.

```text
{text}
```

---

## Questions

{chr(10).join(questions)}
---

Return the JSON object now. {len(cells)} answer(s), one per cell_id above.
"""
        (PROMPT_DIR / f"{doc_id}.md").write_text(body, encoding="utf-8")

    (PROMPT_DIR / "SYSTEM_PROMPT.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")

    stage.info(f"wrote {len(by_doc)} prompt file(s) to {PROMPT_DIR}")
    stage.info(f"wrote SYSTEM_PROMPT.txt — paste this first in every session")
    stage.info(f"cells covered: {len(rows)}")
    if oversize:
        stage.info(f"{len(oversize)} document(s) over 150k characters — long for a")
        stage.info("chat paste; use the file-upload control rather than pasting inline:")
        for doc_id, size in sorted(oversize, key=lambda x: -x[1]):
            stage.info(f"    {doc_id}  {size:,} chars")
    stage.info(f"save each reply to {REPLY_DIR}/<DOC-ID>.json, then --ingest")
    return 0


def ingest(stage: Stage) -> int:
    rows = read_jsonl(SAMPLE)
    if not rows:
        stage.fail(f"no {SAMPLE.name}")
        return 1
    by_cell = {r["cell_id"]: r for r in rows}

    replies = sorted(REPLY_DIR.glob("*.json")) if REPLY_DIR.is_dir() else []
    if not replies:
        stage.fail(f"no replies in {REPLY_DIR} — run --emit and do the Gemini pass first")
        return 1

    filled = 0
    collected: list[dict] = []
    problems: list[str] = []
    for path in replies:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"{path.name}: not valid JSON — {error}")
            continue
        answers = payload.get("answers") if isinstance(payload, dict) else payload
        if not isinstance(answers, list):
            problems.append(f"{path.name}: expected an 'answers' list")
            continue
        for answer in answers:
            cell_id = answer.get("cell_id")
            row = by_cell.get(cell_id)
            if row is None:
                problems.append(f"{path.name}: {cell_id!r} is not in the sample")
                continue
            verdict = answer.get("is_impossible")
            if verdict not in (True, False):
                problems.append(f"{cell_id}: is_impossible must be true or false")
                continue
            spans = answer.get("spans") or []
            if verdict is False and not spans:
                problems.append(f"{cell_id}: is_impossible=false but no span given")
                continue
            if verdict is True and spans:
                problems.append(f"{cell_id}: is_impossible=true but spans were given")
                continue
            collected.append({
                "cell_id": cell_id,
                "is_impossible": verdict,
                "spans": spans,
                "notes": answer.get("notes", ""),
                "annotator": ANNOTATOR,
            })
            filled += 1

    for problem in problems[:20]:
        stage.fail(problem)
    if len(problems) > 20:
        stage.fail(f"... and {len(problems) - 20} more")

    answered = {a["cell_id"] for a in collected}
    outstanding = [r["cell_id"] for r in rows if r["cell_id"] not in answered]
    stage.info(f"cells answered by the second annotator: {filled}")
    stage.info(f"cells still outstanding:                {len(outstanding)}")
    if outstanding:
        stage.info(f"    first few: {outstanding[:6]}")

    if problems:
        stage.info("nothing written — fix the replies and re-run")
        emit_report("gemini_pass_report.json",
                    {"status": "FAILED", "problems": problems[:100],
                     "cells_filled": filled, "outstanding": len(outstanding)})
        return 1

    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(ANSWER_FILE, collected)
    stage.info(f"wrote {ANSWER_FILE.relative_to(ANNOTATIONS_DIR.parent)}")
    stage.info("next: blind_subsample.py --score")
    emit_report("gemini_pass_report.json",
                {"status": "PASS", "annotator": ANNOTATOR,
                 "cells_filled": filled, "outstanding": len(outstanding),
                 "replies_read": [p.name for p in replies]})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="write the prompt files")
    parser.add_argument("--ingest", action="store_true", help="read the replies back")
    args = parser.parse_args()
    if not (args.emit or args.ingest):
        parser.error("choose --emit or --ingest")

    stage = Stage("A-5b", "Gemini second-annotator pass")
    if args.emit:
        return emit(stage)
    return ingest(stage)


if __name__ == "__main__":
    sys.exit(main())
