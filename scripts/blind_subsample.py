"""Blind subsample — the measurement amendment A-1 depends on.

705 of 780 cells were decided by a model and approved in bulk. That is
defensible only if someone measures how often the model was wrong, and the
only way to measure it is to have a human annotate a sample **without
seeing the model's answer**, then compare.

This is not a spot-check for reassurance. It produces the precision and
recall figures the datasheet must report in place of describing the corpus
as human-annotated. If the model systematically missed a clause type, this
is where it shows up — and it is the only place it can show up, because
every other number in the pipeline was computed from the model's own
verdicts.

    blind_subsample.py --emit [--n 30] [--seed 20260809]
        Write annotations/BLIND_SUBSAMPLE.jsonl with the cell, the document
        and the category — and NO model verdict. Annotate it from scratch.

    blind_subsample.py --score
        Compare your answers against what the model wrote, and report
        agreement, precision and recall on the positive class.

Stratified so the sample cannot accidentally be all easy cells: it draws
from core-positive, core-empty and negatives in proportion, and always
includes some high-contest cells.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
    write_jsonl,
)

SAMPLE = ANNOTATIONS_DIR / "BLIND_SUBSAMPLE.jsonl"

# One file per annotator, so two of them cannot overwrite each other. The
# sample file holds the QUESTIONS and stays blind; answers live here.
# Falls back to the legacy inline `your_*` fields when the directory is
# empty, so an in-progress worksheet is not orphaned.
ANSWER_DIR = ANNOTATIONS_DIR / "blind"


def load_answers() -> dict[str, dict[str, dict]]:
    """{annotator: {cell_id: answer}} from ANSWER_DIR, plus any inline answers."""
    sets: dict[str, dict[str, dict]] = {}
    if ANSWER_DIR.is_dir():
        for path in sorted(ANSWER_DIR.glob("*.jsonl")):
            answers = {a["cell_id"]: a for a in read_jsonl(path)
                       if a.get("is_impossible") is not None}
            if answers:
                sets[path.stem] = answers
    inline = {
        r["cell_id"]: {
            "cell_id": r["cell_id"],
            "is_impossible": r["your_is_impossible"],
            "spans": r.get("your_spans") or [],
            "notes": r.get("your_notes", ""),
        }
        for r in read_jsonl(SAMPLE)
        if r.get("your_is_impossible") is not None
    }
    if inline:
        sets.setdefault("worksheet", inline)
    return sets


def confusion(gold: dict, answers: dict, cell_ids) -> dict:
    """Score one annotator against the first-pass gold. Positive = clause exists."""
    tp = fp = fn = tn = 0
    disagreements = []
    for cell_id in cell_ids:
        gold_empty = gold[cell_id]["is_impossible"]
        said_empty = answers[cell_id]["is_impossible"]
        if not said_empty and not gold_empty:
            tp += 1
        elif not said_empty and gold_empty:
            fn += 1      # first pass missed a real clause — the dangerous case
        elif said_empty and not gold_empty:
            fp += 1      # first pass proposed a clause this annotator rejects
        else:
            tn += 1
        if said_empty != gold_empty:
            disagreements.append({
                "cell_id": cell_id,
                "first_pass": "empty" if gold_empty else "has clause",
                "annotator": "empty" if said_empty else "has clause",
                "note": answers[cell_id].get("notes", ""),
            })
    n = tp + fp + fn + tn
    return {
        "cells": n,
        "raw_agreement": round((tp + tn) / n, 4) if n else None,
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "precision_positive_class": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall_positive_class": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "kappa": cohen_kappa(tp, fp, fn, tn),
        "disagreements": disagreements,
    }


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Chance-corrected agreement. Raw agreement is misleading here because
    most cells are empty, so two annotators who both always said 'empty'
    would score high."""
    n = tp + fp + fn + tn
    if not n:
        return None
    observed = (tp + tn) / n
    a_pos, b_pos = (tp + fn) / n, (tp + fp) / n
    expected = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def strata(cell: dict) -> str:
    if cell["split"] == "negatives":
        return "negatives"
    if cell.get("is_impossible") is False:
        return "core_positive"
    return "core_empty"


def emit(stage: Stage, n: int, seed: int) -> int:
    cells = [c for c in read_jsonl(current_annotations_path())
             if c.get("is_impossible") is not None]
    if not cells:
        stage.fail("no decided cells to sample from")
        return 1

    model_decided = [c for c in cells
                     if c.get("verdict_source") == "model_proposed_human_approved"]
    if not model_decided:
        stage.fail("no model-decided cells — nothing to validate")
        return 1

    groups: dict[str, list[dict]] = {}
    for cell in model_decided:
        groups.setdefault(strata(cell), []).append(cell)

    rng = random.Random(seed)
    picked: list[dict] = []

    # Deliberately NOT proportional. 572 of the model's verdicts are Reg-30
    # disclosures, which announce a contract rather than containing one —
    # prescreen_negatives.py found no operative content in any of them, so
    # confirming those empty measures almost nothing. The risk this sample
    # exists to measure is a clause hiding in a REAL contract that the model
    # called empty, so core cells are weighted far above their share.
    WEIGHT = {"core_empty": 0.60, "core_positive": 0.20, "negatives": 0.20}
    for name, members in sorted(groups.items()):
        share = round(n * WEIGHT.get(name, 0.0))
        picked.extend(rng.sample(members, min(share, len(members))))

    # Every high-contest cell the model called empty is a place where a
    # subtle clause could have been missed. Make sure some are in the sample.
    high = [c for c in model_decided
            if c["attention_tier"] == "high_contest"
            and c["split"] == "core"
            and c["cell_id"] not in {p["cell_id"] for p in picked}]
    if high:
        picked.extend(rng.sample(high, min(6, len(high))))

    rng.shuffle(picked)
    categories = {c["category_id"]: c for c in read_jsonl(CORPUS_DIR / "categories.jsonl")}

    rows = []
    for cell in picked:
        category = categories[cell["category_id"]]
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "doc_id": cell["doc_id"],
                "document": f"work/normalized/{cell['doc_id']}.txt",
                "category_id": cell["category_id"],
                "category_name": cell["category_name"],
                "attention_tier": cell["attention_tier"],
                "question": category["canonical_question"],
                "boundary_rule": category["boundary_rule"],
                "negative_examples": category["negative_examples"],
                # Your answer. true = no such clause. false = there is one.
                "your_is_impossible": None,
                "your_spans": [],
                "your_notes": "",
            }
        )

    write_jsonl(SAMPLE, rows)
    counts: dict[str, int] = {}
    for cell in picked:
        counts[strata(cell)] = counts.get(strata(cell), 0) + 1

    stage.info(f"wrote {SAMPLE.name} — {len(rows)} cells, seed {seed}")
    for name, count in sorted(counts.items()):
        stage.info(f"  {name:<15} {count}")
    stage.info("The model's verdict is NOT in this file. Read each document and decide.")
    stage.info("Then: blind_subsample.py --score")
    return 0


def score(stage: Stage) -> int:
    rows = read_jsonl(SAMPLE)
    if not rows:
        stage.fail(f"no {SAMPLE.name} — run --emit first")
        return 1
    sampled = [r["cell_id"] for r in rows]

    sets = load_answers()
    if not sets:
        stage.fail(
            f"no answers found. Either fill the `your_*` fields in {SAMPLE.name}, "
            f"or write one file per annotator into {ANSWER_DIR}."
        )
        return 1

    gold = {c["cell_id"]: c for c in read_jsonl(current_annotations_path())}
    report: dict[str, object] = {"status": "PASS", "sample_size": len(rows),
                                 "annotators": {}}

    stage.info(f"sample: {len(rows)} cells   annotators: {', '.join(sorted(sets))}")
    stage.info("")

    for name in sorted(sets):
        answers = sets[name]
        answered = [c for c in sampled if c in answers]
        missing = len(sampled) - len(answered)
        result = confusion(gold, answers, answered)
        result["cells_unanswered"] = missing
        report["annotators"][name] = result

        stage.info(f"--- {name} vs first pass (Claude Opus 5) ---")
        stage.info(f"  answered        {len(answered)} of {len(sampled)}"
                   + (f"   ({missing} outstanding)" if missing else ""))
        stage.info(f"  raw agreement   {result['raw_agreement']}")
        stage.info(f"  Cohen's kappa   {result['kappa']}")
        stage.info(f"  precision       {result['precision_positive_class']}")
        stage.info(f"  recall          {result['recall_positive_class']}")
        stage.info(f"  TP {result['true_positive']}  FP {result['false_positive']}"
                   f"  TN {result['true_negative']}  FN {result['false_negative']}")
        if result["false_negative"]:
            stage.warn(
                f"  {result['false_negative']} clause(s) the first pass called empty, "
                f"{name} says are real. Every such miss is written into gold across "
                "the whole corpus."
            )
        stage.info("")

    # Where two annotators overlap, agreement between THEM is the figure that
    # says whether the task is well specified at all — independent of whether
    # either matches the first pass.
    pairs = {}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [c for c in sampled if c in sets[a] and c in sets[b]]
            if not both:
                continue
            same = sum(1 for c in both
                       if sets[a][c]["is_impossible"] == sets[b][c]["is_impossible"])
            tp = sum(1 for c in both if not sets[a][c]["is_impossible"]
                     and not sets[b][c]["is_impossible"])
            fp = sum(1 for c in both if sets[a][c]["is_impossible"]
                     and not sets[b][c]["is_impossible"])
            fn = sum(1 for c in both if not sets[a][c]["is_impossible"]
                     and sets[b][c]["is_impossible"])
            tn = len(both) - tp - fp - fn
            pairs[f"{a} vs {b}"] = {
                "cells": len(both),
                "raw_agreement": round(same / len(both), 4),
                "kappa": cohen_kappa(tp, fp, fn, tn),
            }
            stage.info(f"--- {a} vs {b} (annotator agreement) ---")
            stage.info(f"  cells compared  {len(both)}")
            stage.info(f"  raw agreement   {pairs[f'{a} vs {b}']['raw_agreement']}")
            stage.info(f"  Cohen's kappa   {pairs[f'{a} vs {b}']['kappa']}")
            stage.info("")
    report["annotator_pairs"] = pairs

    if len(sets) == 1 and "worksheet" not in sets:
        stage.warn(
            "only one annotator answered. Two models agreeing bounds the first "
            "pass; a HUMAN annotator on the same cells is what makes it a "
            "reference standard. See HOW_WE_BUILT_v2.md §4b."
        )

    report["note"] = (
        "Report precision and recall, not agreement alone — with most cells empty, "
        "raw agreement is dominated by the negative class and looks high even if "
        "every positive was missed. Cohen's kappa is the chance-corrected figure. "
        "Positive class = a clause exists; a false negative is the first pass "
        "missing a real clause, which is the error this sample exists to find."
    )
    emit_report("blind_subsample_report.json", report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    stage = Stage("A-5", "Blind subsample")
    if args.emit:
        return emit(stage, args.n, args.seed)
    if args.score:
        return score(stage)
    parser.error("choose --emit or --score")


if __name__ == "__main__":
    sys.exit(main())
