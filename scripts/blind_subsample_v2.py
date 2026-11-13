"""Blind subsample over qa/qa.json — the only source of a defensible kappa.

Why this exists rather than reusing the review passes already done: in those
passes each reviewer saw the existing label before judging, so an unchanged cell
records confirmation, not independent agreement. Agreement computed that way is
anchored upward; agreement computed only on cells where a reviewer bothered to
record a verdict is anchored downward, because reviewers only wrote something
where they disagreed. Neither is Cohen's kappa. Both annotators must judge the
same cells, from scratch, with the label hidden.

    blind_subsample_v2.py --emit --annotators "Akshay Kumar,Kanishka" [--n 60]
        Draws a stratified sample of CELLS, writes one fill-in CSV and one
        reading worksheet per annotator (identical cells, independently
        judged), plus a sealed key nobody opens until scoring.

    blind_subsample_v2.py --score
        Raw agreement and Cohen's kappa between annotators, each annotator
        against the dataset label, bootstrap CI, and a per-stratum breakdown.

The unit is the cell (intent_group_id), never the query row: the three question
registers of a cell are not independent judgments.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / "qa" / "qa.json"
OUT = ROOT / "annotations" / "blind"
KEY = OUT / "_SEALED_KEY.do-not-open.json"
VERDICTS = ("present", "absent")


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def cohen_kappa(a: list[bool], b: list[bool]) -> tuple[float, float, float | None]:
    """Return (observed agreement, chance agreement, kappa)."""
    n = len(a)
    if n == 0:
        return 0.0, 0.0, None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, pe, ((po - pe) / (1 - pe) if pe < 1 else None)


def kappa_ci(a: list[bool], b: list[bool], iters: int = 5000, seed: int = 7) -> tuple:
    """Percentile bootstrap CI — kappa's normal approximation is unreliable at n<100."""
    rng = random.Random(seed)
    n = len(a)
    ks = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        _, _, k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            ks.append(k)
    if len(ks) < iters * 0.5:
        return None, None
    ks.sort()
    return ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))]


def landis_koch(k: float) -> str:
    for edge, label in ((0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                        (0.60, "moderate"), (0.80, "substantial")):
        if k <= edge:
            return label
    return "almost perfect"


def confusion(truth: list[bool], pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(truth, pred) if t and p)
    fp = sum(1 for t, p in zip(truth, pred) if not t and p)
    fn = sum(1 for t, p in zip(truth, pred) if t and not p)
    tn = sum(1 for t, p in zip(truth, pred) if not t and not p)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
def build_cells(rows: list[dict], cats: dict) -> list[dict]:
    """Collapse query rows into cells and attach the stratum."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["intent_group_id"], []).append(r)

    cells = []
    for gid, rs in groups.items():
        r = rs[0]
        tier = cats.get(r["category_id"], {}).get("attention_tier", "unknown")
        cells.append({
            "cell_id": gid,
            "doc_id": r["doc_id"],
            "category_id": r["category_id"],
            "question": next((x["question"] for x in rs
                              if x["question_register"] == "canonical"), r["question"]),
            "split": r["split"],
            "tier": r["tier"],
            "attention_tier": tier,
            "label_present": not r["is_impossible"],   # sealed
            "stratum": f"{r['split']}|{tier}",
        })
    return cells


def stratified(cells: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional allocation, with every stratum guaranteed at least one cell."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for c in cells:
        buckets.setdefault(c["stratum"], []).append(c)

    total = len(cells)
    picked: list[dict] = []
    for name, bucket in sorted(buckets.items()):
        want = max(1, round(n * len(bucket) / total))
        picked += rng.sample(bucket, min(want, len(bucket)))

    rng.shuffle(picked)
    return picked[:n] if len(picked) > n else picked


def worksheet_html(cells: list[dict], paras: dict, cats: dict, who: str) -> str:
    by_doc: dict[str, list[dict]] = {}
    for c in cells:
        by_doc.setdefault(c["doc_id"], []).append(c)

    p = [
        "<title>Blind worksheet — %s</title>" % html.escape(who),
        "<style>body{font:15px/1.6 'Segoe UI',system-ui,sans-serif;max-width:60rem;"
        "margin:0 auto;padding:2rem 1.5rem 6rem;color:#17212F;background:#F6F7F9}"
        "h1{font-family:Georgia,serif}h2{font-family:Georgia,serif;border-top:2px solid #17212F;"
        "padding-top:1rem;margin-top:3rem}"
        ".doc{white-space:pre-wrap;background:#fff;border:1px solid #D2D8E1;border-radius:3px;"
        "padding:1rem;max-height:32rem;overflow:auto;font-size:13.5px}"
        ".cell{background:#fff;border:1px solid #D2D8E1;border-left:3px solid #8A2B34;"
        "border-radius:3px;padding:1rem 1.2rem;margin:1rem 0}"
        ".cid{font-family:Consolas,monospace;font-size:.78rem;color:#616C7D;letter-spacing:.05em}"
        ".rule{font-size:.9rem;color:#3A4657;margin:.4rem 0}"
        "b{font-weight:650}</style>",
        f"<h1>Blind annotation worksheet — {html.escape(who)}</h1>",
        "<p>Judge each cell below <b>present</b> or <b>absent</b> from the contract text "
        "alone. You are not being shown the existing label, and you should not look it up. "
        "Record your verdicts in the CSV that came with this file — one row per cell, "
        "matched by <span class='cid'>cell_id</span>.</p>",
        "<p><b>Present</b> means this document contains operative text for the category. "
        "<b>Absent</b> means it does not, mentions the idea only in passing with no operative "
        "provision, or the category does not apply to this document type. A clause being "
        "unenforceable does not make it absent.</p>",
    ]

    for doc_id, group in sorted(by_doc.items()):
        text = "\n\n".join(x["text"] for x in paras.get(doc_id, []))
        p.append(f"<h2>{html.escape(doc_id)}</h2>")
        p.append(f"<div class='doc'>{html.escape(text)}</div>")
        for c in group:
            cat = cats.get(c["category_id"], {})
            negs = "; ".join(cat.get("negative_examples") or [])
            p.append(
                f"<div class='cell'><div class='cid'>{c['cell_id']} · "
                f"{html.escape(cat.get('attention_tier', ''))}</div>"
                f"<p><b>{html.escape(cat.get('category_name', c['category_id']))}</b></p>"
                f"<p>{html.escape(c['question'])}</p>"
                f"<p class='rule'><b>Boundary rule:</b> {html.escape(cat.get('boundary_rule', ''))}</p>"
                f"<p class='rule'><b>Do not confuse with:</b> {html.escape(negs)}</p></div>"
            )
    return "\n".join(p)


def emit(n: int, seed: int, annotators: list[str]) -> int:
    rows = json.load(open(QA, encoding="utf-8"))
    cats = {c["category_id"]: c for c in load_jsonl(ROOT / "corpus" / "categories.jsonl")}
    paras: dict[str, list[dict]] = {}
    for p in load_jsonl(ROOT / "corpus" / "paragraphs.jsonl"):
        paras.setdefault(p["doc_id"], []).append(p)
    for v in paras.values():
        v.sort(key=lambda x: x["paragraph_index"])

    cells = build_cells(rows, cats)
    sample = stratified(cells, n, seed)
    OUT.mkdir(parents=True, exist_ok=True)

    for who in annotators:
        slug = who.replace(" ", "_")
        csv_path = OUT / f"BLIND_{slug}.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell_id", "doc_id", "category_id", "question",
                        "verdict__present_or_absent", "supporting_quote", "note"])
            for c in sample:
                w.writerow([c["cell_id"], c["doc_id"], c["category_id"],
                            c["question"], "", "", ""])
        (OUT / f"WORKSHEET_{slug}.html").write_text(
            worksheet_html(sample, paras, cats, who), encoding="utf-8")
        print(f"  {csv_path.name} + WORKSHEET_{slug}.html")

    KEY.write_text(json.dumps({
        "seed": seed, "n": len(sample), "annotators": annotators,
        "cells": {c["cell_id"]: {"label_present": c["label_present"],
                                 "stratum": c["stratum"],
                                 "attention_tier": c["attention_tier"],
                                 "split": c["split"]} for c in sample},
    }, indent=2), encoding="utf-8")

    strata = {}
    for c in sample:
        strata[c["stratum"]] = strata.get(c["stratum"], 0) + 1
    print(f"\n{len(sample)} cells drawn (seed {seed}) for {len(annotators)} annotator(s)")
    for s, k in sorted(strata.items()):
        print(f"  {s:32s} {k}")
    present = sum(c["label_present"] for c in sample)
    print(f"\nsealed key: {KEY.name}  (do not open before scoring)")
    print(f"dataset marginal in sample: {present} present / {len(sample)-present} absent")
    return 0


# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #
def read_filled(path: Path) -> dict[str, bool]:
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            v = (row.get("verdict__present_or_absent") or "").strip().lower()
            if v in VERDICTS:
                out[row["cell_id"]] = (v == "present")
            elif v:
                print(f"  ! {path.name} {row['cell_id']}: unrecognised verdict {v!r}")
    return out


def report_pair(name_a: str, a: list[bool], name_b: str, b: list[bool]) -> None:
    po, pe, k = cohen_kappa(a, b)
    print(f"\n--- {name_a} vs {name_b} — {len(a)} cells ---")
    print(f"  raw agreement    {po:.4f}   ({sum(1 for x,y in zip(a,b) if x and y)} both-present, "
          f"{sum(1 for x,y in zip(a,b) if not x and not y)} both-absent)")
    print(f"  chance agreement {pe:.4f}")
    if k is None:
        print("  Cohen's kappa    undefined (one rater used a single class)")
        return
    lo, hi = kappa_ci(a, b)
    ci = f"  [95% CI {lo:.3f}, {hi:.3f}]" if lo is not None else ""
    print(f"  Cohen's kappa    {k:.4f}{ci}  — {landis_koch(k)}")


def score() -> int:
    if not KEY.exists():
        print(f"No sealed key at {KEY}. Run --emit first.")
        return 1
    key = json.loads(KEY.read_text(encoding="utf-8"))
    truth = key["cells"]

    filled: dict[str, dict[str, bool]] = {}
    for who in key["annotators"]:
        p = OUT / f"BLIND_{who.replace(' ', '_')}.csv"
        if not p.exists():
            print(f"missing {p.name}")
            continue
        got = read_filled(p)
        if got:
            filled[who] = got
        print(f"{who}: {len(got)}/{key['n']} cells filled")

    if not filled:
        print("\nNothing scored yet — fill the verdict column and re-run.")
        return 1

    names = list(filled)
    # Inter-annotator agreement: the number the paper needs
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = sorted(set(filled[names[i]]) & set(filled[names[j]]))
            if not shared:
                print(f"\n{names[i]} and {names[j]} share no completed cells "
                      f"— kappa not computable.")
                continue
            report_pair(names[i], [filled[names[i]][c] for c in shared],
                        names[j], [filled[names[j]][c] for c in shared])

    if len(names) < 2:
        print("\nOnly one annotator returned verdicts. Inter-annotator kappa needs two "
              "independent judgments of the SAME cells — this run cannot produce one.")

    # Each annotator against the dataset label
    print("\n\n=== dataset label measured against blind human judgment ===")
    for who, got in filled.items():
        shared = sorted(got)
        t = [truth[c]["label_present"] for c in shared]
        p = [got[c] for c in shared]
        po, pe, k = cohen_kappa(t, p)
        cm = confusion(truth=p, pred=t)      # human = truth, dataset = prediction
        print(f"\n--- dataset vs {who} — {len(shared)} cells ---")
        print(f"  raw agreement    {po:.4f}")
        print(f"  Cohen's kappa    {k:.4f} — {landis_koch(k)}" if k is not None
              else "  Cohen's kappa    undefined")
        print(f"  dataset precision on the present class  "
              f"{cm['precision']:.3f}" if cm["precision"] is not None else "  precision n/a")
        print(f"  dataset recall on the present class     "
              f"{cm['recall']:.3f}" if cm["recall"] is not None else "  recall n/a")
        print(f"  confusion  tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}")
        print(f"  {cm['fn']} cell(s) the dataset calls empty that a human found text in")

        by: dict[str, list[str]] = {}
        for c in shared:
            by.setdefault(truth[c]["attention_tier"], []).append(c)
        print("  by attention tier:")
        for tier, cs in sorted(by.items()):
            tt = [truth[c]["label_present"] for c in cs]
            pp = [got[c] for c in cs]
            _, _, kk = cohen_kappa(tt, pp)
            agree = sum(1 for x, y in zip(tt, pp) if x == y) / len(cs)
            ks = f"{kk:.3f}" if kk is not None else "n/a"
            print(f"    {tier:14s} n={len(cs):3d}  raw={agree:.3f}  kappa={ks}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--annotators", default="Annotator A,Annotator B")
    args = ap.parse_args()

    if args.emit == args.score:
        ap.error("pass exactly one of --emit or --score")
    if args.emit:
        return emit(args.n, args.seed,
                    [a.strip() for a in args.annotators.split(",") if a.strip()])
    return score()


if __name__ == "__main__":
    sys.exit(main())
