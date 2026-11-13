#!/usr/bin/env python3
"""Recompute the paper numbers that are not tables written by `analyze`.

  Sec 5.1  base-checkpoint lift and share of the gap to qwen3_8b_fp16 closed
  Sec 5.2  JSON output audit (schema-valid objects, rows with no JSON object)
  Sec 5.3  coverage confound: qwen3_8b_4bit vs qwen3_8b_fp16 / qwen3_32b_4bit

CPU only, standard library. Reads results/indicuad_v2/{raw,scores,analysis} and
results/base_model_ablation/outputs.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "indicuad_v2"
DIRECT, STRUCT = "direct_extraction", "structured_extraction"
FC, OL = "full_contract", "oracle_local"


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def scores(model: str, cond: str, setting: str | None = DIRECT) -> list[dict]:
    rows = [r for r in jsonl(RES / "scores" / model / f"{cond}.jsonl") if r.get("status") == "ok"]
    return [r for r in rows if setting is None or r["setting"] == setting]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def leaderboard_f1(cond: str) -> dict[tuple[str, str], float]:
    with (RES / "analysis" / f"leaderboard_{cond}.csv").open(encoding="utf-8") as f:
        return {(r["model_key"], r["setting"]): float(r["f1.point_estimate"]) for r in csv.DictReader(f)}


def section_5_1() -> None:
    print("\nSec 5.1 matched base checkpoints (direct_extraction F1)")
    table = json.loads((ROOT / "results/base_model_ablation/outputs/results_table.json").read_text(encoding="utf-8"))
    base = {(r["model"], r["condition"], r["setting"]): r["mean_f1"] for r in table}
    for adapted, base_key in (("aalap_fp16", "mistral"), ("inlegalllama_fp16", "llama2")):
        for cond in (FC, OL):
            lb = leaderboard_f1(cond)
            a, b, ref = lb[(adapted, DIRECT)], base[(base_key, cond, DIRECT)], lb[("qwen3_8b_fp16", DIRECT)]
            print(f"  {adapted:18s} {cond:13s} adapted={a:.3f} base={b:.3f} lift={a - b:+.3f} "
                  f"gap-to-qwen3_8b_fp16 closed={100 * (a - b) / (ref - b):.0f}% | "
                  f"structured: adapted={lb[(adapted, STRUCT)]:.4f} base={base[(base_key, cond, STRUCT)]:.4f}")


def section_5_2() -> None:
    # Same object extraction as benchmark/normalize.py (greedy {...} after a failed full parse).
    print("\nSec 5.2 structured_extraction output audit, full_contract (raw rows with status ok)")
    obj_re = re.compile(r"\{.*\}", re.DOTALL)
    for model in ("aalap_fp16", "inlegalllama_fp16", "qwen3_8b_fp16", "saullm_7b_fp16"):
        rows = [r for r in jsonl(RES / "raw" / model / f"{FC}.jsonl")
                if r["setting"] == STRUCT and r["status"] == "ok"]
        valid = no_obj = 0
        for r in rows:
            m = obj_re.search(r["raw_output"] or "")
            if m is None:
                no_obj += 1
                continue
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            valid += isinstance(obj, dict) and isinstance(obj.get("answer"), str)
        n = len(rows)
        print(f"  {model:18s} rows={n:4d} schema-valid={valid} ({100 * valid / n:.1f}%) "
              f"no JSON object={100 * no_obj / n:.1f}%")


def section_5_3() -> None:
    print("\nSec 5.3 coverage confound (direct_extraction F1)")
    for cond in (FC, OL):
        q4, q16, q32 = (mean(r["f1"] for r in scores(m, cond)) for m in
                        ("qwen3_8b_4bit", "qwen3_8b_fp16", "qwen3_32b_4bit"))
        print(f"  {cond:13s} nf4-fp16={100 * (q4 - q16):+.1f} pts   32B-8B(nf4)={100 * (q32 - q4):+.1f} pts")


if __name__ == "__main__":
    section_5_1()
    section_5_2()
    section_5_3()
