#!/usr/bin/env python3
"""Appendix C: re-run the headline intervals as a document-level bootstrap.

Intent groups from the same contract are not fully independent, so every
headline interval is recomputed resampling the 43 contracts (doc_id) instead of
the 507 intent groups, with the same bootstrap as `analyze`
(benchmark/bootstrap.py, seed and resample count from configs/indicuad_v2.yaml).

  1. Table 2: F1 95% CI per model x condition (direct_extraction rows),
     intent-group vs document clusters, and the width ratio doc / intent.
  2. Table 3: the nine paired full_contract F1 comparisons, on the intent groups
     both systems were scored on and with rows as `analyze` uses them
     (direct + structured), under both clusterings; BH over the nine.

Writes results/indicuad_v2/analysis/document_bootstrap.json. CPU only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.bootstrap import benjamini_hochberg, grouped_bootstrap, paired_grouped_bootstrap_diff  # noqa: E402
from benchmark.config import load_config, write_json  # noqa: E402

CONDITIONS = ("full_contract", "oracle_local")
CLUSTERS = ("intent_group_id", "doc_id")
INDIA = ("aalap_fp16", "inlegalllama_fp16", "inlegalbert")
REFERENCES = ("qwen3_8b_fp16", "saullm_7b_fp16", "cuad_legalbert_or_roberta")


def load_scores(cfg, model: str, cond: str) -> list[dict]:
    path = cfg.results_dir("scores", model, f"{cond}.jsonl")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("status") == "ok" and r.get("f1") is not None]


def by_cluster(rows: list[dict], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(float(r["f1"]))
    return dict(sorted(out.items()))  # sorted keys, as pandas groupby in `analyze`


def main() -> int:
    cfg = load_config()
    n, seed, level = cfg.bootstrap["n_resamples_default"], cfg.bootstrap["seed"], cfg.bootstrap["ci_level"]
    models = sorted(p.name for p in cfg.results_dir("scores").iterdir() if p.is_dir() and not p.name.startswith("_"))
    out: dict = {"n_resamples": n, "seed": seed, "ci_level": level, "intervals": [], "pairwise": []}

    print("Table 2 F1 95% CI (direct_extraction): intent-group vs document bootstrap")
    for cond in CONDITIONS:
        for model in models:
            rows = [r for r in load_scores(cfg, model, cond) if r["setting"] == "direct_extraction"]
            res = {k: grouped_bootstrap(by_cluster(rows, k), n_resamples=n, seed=seed, ci_level=level) for k in CLUSTERS}
            ig, doc = res["intent_group_id"], res["doc_id"]
            ratio = (doc.ci_high - doc.ci_low) / (ig.ci_high - ig.ci_low)
            out["intervals"].append({
                "model_key": model, "context_condition": cond, "f1": ig.point_estimate,
                "intent_ci": [ig.ci_low, ig.ci_high], "n_intent_groups": ig.n_groups,
                "document_ci": [doc.ci_low, doc.ci_high], "n_documents": doc.n_groups,
                "width_ratio": ratio,
            })
            print(f"  {cond:13s} {model:26s} F1={ig.point_estimate:.3f} "
                  f"intent [{ig.ci_low:.3f},{ig.ci_high:.3f}]  doc [{doc.ci_low:.3f},{doc.ci_high:.3f}]  x{ratio:.2f}")
    ratios = [r["width_ratio"] for r in out["intervals"]]
    print(f"  width ratio range: {min(ratios):.2f}-{max(ratios):.2f}")

    print("\nTable 3 (full_contract F1, paired): intent-group vs document bootstrap, BH over the nine")
    results = {k: [] for k in CLUSTERS}
    pairs = [(a, b) for a in INDIA for b in REFERENCES]
    for a, b in pairs:
        ra, rb = load_scores(cfg, a, "full_contract"), load_scores(cfg, b, "full_contract")
        shared = {r["intent_group_id"] for r in ra} & {r["intent_group_id"] for r in rb}
        ra = [r for r in ra if r["intent_group_id"] in shared]
        rb = [r for r in rb if r["intent_group_id"] in shared]
        for k in CLUSTERS:
            results[k].append(paired_grouped_bootstrap_diff(by_cluster(ra, k), by_cluster(rb, k),
                                                            n_resamples=n, seed=seed, ci_level=level))
    bh = {k: benjamini_hochberg([c.bootstrap_p_value for c in results[k]]) for k in CLUSTERS}
    for i, (a, b) in enumerate(pairs):
        entry = {"model_a": a, "model_b": b}
        line = f"  {a:18s} vs {b:26s}"
        for k in CLUSTERS:
            c, adj = results[k][i], bh[k][i]
            entry[k] = {"point_diff": c.point_diff, "ci": [c.ci_low, c.ci_high], "n_clusters": c.n_groups,
                        "p_value": c.bootstrap_p_value, "bh_adjusted_p": adj["adjusted_p_value"],
                        "bh_significant": adj["reject_null"]}
            star = "*" if adj["reject_null"] else " "
            line += f"  {k.split('_')[0]:6s} {c.point_diff:+.3f}{star} [{c.ci_low:+.3f},{c.ci_high:+.3f}] p_BH={adj['adjusted_p_value']:.3f}"
        changed = entry["intent_group_id"]["bh_significant"] != entry["doc_id"]["bh_significant"]
        entry["significance_changes"] = changed
        out["pairwise"].append(entry)
        print(line + ("   <- significance changes" if changed else ""))

    path = cfg.results_dir("analysis", "document_bootstrap.json")
    write_json(path, out)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
