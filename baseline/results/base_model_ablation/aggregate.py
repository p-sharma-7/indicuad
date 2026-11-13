"""Build the final results table + bootstrap CIs from outputs/*.jsonl."""
import glob
import json
import os
import random
from collections import defaultdict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

MODELS = ["mistral", "llama2"]
CONDITIONS = ["full_contract", "oracle_local"]
SETTINGS = ["direct_extraction", "structured_extraction"]

N_ROWS_TOTAL = 1521


def load_qa_intent_map():
    import common as c
    rows = c.load_rows()
    return {r["qa_id"]: r["intent_group_id"] for r in rows}


def bootstrap_f1_ci(records, intent_map, n_resamples=2000, seed=0):
    """Cluster bootstrap over intent_group_id (each group ~3 register paraphrases)."""
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for rec in records:
        grp = intent_map.get(rec["qa_id"])
        by_group[grp].append(rec["f1"])
    groups = list(by_group.values())
    if not groups:
        return None
    means = []
    for _ in range(n_resamples):
        sample_groups = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        flat = [v for g in sample_groups for v in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return lo, hi


def main():
    intent_map = load_qa_intent_map()
    table_rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            for setting in SETTINGS:
                path = os.path.join(OUT_DIR, f"{model}_{condition}_{setting}.jsonl")
                if not os.path.exists(path):
                    print(f"MISSING: {path}")
                    continue
                records = [json.loads(l) for l in open(path, encoding="utf-8")]
                n_scored = len(records)
                mean_f1 = sum(r["f1"] for r in records) / n_scored if n_scored else 0.0
                mean_em = sum(r["em"] for r in records) / n_scored if n_scored else 0.0
                coverage = n_scored / N_ROWS_TOTAL
                ci = bootstrap_f1_ci(records, intent_map)
                table_rows.append({
                    "model": model, "condition": condition, "setting": setting,
                    "n_scored": n_scored, "coverage": coverage,
                    "mean_f1": mean_f1, "mean_em": mean_em,
                    "f1_ci95": ci,
                })

    print(f"{'model':<8} {'condition':<14} {'setting':<22} {'n':>5} {'cov':>6} {'F1':>7} {'EM':>7} {'F1 95% CI'}")
    for r in table_rows:
        ci_str = f"[{r['f1_ci95'][0]:.3f}, {r['f1_ci95'][1]:.3f}]" if r["f1_ci95"] else "n/a"
        print(f"{r['model']:<8} {r['condition']:<14} {r['setting']:<22} {r['n_scored']:>5} "
              f"{r['coverage']*100:>5.1f}% {r['mean_f1']:>7.4f} {r['mean_em']:>7.4f} {ci_str}")

    with open(os.path.join(OUT_DIR, "results_table.json"), "w") as f:
        json.dump(table_rows, f, indent=2)


if __name__ == "__main__":
    main()
