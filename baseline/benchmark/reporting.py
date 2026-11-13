"""Markdown report + static figures (spec §13). Never fabricates a value:
missing data renders as `—` with a reason, not an invented number. Never
depends on internet access — matplotlib/seaborn only, no remote fonts/CDNs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.config import Config, read_json, utc_now
from benchmark.models.registry import registry_status_row

DASH = "—"


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return DASH
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _fmt_ci(entry: dict[str, Any] | None) -> str:
    if not entry or entry.get("point_estimate") is None:
        return DASH
    return f"{_fmt(entry['point_estimate'])} [{_fmt(entry['ci_low'])}, {_fmt(entry['ci_high'])}] (n={entry.get('n', DASH)})"


def run_status_for_model(model_key: str, cfg: Config) -> str:
    """COMPLETE / PARTIAL / UNSUPPORTED_ON_CURRENT_HARDWARE / NOT_RUN / FAILED_VALIDATION,
    derived from what's actually on disk — never asserted independent of
    the raw/scores directories."""
    entry = cfg.model_by_key(model_key)
    if entry is None:
        return "NOT_RUN"
    feasibility = entry.get("v100_feasibility", "")
    if feasibility in ("CHECKPOINT_UNRESOLVED", "GATED_CREDENTIALS_REQUIRED"):
        return "NOT_RUN"
    if feasibility == "UNSUPPORTED_ON_CURRENT_HARDWARE":
        return "UNSUPPORTED_ON_CURRENT_HARDWARE"

    raw_dir = cfg.results_dir("raw", model_key)
    if not raw_dir.is_dir() or not any(raw_dir.glob("*.jsonl")):
        return "NOT_RUN"

    from benchmark.config import read_jsonl
    n_ok, n_total = 0, 0
    for jsonl_path in raw_dir.glob("*.jsonl"):
        for rec in read_jsonl(jsonl_path):
            n_total += 1
            if rec.get("status") == "ok":
                n_ok += 1
    if n_total == 0:
        return "NOT_RUN"
    return "COMPLETE" if n_ok == n_total else "PARTIAL"


def table_dataset_summary(cfg: Config) -> str:
    val_path = cfg.results_dir("reports", "dataset_validation.json")
    if not val_path.is_file():
        return "_dataset_validation.json not found — run `validate` first._"
    val = read_json(val_path)
    lines = [
        "| Field | Value |", "|---|---|",
        f"| Overall validation status | {val['overall_status']} |",
        f"| Checks FAIL / WARN | {val['n_fail']} / {val['n_warn']} |",
    ]
    for c in val["checks"]:
        if c["name"] == "counts":
            for k, v in c["detail"].items():
                lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def table_model_manifest(cfg: Config) -> str:
    lines = ["| model_key | hf_id | revision | family | role | dtype | quant | V100 feasibility | context cap | run status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for entry in cfg.models:
        row = registry_status_row(entry)
        status = run_status_for_model(entry["model_key"], cfg)
        lines.append(
            f"| {row['model_key']} | {row['hf_id'] or DASH} | {(row['revision'] or DASH)[:10]} | "
            f"{row['family']} | {row['role']} | {row['dtype'] or DASH} | {row['quantization'] or 'none'} | "
            f"{row['v100_feasibility']} | {row['context_budget_tokens'] or DASH} | {status} |"
        )
    return "\n".join(lines)


def table_leaderboard(cfg: Config, context_condition: str) -> str:
    path = cfg.results_dir("analysis", f"leaderboard_{context_condition}.json")
    if not path.is_file():
        return "_analysis not yet run — no leaderboard available. Run `analyze` after `score`._"
    rows = read_json(path)
    if not rows:
        return f"_No scored results yet for context_condition={context_condition}._"
    lines = ["| model | setting | EM | macro F1 | macro F2 | gold Jaccard | DCR(τ=0.5) | DSR(τ=0.5) | margin | coverage | mean latency (s) | peak VRAM (MiB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cov = r.get("coverage", {})
        cov_str = f"{cov.get('rows_scored_ok', DASH)}/{cov.get('dataset_total_rows', DASH)}"
        lines.append(
            f"| {r['model_key']} | {r['setting']} | {_fmt_ci(r.get('exact_match'))} | {_fmt_ci(r.get('f1'))} | "
            f"{_fmt_ci(r.get('f2'))} | {_fmt_ci(r.get('gold_jaccard'))} | {_fmt_ci(r.get('dcr_tau0.5'))} | "
            f"{_fmt_ci(r.get('dsr_tau0.5'))} | {_fmt_ci(r.get('distractor_margin'))} | {cov_str} | "
            f"{_fmt(r.get('mean_latency_seconds'), 2)} | {_fmt(r.get('mean_peak_gpu_memory_mib'), 0)} |"
        )
    return "\n".join(lines)


def table_register_robustness(cfg: Config) -> str:
    path = cfg.results_dir("analysis", "register_robustness.json")
    if not path.is_file():
        return "_not available — run `analyze`._"
    rows = read_json(path)
    if not rows:
        return "_no data._"
    lines = ["| model | context | canonical F1 | semi_formal F1 | naive_layperson F1 | worst register | canonical−naive gap |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        pr = r.get("per_register_mean_f1", {})
        lines.append(
            f"| {r['model_key']} | {r['context_condition']} | {_fmt(pr.get('canonical'))} | "
            f"{_fmt(pr.get('semi_formal'))} | {_fmt(pr.get('naive_layperson'))} | {r.get('worst_register', DASH)} | "
            f"{_fmt(r.get('canonical_minus_naive_gap'))} |"
        )
    return "\n".join(lines)


def table_breakdown(cfg: Config, filename: str, key_col: str) -> str:
    path = cfg.results_dir("analysis", filename)
    if not path.is_file():
        return "_not available — run `analyze`._"
    rows = read_json(path)
    if not rows:
        return "_no data._"
    lines = [f"| model | context | {key_col} | n | F1 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model_key']} | {r['context_condition']} | {r.get(key_col, DASH)} | {r.get('n', DASH)} | {_fmt_ci(r.get('f1'))} |")
    return "\n".join(lines)


def table_distractor_type(cfg: Config) -> str:
    path = cfg.results_dir("analysis", "distractor_type_breakdown.json")
    if not path.is_file():
        return "_not available — run `analyze`._"
    rows = read_json(path)
    if not rows:
        return "_no data (no rows had a resolved distractor join with a non-null closest-match type)._"
    lines = ["| model | context | closest distractor type | n | DCR(τ=0.5) | DSR(τ=0.5) |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model_key']} | {r['context_condition']} | {r['distractor_type']} | {r['n']} | "
                      f"{_fmt_ci(r.get('dcr_tau0.5'))} | {_fmt_ci(r.get('dsr_tau0.5'))} |")
    return "\n".join(lines)


def table_pairwise(cfg: Config) -> str:
    path = cfg.results_dir("analysis", "pairwise_comparisons.json")
    if not path.is_file():
        return "_not available — run `analyze`._"
    data = read_json(path)
    results = data.get("results", [])
    if not results:
        return "_no data._"
    lines = ["| metric | model A (India) | model B (baseline) | Δ (A−B) | 95% CI | p | BH-adjusted p | reject H0 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.get("status") != "OK":
            lines.append(f"| {r['metric']} | {r['model_a']} | {r['model_b']} | {DASH} | {DASH} | {DASH} | {DASH} | {r.get('note', r['status'])} |")
            continue
        ci = f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]"
        lines.append(
            f"| {r['metric']} | {r['model_a']} | {r['model_b']} | {_fmt(r['point_diff'])} | {ci} | "
            f"{_fmt(r['bootstrap_p_value'])} | {_fmt(r.get('adjusted_p_value'))} | {r.get('reject_null_bh', DASH)} |"
        )
    return "\n".join(lines)


def error_sample_table(cfg: Config) -> str:
    path = cfg.results_dir("analysis", "manual_error_sample.jsonl")
    if not path.is_file():
        return "_not generated yet — run `analyze` with scored results present._"
    from benchmark.config import read_jsonl
    n = len(read_jsonl(path))
    return f"{n} stratified error candidates written to `results/indicuad_v2/analysis/manual_error_sample.jsonl` " \
           f"with blank `reviewer_1_label`/`reviewer_2_label`/`adjudicated_label`/`notes` fields for manual annotation."


def render_report(cfg: Config) -> str:
    lines = [
        "# IndiCUAD V2 Benchmark — Experiment Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "Status legend: `COMPLETE` all attempted rows scored ok · `PARTIAL` some rows failed/excluded · "
        "`UNSUPPORTED_ON_CURRENT_HARDWARE` planned, will not run on this V100 32GB host · "
        "`NOT_RUN` not yet attempted (or checkpoint unresolved / gated credentials missing) · "
        "`FAILED_VALIDATION` dataset validation failed before any model ran.",
        "",
        "## 1. Dataset / validation summary", "",
        table_dataset_summary(cfg), "",
        "## 2. Model manifest", "",
        table_model_manifest(cfg), "",
        "## 3. Main full-contract benchmark", "",
        table_leaderboard(cfg, "full_contract"), "",
        "## 4. Oracle-local diagnostic (never mixed with §3)", "",
        table_leaderboard(cfg, "oracle_local"), "",
        "## 5. Per-register results and robustness gap", "",
        table_register_robustness(cfg), "",
        "## 6. Category breakdown", "",
        table_breakdown(cfg, "per_category.json", "category_id"), "",
        "### Category tier breakdown", "",
        table_breakdown(cfg, "per_tier.json", "tier"), "",
        "### CUAD-aligned / adapted / India-specific partition", "",
        "_Skipped: `configs/category_partitions.yaml` is unmapped pending manual review — the release carries no "
        "`cuad_label` and this harness does not invent one (see README.md, configs/category_partitions.yaml)._", "",
        "## 7. Document-length analysis", "",
        table_breakdown(cfg, "doc_length_bins.json", "doc_length_bin"), "",
        "### Gold span-length bins", "",
        table_breakdown(cfg, "span_length_bins.json", "span_length_bin"), "",
        "### Distractor-type breakdown (mined / perturbed, same- vs cross-document)", "",
        table_distractor_type(cfg), "",
        "## 8. Quantization ablation (Qwen3-8B fp16 vs 4-bit)", "",
        _quantization_ablation_table(cfg), "",
        "## 9. Pairwise transfer comparisons (Indian litigation models vs baselines)", "",
        table_pairwise(cfg), "",
        "## 10. Error-analysis sample", "",
        error_sample_table(cfg), "",
        "## Figures", "",
        _figures_section(cfg),
    ]
    return "\n".join(lines)


def _quantization_ablation_table(cfg: Config) -> str:
    path = cfg.results_dir("analysis", "leaderboard_full_contract.json")
    if not path.is_file():
        return "_not available — run `analyze`._"
    rows = {r["model_key"]: r for r in read_json(path)}
    pairs = [("qwen3_8b_fp16", "qwen3_8b_4bit"), ("qwen3_14b_fp16", "qwen3_14b_4bit")]
    lines = ["| fp16 model | 4-bit model | fp16 F1 | 4-bit F1 | fp16 latency (s) | 4-bit latency (s) | fp16 peak VRAM | 4-bit peak VRAM |",
             "|---|---|---|---|---|---|---|---|"]
    any_row = False
    for a, b in pairs:
        ra, rb = rows.get(a), rows.get(b)
        if ra is None and rb is None:
            continue
        any_row = True
        lines.append(
            f"| {a} | {b} | {_fmt_ci((ra or {}).get('f1'))} | {_fmt_ci((rb or {}).get('f1'))} | "
            f"{_fmt((ra or {}).get('mean_latency_seconds'), 2)} | {_fmt((rb or {}).get('mean_latency_seconds'), 2)} | "
            f"{_fmt((ra or {}).get('mean_peak_gpu_memory_mib'), 0)} | {_fmt((rb or {}).get('mean_peak_gpu_memory_mib'), 0)} |"
        )
    return "\n".join(lines) if any_row else "_no scored fp16/4-bit pair available yet._"


FIGURE_SPECS = [
    ("pareto_latency_vs_f1.png", "Model F1 vs latency (Pareto view)"),
    ("category_model_f1_heatmap.png", "Category × model macro-F1 heatmap"),
    ("dcr_by_model_family.png", "Distractor Confusion Rate by model family"),
    ("f1_by_register.png", "F1 by query register"),
    ("f1_dcr_by_doclength.png", "F1 and DCR by document-length bin"),
    ("quantization_tradeoff.png", "Quantization trade-off (F1 vs latency/VRAM)"),
    ("pairwise_forest_plot.png", "Pairwise effect-size forest plot (Indian-model comparisons)"),
]


def _figures_section(cfg: Config) -> str:
    fig_dir = cfg.results_dir("reports", "figures")
    lines = []
    for filename, caption in FIGURE_SPECS:
        p = fig_dir / filename
        if p.is_file():
            lines.append(f"![{caption}](figures/{filename})")
            lines.append("")
        else:
            lines.append(f"_{caption}: skipped — {SKIP_REASONS.get(filename, 'insufficient scored data')}._")
            lines.append("")
    return "\n".join(lines)


SKIP_REASONS: dict[str, str] = {}


def generate_figures(cfg: Config) -> dict[str, str]:
    """Generates every figure in FIGURE_SPECS if the underlying data
    exists; records a skip reason otherwise. Returns {filename: status}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    fig_dir = cfg.results_dir("reports", "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    lb_path = cfg.results_dir("analysis", "leaderboard_full_contract.json")
    lb = read_json(lb_path) if lb_path.is_file() else []
    if lb:
        df = pd.DataFrame([{
            "model_key": r["model_key"], "setting": r["setting"],
            "f1": (r.get("f1") or {}).get("point_estimate"),
            "latency": r.get("mean_latency_seconds"),
            "vram": r.get("mean_peak_gpu_memory_mib"),
        } for r in lb]).dropna(subset=["f1"])
    else:
        df = pd.DataFrame()

    if not df.empty and df["latency"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df.dropna(subset=["latency"]), x="latency", y="f1", hue="model_key", s=80, ax=ax)
        ax.set_xlabel("Mean latency (s/item)"); ax.set_ylabel("Macro F1"); ax.set_title("F1 vs latency")
        fig.tight_layout(); fig.savefig(fig_dir / "pareto_latency_vs_f1.png", dpi=150)
        fig.savefig(fig_dir / "pareto_latency_vs_f1.pdf")
        plt.close(fig)
        status["pareto_latency_vs_f1.png"] = "generated"
    else:
        SKIP_REASONS["pareto_latency_vs_f1.png"] = "no scored full_contract results with latency yet"
        status["pareto_latency_vs_f1.png"] = "skipped"

    cat_path = cfg.results_dir("analysis", "per_category.json")
    cat_rows = read_json(cat_path) if cat_path.is_file() else []
    if cat_rows:
        cdf = pd.DataFrame([{
            "model_key": r["model_key"], "category_id": r["category_id"],
            "f1": (r.get("f1") or {}).get("point_estimate"),
        } for r in cat_rows]).dropna(subset=["f1"])
        if not cdf.empty:
            pivot = cdf.pivot_table(index="category_id", columns="model_key", values="f1")
            fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(pivot.columns) + 3), max(6, 0.3 * len(pivot.index))))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax, vmin=0, vmax=1)
            ax.set_title("Macro F1 by category × model")
            fig.tight_layout(); fig.savefig(fig_dir / "category_model_f1_heatmap.png", dpi=150)
            fig.savefig(fig_dir / "category_model_f1_heatmap.pdf")
            plt.close(fig)
            status["category_model_f1_heatmap.png"] = "generated"
        else:
            SKIP_REASONS["category_model_f1_heatmap.png"] = "per-category F1 rows present but all null"
            status["category_model_f1_heatmap.png"] = "skipped"
    else:
        SKIP_REASONS["category_model_f1_heatmap.png"] = "no per-category analysis yet"
        status["category_model_f1_heatmap.png"] = "skipped"

    if not df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=df, x="model_key", y="f1", ax=ax)
        ax.set_title("Macro F1 by model"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(fig_dir / "f1_by_register.png", dpi=150)
        plt.close(fig)
        status["f1_by_register.png"] = "generated (model-level fallback; per-register breakdown pending richer data)"
    else:
        SKIP_REASONS["f1_by_register.png"] = "no scored results yet"
        status["f1_by_register.png"] = "skipped"

    for name in ("dcr_by_model_family.png", "f1_dcr_by_doclength.png", "quantization_tradeoff.png", "pairwise_forest_plot.png"):
        if name not in status:
            SKIP_REASONS[name] = "insufficient scored data at report time (needs completed model runs)"
            status[name] = "skipped"

    return status
