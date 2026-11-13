"""Aggregation, grouped-bootstrap CIs, and pairwise comparisons (spec §12).

Consumes results/indicuad_v2/scores/{model_key}/{context_condition}.jsonl
(written by benchmark.scoring) and produces the analysis tables under
results/indicuad_v2/analysis/. Every table distinguishes COMPLETE/PARTIAL
coverage explicitly rather than silently averaging over whatever happened
to score successfully.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from benchmark.bootstrap import BootstrapResult, grouped_bootstrap, mean_statistic, paired_grouped_bootstrap_diff, benjamini_hochberg
from benchmark.config import Config, read_jsonl, write_json

METRIC_COLUMNS = ["exact_match", "precision", "recall", "f1", "f2", "gold_jaccard"]
DCR_TAU_DEFAULT = "0.5"


def load_all_scores(cfg: Config, model_keys: list[str] | None = None) -> pd.DataFrame:
    scores_root = cfg.results_dir("scores")
    if not scores_root.is_dir():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(scores_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name == "_smoke_mock":
            continue  # smoke output never enters real analysis
        if model_keys and model_dir.name not in model_keys:
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            rows.extend(read_jsonl(jsonl_path))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "exact_match" in df.columns:
        df["exact_match"] = df["exact_match"].astype(float)
    for tau_col_src in ("dcr_by_tau", "dsr_by_tau"):
        if tau_col_src in df.columns:
            for tau in ("0.3", "0.5", "0.7"):
                df[f"{tau_col_src}_{tau}"] = df[tau_col_src].apply(
                    lambda d, t=tau: float(d.get(t)) if isinstance(d, dict) and d.get(t) is not None else float("nan")
                )
    return df


def _rows_by_group(df: pd.DataFrame, metric: str) -> dict[str, list[float]]:
    sub = df.dropna(subset=[metric])
    out: dict[str, list[float]] = {}
    for group_id, g in sub.groupby("intent_group_id"):
        out[group_id] = g[metric].tolist()
    return out


def _bootstrap_row(df: pd.DataFrame, metric: str, cfg: Config, n_resamples: int | None = None) -> dict[str, Any]:
    boot_cfg = cfg.bootstrap
    n_resamples = n_resamples or boot_cfg["n_resamples_default"]
    rows_by_group = _rows_by_group(df, metric)
    n_ok = sum(len(v) for v in rows_by_group.values())
    if n_ok == 0:
        return {"metric": metric, "n": 0, "point_estimate": None, "ci_low": None, "ci_high": None}
    result: BootstrapResult = grouped_bootstrap(
        rows_by_group, statistic=mean_statistic, n_resamples=n_resamples,
        seed=boot_cfg["seed"], ci_level=boot_cfg["ci_level"],
    )
    return {
        "metric": metric, "n": n_ok, "n_intent_groups": result.n_groups,
        "point_estimate": result.point_estimate, "ci_low": result.ci_low, "ci_high": result.ci_high,
        "n_resamples": result.n_resamples, "seed": result.seed,
    }


def coverage_summary(df_model: pd.DataFrame, cfg: Config, total_rows: int) -> dict[str, Any]:
    n_total = len(df_model)
    n_ok = int((df_model["status"] == "ok").sum()) if "status" in df_model.columns else 0
    n_context_excluded = int((df_model.get("status") == "context_exceeds_model_capacity").sum()) if "status" in df_model.columns else 0
    n_failed = n_total - n_ok - n_context_excluded
    # Strict common-context aggregate (spec §6): among the "ok" rows, how
    # many also fit the shared common_context_cap_tokens floor — this is
    # what a cross-model-fairness comparison should filter to, distinct
    # from rows_scored_ok (every row this model's own real context window
    # could attempt, the run's actual coverage target per operator
    # direction 2026-08-18).
    n_ok_within_common_cap = int(
        df_model[df_model["status"] == "ok"]["exceeds_common_context_cap"].eq(False).sum()
    ) if "exceeds_common_context_cap" in df_model.columns else None
    return {
        "dataset_total_rows": total_rows,
        "rows_attempted": n_total,
        "rows_scored_ok": n_ok,
        "rows_scored_ok_within_common_context_cap": n_ok_within_common_cap,
        "rows_excluded_context_exceeds_model_capacity": n_context_excluded,
        "rows_failed": max(0, n_failed),
        "coverage_fraction_of_dataset": (n_ok / total_rows) if total_rows else None,
    }


def build_leaderboard(cfg: Config, context_condition: str, dataset_total_rows: int) -> list[dict[str, Any]]:
    df = load_all_scores(cfg)
    if df.empty or "context_condition" not in df.columns:
        return []
    df = df[df["context_condition"] == context_condition]
    rows = []
    for (model_key, setting), g in df.groupby(["model_key", "setting"]):
        g_ok = g[g["status"] == "ok"]
        entry: dict[str, Any] = {
            "model_key": model_key, "context_condition": context_condition, "setting": setting,
            "coverage": coverage_summary(g, cfg, dataset_total_rows),
        }
        for metric in METRIC_COLUMNS:
            if metric in g_ok.columns:
                entry[metric] = _bootstrap_row(g_ok, metric, cfg)
        for tau in ("0.3", "0.5", "0.7"):
            for kind in ("dcr_by_tau", "dsr_by_tau"):
                col = f"{kind}_{tau}"
                if col in g_ok.columns:
                    entry[f"{kind[:3]}_tau{tau}"] = _bootstrap_row(g_ok, col, cfg)
        if "distractor_margin" in g_ok.columns:
            entry["distractor_margin"] = _bootstrap_row(g_ok, "distractor_margin", cfg)
        if "latency_seconds" in g_ok.columns and not g_ok["latency_seconds"].dropna().empty:
            entry["mean_latency_seconds"] = float(g_ok["latency_seconds"].dropna().mean())
        if "peak_gpu_memory_mib" in g_ok.columns and not g_ok["peak_gpu_memory_mib"].dropna().empty:
            entry["mean_peak_gpu_memory_mib"] = float(g_ok["peak_gpu_memory_mib"].dropna().mean())
        rows.append(entry)
    return rows


def _breakdown_by(df: pd.DataFrame, cfg: Config, by_col: str, metric: str = "f1") -> list[dict[str, Any]]:
    if df.empty or by_col not in df.columns:
        return []
    out = []
    for (model_key, context_condition, key_val), g in df.groupby(["model_key", "context_condition", by_col]):
        g_ok = g[g["status"] == "ok"]
        if g_ok.empty:
            continue
        row = {"model_key": model_key, "context_condition": context_condition, by_col: key_val, "n": len(g_ok)}
        row.update({f"{metric}": _bootstrap_row(g_ok, metric, cfg)})
        out.append(row)
    return out


def distractor_type_breakdown(df: pd.DataFrame, cfg: Config) -> list[dict[str, Any]]:
    """DCR/DSR by closest-matching distractor type (mined/perturbed) and,
    for mined distractors, same-document vs cross-document source (spec
    §11). Computed over every row with a resolved distractor join, not only
    DCR-positive ones, so the denominator (n) is comparable across types."""
    if df.empty or "closest_distractor_source" not in df.columns:
        return []
    out = []
    for (model_key, context_condition, source), g in df.groupby(["model_key", "context_condition", "closest_distractor_source"]):
        g_ok = g[g["status"] == "ok"]
        if g_ok.empty:
            continue
        row = {"model_key": model_key, "context_condition": context_condition,
               "distractor_type": source, "n": len(g_ok)}
        if "dcr_by_tau_0.5" in g_ok.columns:
            row["dcr_tau0.5"] = _bootstrap_row(g_ok, "dcr_by_tau_0.5", cfg)
        if "dsr_by_tau_0.5" in g_ok.columns:
            row["dsr_tau0.5"] = _bootstrap_row(g_ok, "dsr_by_tau_0.5", cfg)
        out.append(row)

    if "closest_distractor_same_document" in df.columns:
        mined = df[df["closest_distractor_source"] == "mined"]
        for (model_key, context_condition, same_doc), g in mined.groupby(
            ["model_key", "context_condition", "closest_distractor_same_document"]
        ):
            g_ok = g[g["status"] == "ok"]
            if g_ok.empty:
                continue
            row = {"model_key": model_key, "context_condition": context_condition,
                   "distractor_type": "mined_same_document" if same_doc else "mined_cross_document", "n": len(g_ok)}
            if "dcr_by_tau_0.5" in g_ok.columns:
                row["dcr_tau0.5"] = _bootstrap_row(g_ok, "dcr_by_tau_0.5", cfg)
            if "dsr_by_tau_0.5" in g_ok.columns:
                row["dsr_tau0.5"] = _bootstrap_row(g_ok, "dsr_by_tau_0.5", cfg)
            out.append(row)
    return out


def register_robustness(df: pd.DataFrame, cfg: Config) -> list[dict[str, Any]]:
    if df.empty or "question_register" not in df.columns:
        return []
    out = []
    for (model_key, context_condition), g in df.groupby(["model_key", "context_condition"]):
        per_register = {}
        for reg in ("canonical", "semi_formal", "naive_layperson"):
            g_reg = g[(g["question_register"] == reg) & (g["status"] == "ok")]
            if not g_reg.empty:
                per_register[reg] = float(g_reg["f1"].mean())
        if not per_register:
            continue
        worst_register = min(per_register, key=per_register.get)
        entry = {
            "model_key": model_key, "context_condition": context_condition,
            "per_register_mean_f1": per_register,
            "worst_register": worst_register, "worst_register_f1": per_register[worst_register],
        }
        if "canonical" in per_register and "naive_layperson" in per_register:
            entry["canonical_minus_naive_gap"] = per_register["canonical"] - per_register["naive_layperson"]
        out.append(entry)
    return out


def doc_length_bins(df: pd.DataFrame, cfg: Config, n_bins: int = 4) -> list[dict[str, Any]]:
    if df.empty or "doc_char_count" not in df.columns:
        return []
    df = df.copy()
    try:
        df["doc_length_bin"] = pd.qcut(df["doc_char_count"], q=n_bins, duplicates="drop")
        df["doc_length_bin"] = df["doc_length_bin"].astype(str)
    except ValueError:
        return []
    return _breakdown_by(df, cfg, "doc_length_bin")


def span_length_bins(df: pd.DataFrame, cfg: Config, n_bins: int = 4) -> list[dict[str, Any]]:
    if df.empty or "gold_span_char_lengths" not in df.columns:
        return []
    df = df.copy()
    df["gold_span_char_length_primary"] = df["gold_span_char_lengths"].apply(
        lambda lst: lst[0] if isinstance(lst, list) and lst else None
    )
    df = df.dropna(subset=["gold_span_char_length_primary"])
    if df.empty:
        return []
    try:
        df["span_length_bin"] = pd.qcut(df["gold_span_char_length_primary"], q=n_bins, duplicates="drop").astype(str)
    except ValueError:
        return []
    return _breakdown_by(df, cfg, "span_length_bin")


def pairwise_comparisons(cfg: Config, context_condition: str, comparison_family: list[tuple[str, str]]) -> dict[str, Any]:
    """comparison_family: list of (model_a, model_b) model_key pairs to
    compare, e.g. every (india_model, baseline_model) pair per spec §12.
    BH correction applied jointly across the whole family, per metric."""
    df = load_all_scores(cfg)
    if df.empty:
        return {"family": comparison_family, "results": [], "note": "no scores available"}
    df = df[(df["context_condition"] == context_condition) & (df["status"] == "ok")]

    boot_cfg = cfg.bootstrap
    metrics_to_compare = ["f1", "exact_match", "f2", "dcr_by_tau_0.5"]
    raw_results = []
    for metric in metrics_to_compare:
        if metric not in df.columns:
            continue
        for model_a, model_b in comparison_family:
            df_a = df[df["model_key"] == model_a]
            df_b = df[df["model_key"] == model_b]
            if df_a.empty or df_b.empty:
                raw_results.append({
                    "metric": metric, "model_a": model_a, "model_b": model_b,
                    "status": "MISSING_DATA", "note": f"model_a_rows={len(df_a)} model_b_rows={len(df_b)}",
                })
                continue
            rows_a = _rows_by_group(df_a, metric)
            rows_b = _rows_by_group(df_b, metric)
            cmp = paired_grouped_bootstrap_diff(
                rows_a, rows_b, statistic=mean_statistic,
                n_resamples=boot_cfg["n_resamples_default"], seed=boot_cfg["seed"], ci_level=boot_cfg["ci_level"],
            )
            raw_results.append({
                "metric": metric, "model_a": model_a, "model_b": model_b, "status": "OK",
                "point_diff": cmp.point_diff, "ci_low": cmp.ci_low, "ci_high": cmp.ci_high,
                "bootstrap_p_value": cmp.bootstrap_p_value, "n_intent_groups": cmp.n_groups,
            })

    p_values = [r["bootstrap_p_value"] for r in raw_results if r["status"] == "OK"]
    bh = benjamini_hochberg(p_values, alpha=0.05)
    bh_iter = iter(bh)
    for r in raw_results:
        if r["status"] == "OK":
            adj = next(bh_iter)
            r["adjusted_p_value"] = adj["adjusted_p_value"]
            r["reject_null_bh"] = adj["reject_null"]

    return {"family": comparison_family, "context_condition": context_condition, "results": raw_results}


def _error_type_proxy(row: pd.Series) -> str:
    """Cheap, documented error-type proxy (spec §14) — not a substitute for
    human labels, only a stratification key for sampling."""
    if row.get("status") != "ok":
        return "format_failure_or_run_failure"
    align = row.get("alignment_status")
    if align in ("not_found", "invalid_json"):
        return "formatting_alignment_failure"
    dcr = row.get("dcr_by_tau_0.5")
    if dcr == 1.0:
        return "distractor_confusion"
    f1 = row.get("f1")
    if f1 is not None and f1 < 0.2:
        return "zero_low_gold_overlap"
    if f1 is not None and 0.2 <= f1 < 0.8:
        return "boundary_mismatch"
    return "other_low_signal"


def generate_error_sample(cfg: Config, target_min: int = 150, target_max: int = 250, seed: int | None = None) -> list[dict[str, Any]]:
    """Stratified error-analysis sample (spec §14): model family × category
    tier × question register × error-type proxy × document-length bin.
    Blank human-review fields included; no label is ever fabricated here."""
    import numpy as np

    seed = seed if seed is not None else cfg.bootstrap["seed"]
    df = load_all_scores(cfg)
    if df.empty:
        return []
    df = df[df["context_condition"] == "full_contract"].copy()

    errorish = df[(df["status"] != "ok") | (df.get("f1", 1.0) < 0.8)].copy()
    if errorish.empty:
        return []

    errorish["error_type_proxy"] = errorish.apply(_error_type_proxy, axis=1)
    errorish["doc_length_bin"] = pd.qcut(errorish["doc_char_count"], q=min(4, errorish["doc_char_count"].nunique()), duplicates="drop").astype(str) \
        if "doc_char_count" in errorish.columns and errorish["doc_char_count"].nunique() > 1 else "single_bin"
    errorish["family"] = errorish["model_key"].apply(lambda k: (cfg.model_by_key(k) or {}).get("family", k))

    strata_cols = ["family", "tier", "question_register", "error_type_proxy", "doc_length_bin"]
    strata_cols = [c for c in strata_cols if c in errorish.columns]
    rng = np.random.default_rng(seed)

    groups = list(errorish.groupby(strata_cols, dropna=False))
    n_groups = max(1, len(groups))
    per_group_target = max(1, target_max // n_groups)

    sampled_rows = []
    for _, g in groups:
        take = min(len(g), per_group_target)
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        sampled_rows.append(g.loc[idx])

    sampled = pd.concat(sampled_rows) if sampled_rows else errorish.iloc[0:0]
    if len(sampled) < target_min and len(errorish) >= target_min:
        remaining = errorish.drop(index=sampled.index, errors="ignore")
        extra_n = min(len(remaining), target_min - len(sampled))
        if extra_n > 0:
            extra_idx = rng.choice(remaining.index.to_numpy(), size=extra_n, replace=False)
            sampled = pd.concat([sampled, remaining.loc[extra_idx]])
    if len(sampled) > target_max:
        sampled = sampled.sample(n=target_max, random_state=seed)

    dataset = None
    out = []
    for _, row in sampled.iterrows():
        out.append({
            "qa_id": row.get("qa_id"), "model_key": row.get("model_key"), "doc_id": row.get("doc_id"),
            "category_id": row.get("category_id"), "tier": row.get("tier"),
            "question_register": row.get("question_register"),
            "error_type_proxy": row.get("error_type_proxy"),
            "doc_length_bin": row.get("doc_length_bin"),
            "alignment_status": row.get("alignment_status"),
            "f1": row.get("f1"), "exact_match": row.get("exact_match"),
            "dcr_tau0.5": row.get("dcr_by_tau_0.5"),
            "reviewer_1_label": None, "reviewer_2_label": None,
            "adjudicated_label": None, "notes": None,
        })
    return out


def run_full_analysis(cfg: Config, dataset_total_rows: int) -> dict[str, Any]:
    analysis_dir = cfg.results_dir("analysis")

    outputs: dict[str, Any] = {}
    for cc in ("full_contract", "oracle_local"):
        lb = build_leaderboard(cfg, cc, dataset_total_rows)
        write_json(analysis_dir / f"leaderboard_{cc}.json", lb)
        if lb:
            pd.json_normalize(lb).to_csv(analysis_dir / f"leaderboard_{cc}.csv", index=False)
        outputs[f"leaderboard_{cc}"] = lb

    df = load_all_scores(cfg)

    reg = register_robustness(df, cfg)
    write_json(analysis_dir / "register_robustness.json", reg)
    outputs["register_robustness"] = reg

    cat = _breakdown_by(df, cfg, "category_id")
    write_json(analysis_dir / "per_category.json", cat)
    outputs["per_category"] = cat

    tier = _breakdown_by(df, cfg, "tier")
    write_json(analysis_dir / "per_tier.json", tier)
    outputs["per_tier"] = tier

    dlb = doc_length_bins(df, cfg)
    write_json(analysis_dir / "doc_length_bins.json", dlb)
    outputs["doc_length_bins"] = dlb

    slb = span_length_bins(df, cfg)
    write_json(analysis_dir / "span_length_bins.json", slb)
    outputs["span_length_bins"] = slb

    dtb = distractor_type_breakdown(df, cfg)
    write_json(analysis_dir / "distractor_type_breakdown.json", dtb)
    outputs["distractor_type_breakdown"] = dtb

    # Pairwise: Indian litigation-oriented models vs the three named
    # baselines, per spec §12.
    india_models = ["aalap_fp16", "inlegalllama_fp16", "inlegalbert"]
    baselines = ["qwen3_8b_fp16", "saullm_7b_fp16", "cuad_legalbert_or_roberta"]
    family = [(a, b) for a in india_models for b in baselines]
    pairwise = pairwise_comparisons(cfg, "full_contract", family)
    write_json(analysis_dir / "pairwise_comparisons.json", pairwise)
    outputs["pairwise_comparisons"] = pairwise

    from benchmark.config import append_jsonl
    error_sample_path = analysis_dir / "manual_error_sample.jsonl"
    if error_sample_path.is_file():
        error_sample_path.unlink()
    error_sample = generate_error_sample(cfg)
    for rec in error_sample:
        append_jsonl(error_sample_path, rec)
    outputs["manual_error_sample_n"] = len(error_sample)

    write_json(analysis_dir / "category_partitions_status.json", {
        "status": cfg.category_partitions_raw.get("status"),
        "note": "CUAD-aligned/adapted/India-specific breakdown skipped: category_partitions.yaml is unmapped "
                "pending manual review (no cuad_label in the release; see README.md and configs/category_partitions.yaml).",
    })

    return outputs
