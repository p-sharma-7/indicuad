"""Argparse-based CLI: `python -m benchmark.cli {preflight,validate,smoke,run,score,analyze,report}`.

Each scripts/*.py entrypoint is a thin wrapper around one function here, so
logic lives in one testable place. `run` is the only subcommand that
touches GPUs/torch; everything else is dataset/CPU-only.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Any

from benchmark.config import Config, load_config, utc_now, write_json


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def cmd_preflight(cfg: Config, args: argparse.Namespace) -> int:
    from benchmark.gpu_preflight import run_preflight

    shard_gpus = [int(x) for x in args.shard_gpus.split(",")] if args.shard_gpus else None
    report = run_preflight(cfg, mode=args.parallelism, requested_gpu=args.gpu, shard_gpus=shard_gpus)

    out_path = cfg.results_dir("manifests", f"preflight_{int(time.time())}.json")
    write_json(out_path, report.to_dict())
    latest_path = cfg.results_dir("manifests", "preflight_latest.json")
    write_json(latest_path, report.to_dict())

    print(f"IndiCUAD V2 preflight — mode={args.parallelism} gpu={args.gpu}")
    for idx, state in sorted(report.all_states.items()):
        elig = report.eligibility.get(idx)
        marker = "ELIGIBLE" if (elig and elig.eligible) else "excluded"
        print(f"  GPU {idx} ({state.name}): free={state.free_gib:.1f}GiB util={state.utilization_pct:.0f}% -> {marker}")
        if elig:
            for r in elig.reasons:
                print(f"      {r}")
    print(f"decision: {report.decision}")
    print(f"wrote {out_path}")

    status = report.decision.get("status")
    if status in ("WAITING_FOR_GPU", "WAITING_FOR_MULTIGPU_RESOURCES"):
        print(status)
    return 0


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def cmd_validate(cfg: Config, args: argparse.Namespace) -> int:
    from benchmark.validation import run_validation, write_reports

    result = run_validation(cfg)
    json_path, md_path = write_reports(cfg, result)
    print(f"dataset validation: {result.overall_status} ({result.n_fail} FAIL, {result.n_warn} WARN)")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 1 if result.overall_status == "FAIL" else 0


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def cmd_smoke(cfg: Config, args: argparse.Namespace) -> int:
    from benchmark.dataset import load_dataset
    from benchmark.models.mock_runner import MockRunner, MOCK_MODEL_KEY
    from benchmark.models.base import RunRecord, write_record, raw_output_path, load_completed_keys
    from benchmark.prompts import load_full_contract_templates, build_prompt
    from benchmark.serialization import serialize_full_contract
    from benchmark import scoring as scoring_mod
    from benchmark.config import read_jsonl

    dataset = load_dataset(cfg)
    templates = load_full_contract_templates(cfg)
    n_items = min(args.limit or 12, len(dataset.qa_rows))
    rows = dataset.qa_rows[:n_items]

    mock_entry = {"model_key": MOCK_MODEL_KEY, "display_name": "Smoke-test mock runner"}
    runner = MockRunner(model_entry=mock_entry)
    runner.load()

    raw_path = raw_output_path(cfg, MOCK_MODEL_KEY, "full_contract")
    if args.overwrite and raw_path.is_file():
        raw_path.unlink()
    completed = load_completed_keys(raw_path)

    settings = ["direct_extraction", "structured_extraction"]
    n_written = 0
    for row in rows:
        doc = serialize_full_contract(dataset, row["doc_id"], cfg)
        gold_text = row["answers"][0]["text"] if row.get("answers") else ""
        for setting in settings:
            if (row["qa_id"], setting) in completed:
                continue
            prompt = build_prompt(templates, setting, row["question"], doc.text)
            pred = runner.predict_mock(row["qa_id"], row["question"], gold_text, setting)
            record = RunRecord(
                qa_id=row["qa_id"], model_key=MOCK_MODEL_KEY, context_condition="full_contract", setting=setting,
                status="ok", raw_output=pred["raw_output"], latency_seconds=pred["latency_seconds"],
                input_tokens=pred["input_tokens"], output_tokens=pred["output_tokens"],
                serialized_document_sha256=doc.sha256, prompt_template_sha256=prompt.prompt_template_sha256,
                model_input_sha256=prompt.model_input_sha256,
            )
            write_record(cfg, MOCK_MODEL_KEY, "full_contract", record)
            n_written += 1
    print(f"smoke: wrote {n_written} mock prediction records ({raw_path})")

    n_scored, n_skipped = scoring_mod.score_model_context(cfg, dataset, MOCK_MODEL_KEY, "full_contract", overwrite=args.overwrite)
    print(f"smoke: scored {n_scored} records ({n_skipped} already scored)")

    score_path = cfg.results_dir("scores", MOCK_MODEL_KEY, "full_contract.jsonl")
    scored = read_jsonl(score_path)
    n_ok = sum(1 for r in scored if r.get("status") == "ok")
    mean_f1 = sum(r.get("f1", 0.0) for r in scored if r.get("status") == "ok") / max(1, n_ok)
    print(f"smoke: {n_ok}/{len(scored)} scored ok, mean F1={mean_f1:.3f} "
          f"(mock runner targets ~{runner.correct_fraction_numerator}/{runner.correct_fraction_denominator} exact-match rate by construction)")

    summary_path = cfg.results_dir("reports", "smoke_test_summary.json")
    write_json(summary_path, {
        "generated_at": utc_now(), "n_items_requested": n_items, "n_predictions_written": n_written,
        "n_scored": len(scored), "n_scored_ok": n_ok, "mean_f1": mean_f1,
        "note": "Synthetic mock outputs only — never a scientific result. See benchmark/models/mock_runner.py.",
    })
    print(f"wrote {summary_path}")
    print("SMOKE_TEST_PASS" if n_ok > 0 else "SMOKE_TEST_FAIL")
    return 0 if n_ok > 0 else 1


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _resolve_models_to_run(cfg: Config, args: argparse.Namespace) -> list[dict[str, Any]]:
    from benchmark.models.registry import feasible_local_models

    if args.all_feasible:
        return feasible_local_models(cfg)
    if not args.models:
        raise SystemExit("--models <key1,key2,...> or --all-feasible is required for `run`")
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    out = []
    for k in keys:
        entry = cfg.model_by_key(k)
        if entry is None:
            raise SystemExit(f"unknown model_key: {k}")
        out.append(entry)
    return out


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    """Dispatches one subprocess (scripts/run_model.py) per (model,
    context_condition) pair — never imports torch in this long-lived
    process. Process isolation means one model's CUDA state, crash, or OOM
    can never leak into the next; "one model at a time" is enforced by
    running them strictly sequentially here, not concurrently."""
    import subprocess
    from benchmark.config import REPO_ROOT

    models = _resolve_models_to_run(cfg, args)
    context_conditions = ["full_contract", "oracle_local"] if args.context == "both" else [args.context]

    overall_status = 0
    for entry in models:
        for cc in context_conditions:
            print(f"=== run: model={entry['model_key']} context={cc} ===", flush=True)
            cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "run_model.py"),
                "--model", entry["model_key"], "--context", cc,
                "--parallelism", args.parallelism, "--gpu", args.gpu,
                "--settings", args.settings, "--cpu-offload", args.cpu_offload,
            ]
            if args.shard_gpus:
                cmd += ["--shard-gpus", args.shard_gpus]
            if args.max_memory:
                cmd += ["--max-memory", args.max_memory]
            if args.allow_disk_offload:
                cmd += ["--allow-disk-offload"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.overwrite:
                cmd += ["--overwrite"]
            if args.trust_remote_code:
                cmd += ["--trust-remote-code"]

            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if proc.returncode != 0:
                print(f"    run_model.py exited {proc.returncode} for {entry['model_key']}/{cc}")
                overall_status = 1
    return overall_status


# --------------------------------------------------------------------------
# score / analyze / report
# --------------------------------------------------------------------------

def cmd_score(cfg: Config, args: argparse.Namespace) -> int:
    from benchmark.dataset import load_dataset
    from benchmark import scoring as scoring_mod

    dataset = load_dataset(cfg)
    raw_root = cfg.results_dir("raw")
    if not raw_root.is_dir():
        print("no raw predictions found — run `run` first")
        return 0
    total_scored, total_skipped = 0, 0
    for model_dir in sorted(raw_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "_smoke_mock":
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            cc = jsonl_path.stem
            n_scored, n_skipped = scoring_mod.score_model_context(cfg, dataset, model_dir.name, cc, overwrite=args.overwrite)
            total_scored += n_scored
            total_skipped += n_skipped
            print(f"  scored {model_dir.name}/{cc}: {n_scored} new, {n_skipped} already scored")
    print(f"score: {total_scored} newly scored, {total_skipped} skipped (already scored)")
    return 0


def cmd_analyze(cfg: Config, args: argparse.Namespace) -> int:
    from benchmark.dataset import load_dataset
    from benchmark.analysis import run_full_analysis

    dataset = load_dataset(cfg)
    outputs = run_full_analysis(cfg, len(dataset.qa_rows))
    print(f"analyze: wrote tables to {cfg.results_dir('analysis')}")
    for k, v in outputs.items():
        n = len(v) if isinstance(v, list) else ("n/a" if isinstance(v, dict) else v)
        print(f"  {k}: {n}")
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    from benchmark.reporting import render_report, generate_figures

    fig_status = generate_figures(cfg)
    md = render_report(cfg)
    out_path = cfg.results_dir("reports", "benchmark_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(md)
    print(f"figures: {fig_status}")
    print(f"wrote {out_path}")
    return 0


# --------------------------------------------------------------------------
# argparse plumbing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="indicuad-v2")
    sub = p.add_subparsers(dest="command", required=True)

    def add_gpu_args(sp):
        sp.add_argument("--parallelism", choices=["single", "auto_shard", "manual_shard"], default="single")
        sp.add_argument("--gpu", default="auto")
        sp.add_argument("--shard-gpus", default=None)
        sp.add_argument("--max-memory", default=None)
        sp.add_argument("--cpu-offload", choices=["none", "auto"], default="none")
        sp.add_argument("--allow-disk-offload", action="store_true")

    sp = sub.add_parser("preflight")
    add_gpu_args(sp)

    sp = sub.add_parser("validate")

    sp = sub.add_parser("smoke")
    add_gpu_args(sp)
    sp.add_argument("--limit", type=int, default=12)
    sp.add_argument("--overwrite", action="store_true")

    sp = sub.add_parser("run")
    add_gpu_args(sp)
    sp.add_argument("--models", default=None)
    sp.add_argument("--all-feasible", action="store_true")
    sp.add_argument("--context", choices=["full_contract", "oracle_local", "both"], default="full_contract")
    sp.add_argument("--settings", default="direct_extraction,structured_extraction")
    sp.add_argument("--limit", type=int, default=None, help="limit to first N qa rows (pilot runs)")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--trust-remote-code", action="store_true")

    sp = sub.add_parser("score")
    sp.add_argument("--overwrite", action="store_true")

    sp = sub.add_parser("analyze")

    sp = sub.add_parser("report")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config()

    dispatch = {
        "preflight": cmd_preflight, "validate": cmd_validate, "smoke": cmd_smoke,
        "run": cmd_run, "score": cmd_score, "analyze": cmd_analyze, "report": cmd_report,
    }
    return dispatch[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
