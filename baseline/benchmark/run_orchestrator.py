"""Per-(model, context_condition) run loop (spec §8/§9).

Called from scripts/run_model.py, which has already ensured
CUDA_VISIBLE_DEVICES/LD_LIBRARY_PATH are correct (for local models) before
this module — and therefore torch — is imported. Handles exactly one model
at a time, one document at a time, resumable via the raw JSONL's existing
(qa_id, setting) keys.
"""

from __future__ import annotations

import argparse
from typing import Any

from benchmark.config import Config, base_run_manifest, utc_now, write_json
from benchmark.models.base import RunRecord, load_completed_keys, raw_output_path, write_record


def _context_common_cap(cfg: Config) -> int:
    return cfg.context["common_context_cap_tokens"]


def _shard_suffix(args: argparse.Namespace) -> str | None:
    count = getattr(args, "row_shard_count", None)
    return f"shard{args.row_shard_index}of{count}" if count else None


def _shard_rows(rows: list, args: argparse.Namespace) -> list:
    """Data-parallel row split for concurrent multi-GPU runs of one model
    (spec-external — added to fit the full roster in a fixed wall-clock
    budget on decode-bound models). Interleaved (i % count) rather than
    contiguous blocks so every shard sees a representative mix of document
    lengths/categories rather than e.g. all the longest documents."""
    count = getattr(args, "row_shard_count", None)
    if not count:
        return rows
    idx = args.row_shard_index
    return [r for i, r in enumerate(rows) if i % count == idx]


def _write_run_manifest(cfg: Config, model_key: str, context_condition: str, physical_gpus: list[int] | None,
                         extra: dict[str, Any]) -> None:
    """Merges the required provenance fields (spec §15/§N: git commit,
    environment/package versions, GPU, dtype, params, timestamps) with the
    run-specific details every caller already collects."""
    manifest = base_run_manifest({
        "model_key": model_key, "context_condition": context_condition,
        "physical_gpus": physical_gpus,
    })
    manifest.update(extra)
    write_json(cfg.results_dir("manifests", f"{model_key}__{context_condition}.json"), manifest)


def run_causal_model(cfg: Config, dataset, entry: dict[str, Any], context_condition: str, args: argparse.Namespace,
                      device_map, max_memory, physical_gpus: list[int] | None = None) -> str:
    from benchmark.models.hf_causal import HFCausalRunner, ModelLoadError, GenerationError, split_thinking_output
    from benchmark.prompts import load_full_contract_templates, load_oracle_local_templates, build_prompt
    from benchmark.serialization import serialize_full_contract, serialize_oracle_local

    model_key = entry["model_key"]
    runner = HFCausalRunner(
        model_entry=entry, device_map=device_map, max_memory=max_memory,
        trust_remote_code_override=args.trust_remote_code,
    )
    try:
        runner.load()
    except ModelLoadError as exc:
        write_json(cfg.results_dir("manifests", f"{model_key}__{context_condition}__load_failure.json"), {
            "model_key": model_key, "context_condition": context_condition,
            "failure_type": exc.failure_type, "failure_message": exc.message, "timestamp": utc_now(),
        })
        print(f"MODEL_LOAD_FAILED model={model_key} type={exc.failure_type}: {exc.message}")
        return "FAILED"

    templates = load_full_contract_templates(cfg) if context_condition == "full_contract" else load_oracle_local_templates(cfg)
    raw_path = raw_output_path(cfg, model_key, context_condition, shard_suffix=_shard_suffix(args))
    if args.overwrite and raw_path.is_file():
        raw_path.unlink()
    completed = load_completed_keys(raw_path)

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    max_new_tokens = entry.get("max_new_tokens", 512)
    common_cap = _context_common_cap(cfg)
    # Per spec §12 ("test all models on all queries" per operator direction,
    # 2026-08-18): the shared common_context_cap_tokens (8192) was excluding
    # ~60% of rows even for models registered with a real 32768-token
    # window (Qwen3 family, SaulLM, Granite, Phi-4) — an artifact of a
    # single shared floor, not those models' actual capacity. The run
    # itself now uses each model's own context_budget_tokens (minus
    # generation + safety headroom) as the real attempt/exclude threshold,
    # so every model is tried against as much of the release as it can
    # genuinely handle. `exceeds_common_context_cap` is still recorded on
    # every successful row so the stricter common-context aggregate spec
    # §6 describes remains reconstructable later by filtering on it — nothing
    # about the original cross-model-fairness analysis is lost, it just no
    # longer gates whether an attempt happens.
    model_context_budget = entry.get("context_budget_tokens") or common_cap
    usable_input_budget = max(1, model_context_budget - max_new_tokens - 32)
    enable_thinking = bool(entry.get("enable_thinking", False))

    rows = _shard_rows(dataset.qa_rows[: args.limit] if args.limit else dataset.qa_rows, args)
    n_ok, n_failed, n_context_excluded, n_skipped = 0, 0, 0, 0

    for row in rows:
        try:
            doc = serialize_full_contract(dataset, row["doc_id"], cfg) if context_condition == "full_contract" \
                else serialize_oracle_local(dataset, row, cfg)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            write_record(cfg, model_key, context_condition, RunRecord(
                qa_id=row["qa_id"], model_key=model_key, context_condition=context_condition, setting=settings[0],
                status="failed", failure_type="model_load_error", failure_message=f"document serialization failed: {exc}",
            ))
            continue

        for setting in settings:
            if (row["qa_id"], setting) in completed:
                n_skipped += 1
                continue

            prompt = build_prompt(templates, setting, row["question"], doc.text)

            n_tokens = None
            exceeds_common_cap = False
            if context_condition == "full_contract":
                try:
                    n_tokens = int(runner.tokenizer(prompt.prompt_text, return_tensors="pt")["input_ids"].shape[-1])
                except Exception:  # noqa: BLE001
                    n_tokens = len(prompt.prompt_text.split())
                exceeds_common_cap = n_tokens > common_cap
                if n_tokens > usable_input_budget:
                    n_context_excluded += 1
                    write_record(cfg, model_key, context_condition, RunRecord(
                        qa_id=row["qa_id"], model_key=model_key, context_condition=context_condition, setting=setting,
                        status="context_exceeds_model_capacity",
                        failure_message=f"{n_tokens} tokens > this model's usable input budget {usable_input_budget} "
                                        f"(context_budget_tokens={model_context_budget} - max_new_tokens={max_new_tokens} - 32 safety margin)",
                        serialized_document_sha256=doc.sha256, prompt_template_sha256=prompt.prompt_template_sha256,
                        model_input_sha256=prompt.model_input_sha256, model_revision=entry.get("revision"),
                        dtype=entry.get("default_dtype"), quantization=entry.get("quantization"),
                        extra={"exceeds_common_context_cap": True, "input_tokens_measured": n_tokens},
                    ))
                    continue

            try:
                pred = runner.predict(prompt.prompt_text, max_new_tokens, enable_thinking=enable_thinking)
            except GenerationError as exc:
                n_failed += 1
                write_record(cfg, model_key, context_condition, RunRecord(
                    qa_id=row["qa_id"], model_key=model_key, context_condition=context_condition, setting=setting,
                    status="failed", failure_type=exc.failure_type, failure_message=exc.message,
                    serialized_document_sha256=doc.sha256, prompt_template_sha256=prompt.prompt_template_sha256,
                    model_input_sha256=prompt.model_input_sha256,
                ))
                continue

            raw_output = pred["raw_output"]
            if enable_thinking:
                answer_only, thinking_text = split_thinking_output(raw_output)
                if thinking_text is None:
                    write_record(cfg, model_key, context_condition, RunRecord(
                        qa_id=row["qa_id"], model_key=model_key, context_condition=context_condition, setting=setting,
                        status="failed", failure_type="thinking_parse_failure",
                        failure_message="enable_thinking=true but no </think> tag found in output; refusing to score an "
                                        "unparseable reasoning trace as a direct answer (spec §8).",
                    ))
                    n_failed += 1
                    continue
                raw_output = answer_only  # thinking_text is discarded here — never written to any record

            write_record(cfg, model_key, context_condition, RunRecord(
                qa_id=row["qa_id"], model_key=model_key, context_condition=context_condition, setting=setting,
                status="ok", raw_output=raw_output,
                latency_seconds=pred["latency_seconds"], input_tokens=pred["input_tokens"],
                output_tokens=pred["output_tokens"], peak_gpu_memory_mib=pred.get("peak_gpu_memory_mib"),
                serialized_document_sha256=doc.sha256, prompt_template_sha256=prompt.prompt_template_sha256,
                model_input_sha256=prompt.model_input_sha256, model_revision=entry.get("revision"),
                dtype=entry.get("default_dtype"), quantization=entry.get("quantization"),
                extra={"exceeds_common_context_cap": exceeds_common_cap},
            ))
            n_ok += 1

    runner.unload()
    _write_run_manifest(cfg, model_key, context_condition, physical_gpus, {
        "hf_id": entry.get("hf_id"), "revision": entry.get("revision"),
        "dtype": entry.get("default_dtype"), "quantization": entry.get("quantization"),
        "hf_device_map": runner.hf_device_map, "load_time_seconds": runner.load_time_seconds,
        "n_ok": n_ok, "n_failed": n_failed, "n_context_excluded": n_context_excluded, "n_skipped_resumed": n_skipped,
        "n_dataset_rows": len(rows), "prompt_template_path": templates.path.name,
        "prompt_template_version": templates.version,
    })
    print(f"{model_key}/{context_condition}: ok={n_ok} failed={n_failed} context_excluded={n_context_excluded} resumed_skip={n_skipped}")
    return "COMPLETE" if n_failed == 0 else "PARTIAL"


def run_extractqa_model(cfg: Config, dataset, entry: dict[str, Any], context_condition: str, args: argparse.Namespace,
                         physical_gpus: list[int] | None = None) -> str:
    from benchmark.models.hf_causal import ModelLoadError
    from benchmark.serialization import serialize_full_contract, serialize_oracle_local

    model_key = entry["model_key"]
    is_embedding_ranker = entry.get("adapter_type") == "embedding_span_ranking"

    if is_embedding_ranker:
        from benchmark.models.embedding_ranker import EmbeddingSpanRanker
        runner = EmbeddingSpanRanker(model_entry=entry)
    else:
        from benchmark.models.hf_extractqa import ExtractQARunner
        runner = ExtractQARunner(model_entry=entry)

    try:
        runner.load()
    except ModelLoadError as exc:
        write_json(cfg.results_dir("manifests", f"{model_key}__{context_condition}__load_failure.json"), {
            "model_key": model_key, "context_condition": context_condition,
            "failure_type": exc.failure_type, "failure_message": exc.message, "timestamp": utc_now(),
        })
        print(f"MODEL_LOAD_FAILED model={model_key} type={exc.failure_type}: {exc.message}")
        return "FAILED"

    raw_path = raw_output_path(cfg, model_key, context_condition, shard_suffix=_shard_suffix(args))
    if args.overwrite and raw_path.is_file():
        raw_path.unlink()
    completed = load_completed_keys(raw_path)
    rows = _shard_rows(dataset.qa_rows[: args.limit] if args.limit else dataset.qa_rows, args)
    n_ok, n_failed, n_skipped = 0, 0, 0

    for row in rows:
        if (row["qa_id"], "direct_extraction") in completed:
            n_skipped += 1
            continue
        doc = serialize_full_contract(dataset, row["doc_id"], cfg) if context_condition == "full_contract" \
            else serialize_oracle_local(dataset, row, cfg)
        try:
            if is_embedding_ranker:
                if context_condition == "full_contract":
                    result = runner.predict_span(row["question"], dataset, row["doc_id"])
                else:
                    result = runner.predict_span_over_candidates(row["question"], doc.segments)
            else:
                result = runner.predict_span(row["question"], doc.text)
        except ModelLoadError as exc:
            n_failed += 1
            write_record(cfg, model_key, context_condition, RunRecord(
                qa_id=row["qa_id"], model_key=model_key, context_condition=context_condition, setting="direct_extraction",
                status="failed", failure_type=exc.failure_type, failure_message=exc.message,
            ))
            continue

        detail = {"n_candidates": result["n_candidates"], "best_span": result["best_span"]}
        if not is_embedding_ranker:
            detail["n_windows"] = result["n_windows"]
        detail["adapter_type"] = "embedding_span_ranking" if is_embedding_ranker else "trained_extractive_qa_head"

        write_record(cfg, model_key, context_condition, RunRecord(
            qa_id=row["qa_id"], model_key=model_key, context_condition=context_condition, setting="direct_extraction",
            status="ok", raw_output=result["raw_output"],
            latency_seconds=result["latency_seconds"], input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"], peak_gpu_memory_mib=result.get("peak_gpu_memory_mib"),
            serialized_document_sha256=doc.sha256, model_revision=entry.get("revision"),
            extra={"extractive_qa_detail": detail},
        ))
        n_ok += 1

    runner.unload()
    _write_run_manifest(cfg, model_key, context_condition, physical_gpus, {
        "hf_id": entry.get("hf_id"), "revision": entry.get("revision"), "load_time_seconds": runner.load_time_seconds,
        "n_ok": n_ok, "n_failed": n_failed, "n_skipped_resumed": n_skipped, "n_dataset_rows": len(rows),
        "adapter_type": "embedding_span_ranking" if is_embedding_ranker else "trained_extractive_qa_head",
        "architecture_note": (
            "zero-shot paragraph-level embedding similarity ranking, no trained QA head — whole-paragraph spans, "
            "not minimal token spans; not directly comparable to a trained extractive-QA model. See spec §9."
            if is_embedding_ranker else
            "windowed extractive-QA with a trained QA head — not directly comparable to full-contract generative "
            "prompting; see spec §9."
        ),
    })
    print(f"{model_key}/{context_condition}: ok={n_ok} failed={n_failed} resumed_skip={n_skipped}")
    return "COMPLETE" if n_failed == 0 else "PARTIAL"
