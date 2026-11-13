"""Shared runner infrastructure: structured failures, resumable JSONL I/O.

Every runner (hf_causal, hf_extractqa, embedding_ranker) writes one JSONL record per
(qa_id, model_key, context_condition, setting) to
results/indicuad_v2/raw/{model_key}/{context_condition}.jsonl, resuming by
skipping keys already present unless --overwrite is passed. Raw output is
always preserved; a record is never silently overwritten in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.config import append_jsonl, read_jsonl, utc_now

# Structured failure taxonomy (spec §0A/§8): every failure a runner can hit
# maps to one of these, so results/indicuad_v2/raw/*.jsonl failures are
# machine-triageable rather than free-text.
FAILURE_TYPES = frozenset({
    "cuda_oom",
    "cuda_fragmentation",
    "gpu_unavailable",
    "device_map_error",
    "bitsandbytes_incompatibility",
    "cpu_ram_offload_failure",
    "disk_offload_failure",
    "checkpoint_unresolved",
    "gated_credentials_missing",
    "trust_remote_code_required",
    "context_exceeds_model_capacity",
    "model_load_error",
    "generation_error",
    "thinking_parse_failure",
    "dependency_missing",
    "other",
})


@dataclass
class RunRecord:
    """One (qa_id, model_key, context_condition, setting) result. Either a
    successful prediction or a structured failure — never both empty."""
    qa_id: str
    model_key: str
    context_condition: str   # "full_contract" | "oracle_local"
    setting: str             # "direct_extraction" | "structured_extraction"
    status: str              # "ok" | "failed" | "context_exceeds_model_capacity" | "skipped_not_run"
    raw_output: str | None = None
    normalized: dict[str, Any] | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_gpu_memory_mib: float | None = None
    serialized_document_sha256: str | None = None
    prompt_template_sha256: str | None = None
    model_input_sha256: str | None = None
    model_revision: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    physical_gpu: list[int] | None = None
    timestamp: str = field(default_factory=utc_now)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def raw_output_path(cfg, model_key: str, context_condition: str, shard_suffix: str | None = None) -> Path:
    # shard_suffix (e.g. "shard0of3"): row-sharded runs write to a separate
    # file per shard so concurrent processes never share a writer — merged
    # into the canonical {context_condition}.jsonl afterward by
    # scripts/merge_shards.py, which scoring/analysis alone ever reads.
    name = f"{context_condition}.{shard_suffix}.jsonl" if shard_suffix else f"{context_condition}.jsonl"
    return cfg.results_dir("raw", model_key, name)


def load_completed_keys(path: Path) -> set[tuple[str, str]]:
    """(qa_id, setting) pairs already present with status != 'failed' —
    failed records are retried on the next run unless the caller explicitly
    wants to preserve them (OOM/failure retries are not "indefinite": the
    runner still only tries once per invocation, but a fresh invocation may
    retry a previously-failed item, e.g. after freeing GPU memory)."""
    completed = set()
    for rec in read_jsonl(path):
        if rec.get("status") == "ok":
            completed.add((rec["qa_id"], rec["setting"]))
    return completed


def write_record(cfg, model_key: str, context_condition: str, record: RunRecord) -> None:
    append_jsonl(raw_output_path(cfg, model_key, context_condition), record.to_dict())


class BaseRunner:
    """Interface every model runner implements. Not an ABC to keep the
    smoke/mock runner trivially substitutable in tests."""

    model_entry: dict[str, Any]

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, prompt_text: str, max_new_tokens: int) -> dict[str, Any]:
        """Returns at least {"raw_output": str, "input_tokens": int,
        "output_tokens": int, "latency_seconds": float}."""
        raise NotImplementedError

    def unload(self) -> None:
        pass
