"""Model registry helpers (spec §7). configs/models.yaml is the source of
truth; this module adds derived status logic on top of it."""

from __future__ import annotations

import os
from typing import Any

from benchmark.config import Config

RUNNABLE_LOCAL_STATUSES = frozenset({"FEASIBLE_FP16", "FEASIBLE_4BIT", "ATTEMPT_WITH_PREFLIGHT"})
NOT_RUN_STATUSES = frozenset({"CHECKPOINT_UNRESOLVED", "GATED_CREDENTIALS_REQUIRED", "FALLBACK_REQUIRED"})


def gated_credentials_present(entry: dict[str, Any]) -> tuple[bool, str]:
    if not entry.get("gated"):
        return True, "not gated"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return False, "gated model, no HF_TOKEN/HUGGING_FACE_HUB_TOKEN set"
    return True, "HF token present (terms acceptance still verified at load time)"


def feasible_local_models(cfg: Config) -> list[dict[str, Any]]:
    """Models eligible for `--all-feasible` on current hardware: local
    causal_lm/extractive_qa entries whose v100_feasibility is runnable and
    whose hf_id actually resolved (excludes CHECKPOINT_UNRESOLVED like
    inlegalllama_fp16 and inlegalbert)."""
    out = []
    for m in cfg.models:
        if m.get("model_type") == "api":
            continue
        if m.get("enabled_by_default") is False:
            continue
        if m.get("v100_feasibility") in RUNNABLE_LOCAL_STATUSES and m.get("hf_id"):
            out.append(m)
    return out


def registry_status_row(entry: dict[str, Any]) -> dict[str, Any]:
    """A row for the model-manifest table in the report (spec §13 table 2)."""
    return {
        "model_key": entry["model_key"],
        "display_name": entry.get("display_name"),
        "hf_id": entry.get("hf_id"),
        "revision": entry.get("revision"),
        "family": entry.get("family"),
        "role": entry.get("role"),
        "dtype": entry.get("default_dtype"),
        "quantization": entry.get("quantization"),
        "v100_feasibility": entry.get("v100_feasibility"),
        "context_budget_tokens": entry.get("context_budget_tokens"),
    }
