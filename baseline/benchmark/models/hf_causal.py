"""Hugging Face causal-LM runner (spec §8).

Caller contract: `CUDA_VISIBLE_DEVICES` (and, transitively, `LD_LIBRARY_PATH`
— see benchmark/gpu_env.py) must already be correct in this process's
environment *before* this module is imported, since importing it imports
torch. scripts/run_model.py enforces this via
`benchmark.gpu_env.ensure_cuda_lib_path(cuda_visible_devices=...)` before
importing this module.

A second real driver-gap discovered running gemma4_26b_a4b_it (transformers
5.15.0 / torch 2.5.1+cu118 and 2.7.1+cu118 — reproduced on both): this
host's driver (460.91.03) has no `nvmlDeviceGetNvLinkRemoteDeviceType`
symbol in its `libnvidia-ml.so.1`. transformers' `_load_pretrained_model`
calls `caching_allocator_warmup()` — a `torch.empty(..., device=...)`
pre-allocation optimization — whenever `from_pretrained` is given a
`device_map`, and *that specific allocation path* (not plain `.cuda()`
tensor ops, which work fine on this driver) triggers the missing NVML
symbol via torch's internal `c10::cuda::DriverAPI`. `_is_nvml_driver_gap`
below detects this exact signature; `load()` retries with `device_map`
omitted (loads to CPU, then a manual `.to(f"cuda:{n}")`) for the
single-device case only — the buggy path only runs when device_map is not
None, so omitting it skips the warmup entirely. Not attempted for a
sharded/auto device_map, which this workaround cannot replicate.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.models.base import BaseRunner

CHAT_TEMPLATE_FAMILIES = {"qwen3_chat", "mistral_instruct", "gemma_chat", "llama3_chat"}

# Manual fallback templates, used only when a checkpoint's own tokenizer
# ships no chat_template at all (confirmed in production for
# opennyaiorg/Aalap-Mistral-7B-v0.1-bf16 — tokenizer.chat_template is None,
# so tokenizer.apply_chat_template() raises ValueError before this fallback
# was added). Kept minimal and standard per family rather than invented;
# never used to override a checkpoint's own template when one exists.
MANUAL_CHAT_TEMPLATES: dict[str, str] = {
    "mistral_instruct": "<s>[INST] {prompt} [/INST]",
}


class ModelLoadError(Exception):
    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message


class GenerationError(Exception):
    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message


def _is_oom(exc: Exception) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    if isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower()


def _is_nvml_driver_gap(exc: Exception) -> bool:
    msg = str(exc)
    return "nvmlDeviceGetNvLinkRemoteDeviceType" in msg or (
        "INTERNAL ASSERT FAILED" in msg and "driver_api.cpp" in msg and "nvml" in msg.lower()
    )


def _is_single_device_map(device_map: Any) -> bool:
    if isinstance(device_map, dict):
        return len(set(device_map.values())) == 1
    return False


def _single_device_map_target(device_map: dict) -> int:
    return next(iter(device_map.values()))


@dataclass
class HFCausalRunner(BaseRunner):
    model_entry: dict[str, Any]
    device_map: str | dict | None = None       # None -> single "cuda:0" (post CUDA_VISIBLE_DEVICES remap)
    max_memory: dict[str, str] | None = None     # for auto_shard/manual_shard
    trust_remote_code_override: bool = False

    def __post_init__(self):
        self.model = None
        self.tokenizer = None
        self.hf_device_map: dict[str, Any] | None = None
        self.load_time_seconds: float | None = None

    def load(self) -> None:
        entry = self.model_entry
        hf_id = entry.get("hf_id")
        revision = entry.get("revision")
        if not hf_id:
            raise ModelLoadError("checkpoint_unresolved", f"{entry['model_key']}: no hf_id registered (see configs/models.yaml notes)")

        trust_remote_code = bool(entry.get("trust_remote_code", False)) or self.trust_remote_code_override
        if entry.get("trust_remote_code") and not self.trust_remote_code_override:
            raise ModelLoadError("trust_remote_code_required",
                                  f"{entry['model_key']} is registered with trust_remote_code=true but the runner "
                                  f"was not given an explicit override; refusing to load with untrusted remote code.")

        quantization = entry.get("quantization", "none")
        quantization_config = None
        if quantization == "bnb_nf4":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                )
            except ImportError as exc:
                raise ModelLoadError("dependency_missing", f"bitsandbytes not available: {exc}") from exc

        t0 = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision, trust_remote_code=trust_remote_code)
        except Exception as exc:  # noqa: BLE001
            if "gated" in str(exc).lower() or "authenticat" in str(exc).lower() or "401" in str(exc):
                raise ModelLoadError("gated_credentials_missing", f"{hf_id}: {exc}") from exc
            raise ModelLoadError("model_load_error", f"tokenizer load failed for {hf_id}@{revision}: {exc}") from exc

        device_map = self.device_map if self.device_map is not None else {"": 0}
        load_kwargs: dict[str, Any] = dict(
            revision=revision, torch_dtype=torch.float16, device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        if self.max_memory is not None:
            load_kwargs["max_memory"] = self.max_memory

        try:
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        except ImportError as exc:
            if "bitsandbytes" in str(exc).lower():
                raise ModelLoadError("bitsandbytes_incompatibility", f"{hf_id}: {exc}") from exc
            raise ModelLoadError("dependency_missing", f"{hf_id}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            if _is_nvml_driver_gap(exc) and _is_single_device_map(device_map):
                # Host driver (460.91.03) doesn't implement the NVML symbol
                # transformers>=5's caching_allocator_warmup() calls when a
                # device_map is passed to from_pretrained (see
                # benchmark/models/hf_causal.py module docstring / configs/
                # models.yaml gemma4_26b_a4b_it notes for the full trace).
                # That warmup only runs when device_map is not None, so for
                # the simple single-GPU case, load onto CPU (device_map
                # omitted entirely) and move the whole model over manually
                # — slower (no warmed allocator, full CPU RAM copy first)
                # but avoids the broken code path completely. Not attempted
                # for a real multi-device map (auto_shard) — accelerate's
                # placement logic isn't something a single .to() replaces.
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs.pop("device_map", None)
                fallback_kwargs.pop("max_memory", None)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(hf_id, **fallback_kwargs)
                    target = _single_device_map_target(device_map)
                    self.model = self.model.to(f"cuda:{target}")
                except Exception as exc2:  # noqa: BLE001
                    if _is_oom(exc2):
                        self._free()
                        raise ModelLoadError("cuda_oom", f"{hf_id}@{revision}: OOM during CPU-load NVML-workaround fallback: {exc2}") from exc2
                    raise ModelLoadError("model_load_error", f"{hf_id}@{revision}: NVML workaround also failed: {exc2}") from exc2
            elif _is_oom(exc):
                self._free()
                raise ModelLoadError("cuda_oom", f"{hf_id}@{revision} ({quantization}): OOM during model load: {exc}") from exc
            elif "gated" in str(exc).lower() or "authenticat" in str(exc).lower() or "401" in str(exc):
                raise ModelLoadError("gated_credentials_missing", f"{hf_id}: {exc}") from exc
            elif "device_map" in str(exc).lower() or "no_split" in str(exc).lower():
                raise ModelLoadError("device_map_error", f"{hf_id}: {exc}") from exc
            else:
                raise ModelLoadError("model_load_error", f"{hf_id}@{revision}: {exc}") from exc

        self.model.eval()
        self.hf_device_map = dict(getattr(self.model, "hf_device_map", {}) or {})
        self.load_time_seconds = time.time() - t0

    def _free(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self._free()

    def predict(self, prompt_text: str, max_new_tokens: int, enable_thinking: bool = False) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise GenerationError("model_load_error", "predict() called before load()")

        template_kwargs: dict[str, Any] = {}
        if self.model_entry.get("inference_template") == "qwen3_chat":
            template_kwargs["enable_thinking"] = enable_thinking

        messages = [{"role": "user", "content": prompt_text}]
        if getattr(self.tokenizer, "chat_template", None) is None:
            family = self.model_entry.get("inference_template")
            manual = MANUAL_CHAT_TEMPLATES.get(family)
            if manual is None:
                raise GenerationError(
                    "model_load_error",
                    f"tokenizer.chat_template is not set for {self.model_entry.get('hf_id')} and no manual "
                    f"fallback is registered for inference_template={family!r} in MANUAL_CHAT_TEMPLATES "
                    f"(benchmark/models/hf_causal.py) — refusing to invent one.",
                )
            text = manual.format(prompt=prompt_text)
            templated = self.tokenizer(text, return_tensors="pt")
        else:
            try:
                templated = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", **template_kwargs,
                )
            except TypeError:
                # Tokenizer's chat template doesn't accept enable_thinking (non-Qwen3) — retry without it.
                templated = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        # transformers>=5 returns a BatchEncoding (dict-like, has
        # .input_ids) from apply_chat_template(return_tensors="pt");
        # transformers<5 (the default indicuad-bench-v2 env, 4.51.3)
        # returns a bare tensor. Support both so the same runner code works
        # across both pinned environments (see indicuad-bench-v2-gemma4).
        input_ids = templated.input_ids if hasattr(templated, "input_ids") else templated

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        # Single-sequence batch, no padding — attention_mask is all-ones,
        # but pass it explicitly rather than let generate() guess from
        # pad_token_id (which for Qwen3 equals eos_token_id, and HF warns
        # that this makes the guess ambiguous).
        attention_mask = torch.ones_like(input_ids)
        n_input_tokens = int(input_ids.shape[-1])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False, top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        t0 = time.time()
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                self._clear_cache_only()
                raise GenerationError("cuda_oom", f"OOM during generate(): {exc}") from exc
            raise GenerationError("generation_error", str(exc)) from exc
        latency = time.time() - t0

        new_tokens = output_ids[0][n_input_tokens:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        peak_mem_mib = None
        if torch.cuda.is_available():
            peak_mem_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        self._clear_cache_only()

        return {
            "raw_output": raw_text,
            "input_tokens": n_input_tokens,
            "output_tokens": int(new_tokens.shape[-1]),
            "latency_seconds": latency,
            "peak_gpu_memory_mib": peak_mem_mib,
        }

    def _clear_cache_only(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def split_thinking_output(raw_text: str) -> tuple[str, str | None]:
    """Official Qwen3 format: split on the literal </think> tag. Returns
    (final_answer_text, discarded_thinking_text_or_None). The thinking text
    is returned only so a caller can assert it is never persisted anywhere
    — callers must not write it to any output record."""
    marker = "</think>"
    if marker in raw_text:
        idx = raw_text.index(marker)
        thinking = raw_text[:idx]
        answer = raw_text[idx + len(marker):].strip()
        return answer, thinking
    return raw_text, None
