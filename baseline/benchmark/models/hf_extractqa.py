"""Extractive-QA encoder runner with sliding-window preprocessing (spec §9).

Never forced into chat generation. Standard SQuAD-style sliding window over
the full frozen document text, answerability-free (V2 is answerable-only —
no null-span/CLS-index handling is implemented since it would never be
exercised and its presence would misleadingly imply abstention is
supported, which spec §2 explicitly rules out for the primary task).

Deterministic minimal-span policy (documented, spec §9): among all
candidate (start, end) token spans across all windows with end >= start and
span length <= max_answer_len_tokens, keep the single highest-scoring
(start_logit + end_logit) span; ties broken by shorter token length, then
by lower window index, then by lower start token index. This is applied
uniformly regardless of checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from benchmark.models.base import BaseRunner
from benchmark.models.hf_causal import ModelLoadError, _is_oom


@dataclass
class ExtractQARunner(BaseRunner):
    model_entry: dict[str, Any]

    def __post_init__(self):
        self.model = None
        self.tokenizer = None
        self.load_time_seconds: float | None = None

    def load(self) -> None:
        entry = self.model_entry
        hf_id = entry.get("hf_id")
        revision = entry.get("revision")
        if not hf_id:
            raise ModelLoadError("checkpoint_unresolved", f"{entry['model_key']}: no hf_id registered")
        trust_remote_code = bool(entry.get("trust_remote_code", False))
        t0 = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision, trust_remote_code=trust_remote_code)
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                hf_id, revision=revision, trust_remote_code=trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise ModelLoadError("cuda_oom", f"{hf_id}: OOM during load: {exc}") from exc
            raise ModelLoadError("model_load_error", f"{hf_id}@{revision}: {exc}") from exc

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.load_time_seconds = time.time() - t0

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_span(self, question: str, document_text: str) -> dict[str, Any]:
        """Runs the full sliding-window pipeline and returns the single
        best span with its character offsets into `document_text`, plus the
        raw candidate list for auditing (spec §9: 'save predicted character
        offsets, source text, confidence/logit score, window metadata, and
        raw candidate span details')."""
        entry = self.model_entry
        max_seq_length = entry.get("max_seq_length", 384)
        doc_stride = entry.get("doc_stride", 128)
        n_best = entry.get("n_best", 20)
        max_answer_len_tokens = entry.get("max_answer_len_tokens", 64)

        device = next(self.model.parameters()).device
        t0 = time.time()
        try:
            encoded = self.tokenizer(
                question, document_text,
                max_length=max_seq_length, truncation="only_second", stride=doc_stride,
                return_overflowing_tokens=True, return_offsets_mapping=True,
                padding="max_length", return_tensors="pt",
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError("model_load_error", f"tokenization failed: {exc}") from exc

        n_windows = encoded["input_ids"].shape[0]
        offset_mappings = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)

        candidates: list[dict[str, Any]] = []
        try:
            with torch.inference_mode():
                for w in range(n_windows):
                    window_inputs = {k: v[w : w + 1].to(device) for k, v in encoded.items() if k in
                                      ("input_ids", "attention_mask", "token_type_ids")}
                    outputs = self.model(**window_inputs)
                    start_logits = outputs.start_logits[0].detach().cpu().numpy()
                    end_logits = outputs.end_logits[0].detach().cpu().numpy()
                    sequence_ids = encoded.sequence_ids(w)
                    offsets = offset_mappings[w].tolist()

                    context_token_idx = [i for i, sid in enumerate(sequence_ids) if sid == 1]
                    if not context_token_idx:
                        continue
                    top_start = np.argsort(start_logits)[-n_best:][::-1]
                    top_end = np.argsort(end_logits)[-n_best:][::-1]

                    for s in top_start:
                        if sequence_ids[s] != 1:
                            continue
                        for e in top_end:
                            if sequence_ids[e] != 1:
                                continue
                            if e < s or (e - s + 1) > max_answer_len_tokens:
                                continue
                            char_start, char_end = offsets[s][0], offsets[e][1]
                            if char_start is None or char_end is None or char_end <= char_start:
                                continue
                            candidates.append({
                                "window_index": w, "start_token": int(s), "end_token": int(e),
                                "char_start": int(char_start), "char_end": int(char_end),
                                "text": document_text[char_start:char_end],
                                "score": float(start_logits[s] + end_logits[e]),
                            })
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ModelLoadError("cuda_oom", f"OOM during extractive-QA inference: {exc}") from exc
            raise ModelLoadError("generation_error", str(exc)) from exc

        latency = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not candidates:
            return {
                "raw_output": "", "input_tokens": int(encoded["input_ids"].numel()),
                "output_tokens": 0, "latency_seconds": latency, "peak_gpu_memory_mib": None,
                "best_span": None, "n_windows": n_windows, "n_candidates": 0, "candidates": [],
            }

        candidates.sort(key=lambda c: (-c["score"], c["end_token"] - c["start_token"], c["window_index"], c["start_token"]))
        best = candidates[0]

        return {
            "raw_output": best["text"],
            "input_tokens": int(encoded["input_ids"].numel()),
            "output_tokens": best["end_token"] - best["start_token"] + 1,
            "latency_seconds": latency,
            "peak_gpu_memory_mib": (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if torch.cuda.is_available() else None,
            "best_span": {"char_start": best["char_start"], "char_end": best["char_end"], "text": best["text"], "score": best["score"]},
            "n_windows": n_windows,
            "n_candidates": len(candidates),
            "candidates": sorted(candidates, key=lambda c: -c["score"])[:n_best],
        }
