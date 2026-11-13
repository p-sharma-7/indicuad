"""Deterministic mock runner for the smoke test (spec §O).

Produces synthetic, clearly-labelled outputs to exercise pipeline mechanics
(prompt construction, normalization, scoring, resumability, reporting) with
no model download and no network/API call. Smoke outputs are written under
results/indicuad_v2/raw/_smoke/ — a separate namespace from any real
model's results/indicuad_v2/raw/{model_key}/ — and must never be scored
alongside real results (score_indicuad_v2.py refuses to mix them; see
scripts/score_indicuad_v2.py).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from benchmark.models.base import BaseRunner

MOCK_MODEL_KEY = "_smoke_mock"


@dataclass
class MockRunner(BaseRunner):
    model_entry: dict[str, Any]
    # Deterministic behaviour knobs, fixed for reproducibility of the smoke
    # test itself (not configurable via CLI — this is a pipeline-mechanics
    # check, not a benchmarked system).
    correct_fraction_numerator: int = 2   # answer correctly on qa_ids whose
    correct_fraction_denominator: int = 3  # hash % denom < numerator

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def predict_mock(self, qa_id: str, question: str, gold_text: str, setting: str) -> dict[str, Any]:
        t0 = time.time()
        h = int(hashlib.sha256(qa_id.encode("utf-8")).hexdigest(), 16)
        will_be_correct = (h % self.correct_fraction_denominator) < self.correct_fraction_numerator

        if will_be_correct:
            body = gold_text
        else:
            # Deterministic, clearly-wrong-but-plausible-shaped output.
            body = f"[mock non-gold span for {qa_id}]"

        if setting == "structured_extraction":
            raw = '{"answer": "%s"}' % body.replace('"', '\\"')
        else:
            raw = body

        latency = time.time() - t0
        return {
            "raw_output": raw,
            "input_tokens": len(question.split()) + 50,
            "output_tokens": len(body.split()),
            "latency_seconds": latency,
            "peak_gpu_memory_mib": None,
        }
