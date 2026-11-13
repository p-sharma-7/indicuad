"""Distractor join and adversarial confusion metrics (spec §11).

The join is exact (validated in benchmark/validation.py: 1953/1953 answer↔
distractor-slot joins checked, 0 malformed at implementation time) via
(qa_id, answer_index) -> distractor record -> 3 distractor slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.metrics import jaccard, tokenize


@dataclass
class DistractorScore:
    gold_jaccard: float
    best_distractor_jaccard: float
    margin: float                       # gold_jaccard - max(distractor_jaccard)
    dcr_by_tau: dict[float, bool] = field(default_factory=dict)
    dsr_by_tau: dict[float, bool] = field(default_factory=dict)
    distractor_jaccards: list[float] = field(default_factory=list)
    best_distractor_slot_index: int | None = None  # for type/source breakdowns (spec §11)


def compute_distractor_score(
    pred_text: str,
    gold_text: str,
    distractor_texts: list[str],
    tau_values: list[float],
    dsr_margin: float,
) -> DistractorScore:
    pred_tokens = tokenize(pred_text)
    gold_jac = jaccard(pred_tokens, tokenize(gold_text))
    distractor_jacs = [jaccard(pred_tokens, tokenize(d)) for d in distractor_texts]
    best_distractor_jac = max(distractor_jacs) if distractor_jacs else 0.0
    best_distractor_slot_index = distractor_jacs.index(best_distractor_jac) if distractor_jacs else None
    margin = gold_jac - best_distractor_jac

    dcr_by_tau: dict[float, bool] = {}
    dsr_by_tau: dict[float, bool] = {}
    for tau in tau_values:
        # DCR: prediction reaches similarity >= tau with some distractor.
        dcr_by_tau[tau] = any(dj >= tau for dj in distractor_jacs)
        # DSR: stricter — best distractor overlap exceeds gold overlap by
        # >= dsr_margin AND meets tau.
        dsr_by_tau[tau] = (best_distractor_jac >= tau) and (best_distractor_jac - gold_jac >= dsr_margin)

    return DistractorScore(
        gold_jaccard=gold_jac,
        best_distractor_jaccard=best_distractor_jac,
        margin=margin,
        dcr_by_tau=dcr_by_tau,
        dsr_by_tau=dsr_by_tau,
        distractor_jaccards=distractor_jacs,
        best_distractor_slot_index=best_distractor_slot_index,
    )


def resolve_distractor_texts(dataset, qa_id: str, answer_index: int) -> list[str] | None:
    slots = dataset.distractor_texts_for(qa_id, answer_index)
    if slots is None:
        return None
    return [s.get("text", "") for s in slots]


def resolve_distractor_record(dataset, qa_id: str, answer_index: int) -> dict[str, Any] | None:
    for rec in dataset.distractor_by_qa_id.get(qa_id, []):
        if rec.get("answer_index") == answer_index:
            return rec
    return None
