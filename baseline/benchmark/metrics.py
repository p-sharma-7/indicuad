"""Deterministic span-matching / scoring metrics (spec §10-§11).

Tokenization: a single, reproducible regex word tokenizer, used
consistently everywhere token-level metrics are computed (never a model
tokenizer — that would make macro/micro comparisons across models and
architectures incoherent). Documented explicitly per spec §11.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from benchmark.normalize import comparison_form

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Reproducible regex word tokenizer: runs of letters (any script) or
    runs of digits, on the NFKC + lowercase comparison form of the text.
    Not a model tokenizer; used only for token-level P/R/F1/Jaccard."""
    return _TOKEN_RE.findall(comparison_form(text))


@dataclass
class SpanScore:
    exact_match: bool
    precision: float
    recall: float
    f1: float
    f2: float
    jaccard: float


def _prf(pred_tokens: list[str], gold_tokens: list[str], beta: float = 1.0) -> tuple[float, float, float]:
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts: dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = sum(min(c, gold_counts.get(t, 0)) for t, c in pred_counts.items())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        f_beta = 0.0
    else:
        b2 = beta * beta
        f_beta = (1 + b2) * precision * recall / (b2 * precision + recall)
    return precision, recall, f_beta


def _f_from_pr(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def jaccard(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    pred_set, gold_set = set(pred_tokens), set(gold_tokens)
    if not pred_set and not gold_set:
        return 1.0
    union = pred_set | gold_set
    if not union:
        return 0.0
    return len(pred_set & gold_set) / len(union)


def exact_match(pred_text: str, gold_text: str) -> bool:
    return comparison_form(pred_text) == comparison_form(gold_text)


def score_span(pred_text: str, gold_text: str, f2_beta: float = 2.0) -> SpanScore:
    pred_tokens = tokenize(pred_text)
    gold_tokens = tokenize(gold_text)
    precision, recall, f1 = _prf(pred_tokens, gold_tokens, beta=1.0)
    f2 = _f_from_pr(precision, recall, beta=f2_beta)
    jac = jaccard(pred_tokens, gold_tokens)
    return SpanScore(
        exact_match=exact_match(pred_text, gold_text),
        precision=precision, recall=recall, f1=f1, f2=f2, jaccard=jac,
    )


def score_best_of(pred_texts: list[str], gold_text: str, f2_beta: float = 2.0) -> SpanScore:
    """Deterministic union/best-match policy for multi-span predictions
    (spec §10.8): score each candidate span against gold, keep the
    best-F1 candidate. Ties broken by higher jaccard, then first-seen."""
    if not pred_texts:
        return score_span("", gold_text, f2_beta=f2_beta)
    scored = [(score_span(p, gold_text, f2_beta=f2_beta), i) for i, p in enumerate(pred_texts)]
    scored.sort(key=lambda t: (t[0].f1, t[0].jaccard, -t[1]), reverse=True)
    return scored[0][0]


def score_against_multi_gold(pred_texts: list[str], gold_texts: list[str], f2_beta: float = 2.0) -> SpanScore:
    """When a row has multiple gold answers (399/1521 rows in the release —
    e.g. "Parties" naming two separate party spans), score the prediction
    against the best-matching gold answer. This mirrors score_best_of's
    logic but on the gold side: a prediction that verbatim-copies any one
    of the row's gold spans should not be penalized for not covering the
    others (each is independently a valid, complete answer to the same
    question in this release's schema)."""
    if not gold_texts:
        return score_span(" ".join(pred_texts) if pred_texts else "", "", f2_beta=f2_beta)
    best = None
    for gold in gold_texts:
        s = score_best_of(pred_texts, gold, f2_beta=f2_beta)
        if best is None or (s.f1, s.jaccard) > (best.f1, best.jaccard):
            best = s
    return best


def macro_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def micro_f1_from_counts(total_overlap: int, total_pred_len: int, total_gold_len: int) -> float:
    precision = total_overlap / total_pred_len if total_pred_len else 0.0
    recall = total_overlap / total_gold_len if total_gold_len else 0.0
    return _f_from_pr(precision, recall, beta=1.0)
