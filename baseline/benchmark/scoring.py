"""Scores raw prediction JSONL against gold + distractors (spec §10-§11).

Input:   results/indicuad_v2/raw/{model_key}/{context_condition}.jsonl
         (written by scripts/run_model.py / benchmark.models.*)
Outputs:
  results/indicuad_v2/normalized/{model_key}/{context_condition}.jsonl
    the auditable intermediate normalization trail (§10): unwrapped spans,
    alignment_status, aligned character offsets, parse notes — kept
    separate from final scores so the *reasoning* behind a score (why a
    span was/wasn't matched) is inspectable without re-running anything.
  results/indicuad_v2/scores/{model_key}/{context_condition}.jsonl
    final per-item metric values.

One record per raw record in each output. Never mixes smoke-mock output
(model_key == "_smoke_mock") into a scoring pass over real models unless
explicitly requested (mirrors the raw-side separation).
"""

from __future__ import annotations

from typing import Any

from benchmark.config import Config, append_jsonl, read_jsonl, utc_now
from benchmark.distractors import compute_distractor_score, resolve_distractor_record, resolve_distractor_texts
from benchmark.metrics import score_against_multi_gold, score_best_of
from benchmark.normalize import normalize_output


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _base_record(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": raw["qa_id"],
        "model_key": raw["model_key"],
        "context_condition": raw["context_condition"],
        "setting": raw["setting"],
        "status": raw["status"],
        "scored_at": utc_now(),
    }


def score_one_record(raw: dict[str, Any], dataset, cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (scored_record, normalized_record)."""
    qa_id = raw["qa_id"]
    row = dataset.qa_by_id(qa_id)
    scoring_cfg = cfg.scoring
    tau_values = scoring_cfg["distractor_tau_values"]
    dsr_margin = scoring_cfg["dsr_margin"]
    f2_beta = scoring_cfg["f2_beta"]

    scored: dict[str, Any] = _base_record(raw)
    normalized_record: dict[str, Any] = _base_record(raw)

    if row is None:
        scored["status"] = "failed"
        scored["failure_type"] = "unknown_qa_id"
        return scored, normalized_record

    scored.update({
        "doc_id": row["doc_id"],
        "category_id": row["category_id"],
        "intent_group_id": row["intent_group_id"],
        "question_register": row["question_register"],
        "tier": row.get("tier"),
        "doc_origin": row.get("doc_origin"),
    })

    doc_text = dataset.normalized_text(row["doc_id"])
    scored["doc_char_count"] = len(doc_text)
    scored["doc_estimated_tokens"] = _estimate_tokens(doc_text)

    gold_answers = row.get("answers", [])
    gold_texts = [a.get("text", "") for a in gold_answers]
    scored["gold_span_char_lengths"] = [len(t) for t in gold_texts]
    scored["n_gold_answers"] = len(gold_answers)

    if raw["status"] != "ok":
        scored["failure_type"] = raw.get("failure_type")
        scored["failure_message"] = raw.get("failure_message")
        normalized_record["failure_type"] = raw.get("failure_type")
        return scored, normalized_record

    normalized = normalize_output(raw.get("raw_output", ""), raw["setting"], doc_text)
    scored["alignment_status"] = normalized.alignment_status
    normalized_record.update(normalized.to_dict())

    pred_texts = [s["text"] for s in normalized.aligned_spans] or normalized.unwrapped_spans
    span_score = score_against_multi_gold(pred_texts, gold_texts, f2_beta=f2_beta)

    scored.update({
        "exact_match": span_score.exact_match,
        "precision": span_score.precision,
        "recall": span_score.recall,
        "f1": span_score.f1,
        "f2": span_score.f2,
        "gold_jaccard_score": span_score.jaccard,
    })

    # Determine which gold answer_index this prediction best matches, for
    # distractor lookup (a row may carry >1 gold answer, e.g. multi-party
    # "Parties" rows; each answer has its own distractor record).
    best_idx, best_f1 = 0, -1.0
    for i, gt in enumerate(gold_texts):
        s = score_best_of(pred_texts, gt, f2_beta=f2_beta)
        if s.f1 > best_f1:
            best_f1, best_idx = s.f1, i
    scored["best_answer_index"] = best_idx

    distractor_texts = resolve_distractor_texts(dataset, qa_id, best_idx)
    if distractor_texts is not None and gold_texts:
        pred_joined = " ".join(pred_texts)
        d_score = compute_distractor_score(pred_joined, gold_texts[best_idx], distractor_texts, tau_values, dsr_margin)
        scored.update({
            "gold_jaccard": d_score.gold_jaccard,
            "best_distractor_jaccard": d_score.best_distractor_jaccard,
            "distractor_margin": d_score.margin,
            "dcr_by_tau": {str(t): v for t, v in d_score.dcr_by_tau.items()},
            "dsr_by_tau": {str(t): v for t, v in d_score.dsr_by_tau.items()},
        })
        # Type/source of the closest-matching distractor slot (spec §11:
        # "distractor type, if available" / "same-document vs cross-document
        # distractor source, if available") — attributed regardless of
        # whether this item was actually DCR-positive, so the breakdown
        # covers every scored row, not only confused ones.
        rec = resolve_distractor_record(dataset, qa_id, best_idx)
        if rec is not None and d_score.best_distractor_slot_index is not None:
            slot = rec["distractors"][d_score.best_distractor_slot_index]
            scored["closest_distractor_source"] = slot.get("source")  # "mined" | "perturbed"
            if slot.get("source") == "mined":
                scored["closest_distractor_same_document"] = slot.get("from_doc_id") == row["doc_id"]
    else:
        scored["distractor_join_missing"] = True

    for k in ("latency_seconds", "peak_gpu_memory_mib", "input_tokens", "output_tokens"):
        scored[k] = raw.get(k)

    return scored, normalized_record


def score_model_context(cfg: Config, dataset, model_key: str, context_condition: str, overwrite: bool = False) -> tuple[int, int]:
    raw_path = cfg.results_dir("raw", model_key, f"{context_condition}.jsonl")
    if not raw_path.is_file():
        return 0, 0
    score_path = cfg.results_dir("scores", model_key, f"{context_condition}.jsonl")
    normalized_path = cfg.results_dir("normalized", model_key, f"{context_condition}.jsonl")

    already_scored: set[tuple[str, str]] = set()
    if score_path.is_file() and not overwrite:
        for rec in read_jsonl(score_path):
            already_scored.add((rec["qa_id"], rec["setting"]))
    elif overwrite and score_path.is_file():
        score_path.unlink()
        if normalized_path.is_file():
            normalized_path.unlink()

    n_scored, n_skipped = 0, 0
    for raw in read_jsonl(raw_path):
        key = (raw["qa_id"], raw["setting"])
        if key in already_scored:
            n_skipped += 1
            continue
        scored, normalized_record = score_one_record(raw, dataset, cfg)
        append_jsonl(score_path, scored)
        append_jsonl(normalized_path, normalized_record)
        n_scored += 1

    return n_scored, n_skipped
