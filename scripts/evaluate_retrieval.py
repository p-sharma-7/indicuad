"""Stage 12/13 — retrieval sanity check and abstention probe.

This is a consistency check on the dataset, not a model bake-off. It asks
one question: when a query has a gold paragraph, can the retriever find
it? A low score here usually means the corpus, the questions, or the
segmentation disagree with each other — not that the retriever needs
tuning.

Positive queries      Recall@1/5/10 and MRR, broken down by register,
                      category, and tier.
Negative queries      false-retrieval probe. There is no gold paragraph,
                      so we measure how confidently the retriever returns
                      something anyway — the behaviour that turns into
                      hallucination downstream.

Nothing here ever edits an annotation. If recall is poor the script says
so and stops; the spec is explicit that the response is investigation,
not quietly moving gold spans to wherever the retriever looked.
"""

from __future__ import annotations

import statistics
import sys

from indicuad_common import QA_DIR, Stage, emit_report, read_jsonl
from retrieval import get_index

K_VALUES = [1, 5, 10]
MAX_K = max(K_VALUES)

# Below this, the dataset and the retrieval pipeline disagree badly enough
# that something is structurally wrong.
SANITY_THRESHOLD_RECALL_AT_10 = 0.60


def rank_of_gold(hits: list[dict], gold: set[str]) -> int | None:
    for hit in hits:
        if hit["paragraph_id"] in gold:
            return hit["rank"]
    return None


def summarise(ranks: list[int | None]) -> dict:
    total = len(ranks)
    if total == 0:
        return {"queries": 0}
    found = [r for r in ranks if r is not None]
    result = {"queries": total}
    for k in K_VALUES:
        result[f"recall@{k}"] = round(sum(1 for r in found if r <= k) / total, 4)
    result["mrr"] = round(sum(1 / r for r in found) / total, 4)
    result["median_rank_when_found"] = statistics.median(found) if found else None
    return result


def main() -> int:
    stage = Stage("12/13", "Retrieval sanity check and abstention probe")

    qa = read_jsonl(QA_DIR / "qa.jsonl")
    if not qa:
        stage.fail("qa/qa.jsonl not built — run build_qa.py first.")
        return 1

    index = get_index()
    stage.info(f"BM25 index: {index.n} paragraphs")

    positives = [r for r in qa if not r["is_impossible"] and r["gold_paragraph_ids"]]
    negatives = [r for r in qa if r["is_impossible"]]
    stage.info(f"positive queries: {len(positives)}   negative queries: {len(negatives)}")

    overall: list[int | None] = []
    by_register: dict[str, list] = {}
    by_category: dict[str, list] = {}
    by_tier: dict[str, list] = {}
    misses = []

    for row in positives:
        hits = index.search(row["question"], k=MAX_K)
        rank = rank_of_gold(hits, set(row["gold_paragraph_ids"]))
        overall.append(rank)
        by_register.setdefault(row["question_register"], []).append(rank)
        by_category.setdefault(row["category_id"], []).append(rank)
        by_tier.setdefault(row.get("tier") or "unknown", []).append(rank)
        if rank is None:
            misses.append(
                {
                    "qa_id": row["qa_id"],
                    "doc_id": row["doc_id"],
                    "category_id": row["category_id"],
                    "register": row["question_register"],
                    "gold_paragraph_ids": row["gold_paragraph_ids"],
                    "top_hits": [h["paragraph_id"] for h in hits[:3]],
                }
            )

    # Abstention probe: a negative query has no right answer, but the
    # retriever will still return its best guess. Track how strong that
    # guess looks, and whether it even lands in the right document.
    negative_scores = []
    same_document_hits = 0
    for row in negatives:
        hits = index.search(row["question"], k=MAX_K)
        if not hits:
            negative_scores.append(0.0)
            continue
        negative_scores.append(hits[0]["score"])
        if hits[0]["doc_id"] == row["doc_id"]:
            same_document_hits += 1

    positive_summary = summarise(overall)
    negative_summary = {
        "queries": len(negatives),
        "mean_top1_score": round(statistics.mean(negative_scores), 4) if negative_scores else 0.0,
        "median_top1_score": round(statistics.median(negative_scores), 4) if negative_scores else 0.0,
        "top1_in_same_document": same_document_hits,
        "top1_in_same_document_rate": (
            round(same_document_hits / len(negatives), 4) if negatives else 0.0
        ),
        "note": (
            "There is no correct paragraph for these queries. A high score with a "
            "confident in-document hit is the false-retrieval pattern that leads a "
            "downstream model to answer instead of abstaining."
        ),
    }

    recall_at_10 = positive_summary.get("recall@10", 0.0)
    failed = len(positives) > 0 and recall_at_10 < SANITY_THRESHOLD_RECALL_AT_10
    status = "RETRIEVAL_SANITY_CHECK_FAILED" if failed else "PASS"

    for k in K_VALUES:
        stage.info(f"  Recall@{k}: {positive_summary.get(f'recall@{k}')}")
    stage.info(f"  MRR:       {positive_summary.get('mrr')}")

    if failed:
        stage.fail(
            f"Recall@10 {recall_at_10} is below {SANITY_THRESHOLD_RECALL_AT_10}. "
            "Investigate the corpus, questions, or segmentation. Do NOT adjust "
            "annotations to match retriever behaviour."
        )

    emit_report(
        "retrieval_report.json",
        {
            "status": status,
            "sanity_threshold_recall_at_10": SANITY_THRESHOLD_RECALL_AT_10,
            "retriever": "BM25 (scripts/retrieval.py)",
            "positive": positive_summary,
            "by_register": {k: summarise(v) for k, v in sorted(by_register.items())},
            "by_tier": {k: summarise(v) for k, v in sorted(by_tier.items())},
            "by_category": {k: summarise(v) for k, v in sorted(by_category.items())},
            "negative_abstention_probe": negative_summary,
            "misses": misses[:100],
            "miss_count": len(misses),
        },
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
