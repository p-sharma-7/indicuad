"""Stage 11 — build the BM25 index over corpus/paragraphs.jsonl.

The index is built from paragraphs, never from qa.jsonl. Tier D documents
are included: the protocol keeps them in the retrieval index as hard
negatives even though they are never annotated.

A smoke query is run at the end — an index that returns nothing for an
obvious query is broken, and that should surface here rather than during
the retrieval sanity check.

Outputs:
    retrieval/bm25/index.pkl
    retrieval/retrieval_config.json
"""

from __future__ import annotations

import sys

from indicuad_common import (
    CORPUS_DIR,
    RETRIEVAL_DIR,
    Stage,
    read_jsonl,
    write_json,
)
from retrieval import B, BM25_PATH, K1, BM25Index, indexable

SMOKE_QUERIES = [
    "arbitration seat and governing law",
    "employee salary and provident fund",
    "bank guarantee amount and validity",
    "liquidated damages for delay",
]


def main() -> int:
    stage = Stage("11", "BM25 index construction")

    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    if not paragraphs:
        stage.fail("corpus/paragraphs.jsonl missing — run segment_paragraphs.py first.")
        return 1

    eligible = indexable(paragraphs)
    excluded = len(paragraphs) - len(eligible)

    index = BM25Index(eligible)
    index.save(BM25_PATH)

    stage.info(
        f"indexed {index.n} paragraphs from {len({p['doc_id'] for p in eligible})} documents "
        f"({excluded} boilerplate paragraphs excluded)"
    )
    stage.info(f"vocabulary: {len(index.postings):,} terms   avg length: {index.avg_length:.1f} tokens")

    leaked = {"category_id", "gold_paragraph_ids", "is_impossible", "text"}
    present = leaked & set(index.meta[0].keys())
    if present:
        stage.fail(f"gold-bearing fields present in retrieval metadata: {sorted(present)}")
        return 1
    stage.info("index metadata carries no gold-bearing fields")

    empty = 0
    for query in SMOKE_QUERIES:
        hits = index.search(query, k=3)
        if not hits:
            empty += 1
            stage.warn(f"smoke query returned nothing: {query!r}")
            continue
        top = hits[0]
        stage.info(f"  {query!r} -> {top['paragraph_id']} (score {top['score']})")

    if empty:
        stage.fail(f"{empty} smoke queries returned no results")
        return 1

    write_json(
        RETRIEVAL_DIR / "retrieval_config.json",
        {
            "bm25": {
                "implementation": "scripts/retrieval.py (Okapi BM25, pure Python)",
                "k1": K1,
                "b": B,
                "tokenizer": "lowercase [a-z0-9]+, tokens of length >= 2",
                "stemming": None,
                "stopwords": None,
                "index_path": "retrieval/bm25/index.pkl",
            },
            "vector": {
                "status": "not built",
                "note": "run build_vector_index.py; requires sentence-transformers",
            },
            "index_unit": "paragraph",
            "built_from": "corpus/paragraphs.jsonl",
            "paragraphs_indexed": index.n,
            "paragraphs_total": len(paragraphs),
            "boilerplate_excluded": excluded,
            "boilerplate_note": (
                "Running furniture (letterhead, page headers, standing disclaimers, "
                "signature blocks) is flagged is_boilerplate in paragraphs.jsonl and "
                "held out of the index. It belongs to no instrument, so it can never be "
                "a correct answer, but it is near-identical across documents and "
                "competes for top-k slots on generic queries."
            ),
            "vocabulary_terms": len(index.postings),
            "includes_tier_d": True,
            "tier_d_note": "Tier D is indexed as hard negatives but never annotated",
            "gold_fields_in_metadata": [],
        },
    )
    stage.info("wrote retrieval/retrieval_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
