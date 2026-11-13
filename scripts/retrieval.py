"""BM25 retrieval over the frozen paragraph set.

Implemented in pure Python rather than pulling a library, so the ranking
is fully specified by this file and reproduces exactly from the frozen
corpus with no dependency-version drift.

Index metadata is restricted to paragraph_id / doc_id / paragraph_index.
Per spec §17, gold-bearing fields (category_id, gold_paragraph_ids,
is_impossible) must never enter retrieval metadata — indexing them would
leak the evaluation signal into the retriever.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter

from indicuad_common import CORPUS_DIR, RETRIEVAL_DIR, read_jsonl

K1 = 1.5
B = 0.75

TOKEN = re.compile(r"[a-z0-9]+")

# Deliberately minimal: no stemming, no stopword list tuned to this corpus.
# Both would need to be reported as part of the retrieval configuration and
# neither is needed for a sanity check.
def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1]


class BM25Index:
    def __init__(self, records: list[dict]) -> None:
        # Only non-gold metadata is retained.
        self.meta = [
            {
                "paragraph_id": r["paragraph_id"],
                "doc_id": r["doc_id"],
                "paragraph_index": r["paragraph_index"],
            }
            for r in records
        ]
        self.docs = [tokenize(r["text"]) for r in records]
        self.lengths = [len(d) for d in self.docs]
        self.avg_length = sum(self.lengths) / max(len(self.docs), 1)
        self.n = len(self.docs)

        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(self.docs):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, []).append((index, freq))

        self.idf = {
            term: math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def search(self, query: str, k: int = 10) -> list[dict]:
        scores: dict[int, float] = {}
        for term in tokenize(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for index, freq in plist:
                norm = 1 - B + B * self.lengths[index] / max(self.avg_length, 1e-9)
                scores[index] = scores.get(index, 0.0) + idf * (
                    freq * (K1 + 1) / (freq + K1 * norm)
                )

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            {**self.meta[index], "rank": rank, "score": round(score, 4)}
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path) -> "BM25Index":
        with path.open("rb") as handle:
            return pickle.load(handle)


BM25_PATH = RETRIEVAL_DIR / "bm25" / "index.pkl"


def indexable(paragraphs: list[dict]) -> list[dict]:
    """Paragraphs eligible for retrieval.

    Running furniture (letterhead, page headers, standing disclaimers,
    signature blocks) is excluded. It is not part of any instrument, so it
    can never be a correct answer, but it is near-identical across
    documents and therefore competes for top-k slots on generic queries —
    displacing real clauses and depressing recall for reasons that have
    nothing to do with the retriever.

    Superseded paragraphs are excluded for the same reason: where an
    agreement was split out of a larger document, the parent still holds a
    byte-identical copy of the child's gold paragraphs.
    """
    return [
        p for p in paragraphs
        if not p.get("is_boilerplate") and not p.get("is_superseded")
    ]


def load_paragraphs() -> list[dict]:
    return indexable(read_jsonl(CORPUS_DIR / "paragraphs.jsonl"))


def get_index() -> BM25Index:
    if BM25_PATH.is_file():
        return BM25Index.load(BM25_PATH)
    index = BM25Index(load_paragraphs())
    index.save(BM25_PATH)
    return index
