"""Loading and joining the frozen IndiCUAD release.

Strictly read-only with respect to the dataset (the repository root).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from benchmark.config import Config, read_json, read_jsonl


@dataclass
class Dataset:
    cfg: Config
    qa_rows: list[dict[str, Any]]
    documents: dict[str, dict[str, Any]]          # doc_id -> record, core split only
    paragraphs: dict[str, dict[str, Any]]          # paragraph_id -> record, core split only
    paragraphs_by_doc: dict[str, list[dict[str, Any]]]  # doc_id -> paragraphs ordered by paragraph_index
    categories: dict[str, dict[str, Any]]          # category_id -> record
    distractors: list[dict[str, Any]]              # raw distractor records
    distractor_by_qa_id: dict[str, list[dict[str, Any]]]  # qa_id -> [distractor record, ...] (one per answer_index present)

    def qa_by_id(self, qa_id: str) -> dict[str, Any] | None:
        return self._qa_index.get(qa_id)

    @functools.cached_property
    def _qa_index(self) -> dict[str, dict[str, Any]]:
        return {row["qa_id"]: row for row in self.qa_rows}

    @functools.cached_property
    def intent_groups(self) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in self.qa_rows:
            groups.setdefault(row["intent_group_id"], []).append(row)
        return groups

    def normalized_text(self, doc_id: str) -> str:
        """Per-instance cache via a plain dict — Dataset is a mutable
        dataclass (unhashable), so functools.lru_cache on this bound method
        would raise `TypeError: unhashable type` on the implicit `self`
        argument."""
        cache = self.__dict__.setdefault("_normalized_text_cache", {})
        if doc_id not in cache:
            path = self.cfg.dataset.normalized_dir / f"{doc_id}.txt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"work/normalized/{doc_id}.txt is absent from the dataset checkout "
                    f"({self.cfg.dataset.normalized_dir})."
                )
            cache[doc_id] = path.read_text(encoding="utf-8")
        return cache[doc_id]

    def gold_paragraphs(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.paragraphs[pid] for pid in row.get("gold_paragraph_ids", []) if pid in self.paragraphs]

    def distractor_texts_for(self, qa_id: str, answer_index: int) -> list[dict[str, Any]] | None:
        for rec in self.distractor_by_qa_id.get(qa_id, []):
            if rec.get("answer_index") == answer_index:
                return rec.get("distractors")
        return None


def _load_qa_rows(cfg: Config) -> list[dict[str, Any]]:
    return read_json(cfg.dataset.qa_release)


def _load_documents(cfg: Config) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(cfg.dataset.documents_jsonl)
    return {r["doc_id"]: r for r in rows if r.get("split") == "core"}


def _load_paragraphs(cfg: Config, core_doc_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = read_jsonl(cfg.dataset.paragraphs_jsonl)
    by_id: dict[str, dict[str, Any]] = {}
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["doc_id"] not in core_doc_ids:
            continue
        by_id[r["paragraph_id"]] = r
        by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda p: p["paragraph_index"])
    return by_id, by_doc


def _load_categories(cfg: Config) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(cfg.dataset.categories_jsonl)
    return {r["category_id"]: r for r in rows}


def _load_distractors(cfg: Config) -> list[dict[str, Any]]:
    return read_json(cfg.dataset.distractors)


def _index_distractors_by_qa(distractor_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for rec in distractor_records:
        for qa_id in rec.get("qa_ids", []):
            index.setdefault(qa_id, []).append(rec)
    return index


def load_dataset(cfg: Config, include_unanswerable: bool = False) -> Dataset:
    """Load the full V2 dataset. `include_unanswerable` defaults False per
    spec §1: dropped no-answer rows must not enter scoring unless an
    explicit future flag enables them; this parameter *is* that flag, and
    still defaults closed."""
    qa_rows = _load_qa_rows(cfg)
    if include_unanswerable:
        qa_rows = qa_rows + read_json(cfg.dataset.qa_unanswerable)

    documents = _load_documents(cfg)
    core_doc_ids = set(documents.keys())
    paragraphs, paragraphs_by_doc = _load_paragraphs(cfg, core_doc_ids)
    categories = _load_categories(cfg)
    distractors = _load_distractors(cfg)
    distractor_by_qa_id = _index_distractors_by_qa(distractors)

    return Dataset(
        cfg=cfg,
        qa_rows=qa_rows,
        documents=documents,
        paragraphs=paragraphs,
        paragraphs_by_doc=paragraphs_by_doc,
        categories=categories,
        distractors=distractors,
        distractor_by_qa_id=distractor_by_qa_id,
    )
