"""Dataset loading + validation invariants (spec §5, deliverable P)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from benchmark.config import load_config
from benchmark.dataset import load_dataset


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def dataset(cfg):
    return load_dataset(cfg)


def test_release_cardinalities(cfg, dataset):
    exp = cfg.expected
    assert len(dataset.qa_rows) == exp["query_rows"]
    assert len({r["qa_id"] for r in dataset.qa_rows}) == exp["query_rows"]
    assert len(dataset.intent_groups) == exp["intent_groups"]
    assert len(dataset.documents) == exp["core_documents"]
    # "651 gold spans" is a per-intent-group count: all 3 registers of a
    # group carry identical answers[] content, so summing over all 1521
    # rows would triple-count to 1953 (verified against qa.json directly
    # during implementation). One representative row per group instead.
    first_row_per_group = {}
    for r in dataset.qa_rows:
        first_row_per_group.setdefault(r["intent_group_id"], r)
    assert sum(len(r["answers"]) for r in first_row_per_group.values()) == exp["gold_spans"]
    assert sum(len(r["answers"]) for r in dataset.qa_rows) == exp["gold_spans"] * 3


def test_every_row_is_core_and_answerable(dataset):
    for row in dataset.qa_rows:
        assert row["split"] == "core"
        assert row["is_impossible"] is False
        assert len(row["answers"]) >= 1


def test_registers_partition_evenly(dataset):
    for group_id, rows in dataset.intent_groups.items():
        assert len(rows) == 3, group_id
        registers = {r["question_register"] for r in rows}
        assert registers == {"canonical", "semi_formal", "naive_layperson"}, group_id


def test_answer_offsets_slice_frozen_text_exactly(cfg, dataset):
    checked = 0
    for row in dataset.qa_rows:
        text = dataset.normalized_text(row["doc_id"])
        for ans in row["answers"]:
            sliced = text[ans["answer_start"]: ans["answer_end"]]
            assert sliced == ans["text"], (row["qa_id"], ans)
            checked += 1
    # Every row (all 3 registers per group) is checked independently, so
    # this is 651 gold spans x 3 registers, not 651 — see
    # test_release_cardinalities for the per-group figure.
    assert checked == cfg.expected["gold_spans"] * 3


def test_gold_paragraph_ids_resolve(dataset):
    for row in dataset.qa_rows:
        for pid in row["gold_paragraph_ids"]:
            assert pid in dataset.paragraphs, (row["qa_id"], pid)


def test_distractor_join_exact(dataset):
    """Every gold answer joins exactly one distractor record with 3 slots
    (mirrors the check in benchmark/validation.py that found 1953/1953
    joins clean at implementation time)."""
    n_checked = 0
    for row in dataset.qa_rows:
        for i, ans in enumerate(row["answers"]):
            recs = [r for r in dataset.distractor_by_qa_id.get(row["qa_id"], []) if r.get("answer_index") == i]
            assert len(recs) == 1, (row["qa_id"], i, len(recs))
            assert len(recs[0]["distractors"]) == 3
            assert recs[0]["gold_text"] == ans["text"]
            n_checked += 1
    assert n_checked > 0


def test_distractors_never_equal_gold_after_normalization(dataset):
    from benchmark.validation import _normalize_for_compare
    for row in dataset.qa_rows:
        for i, ans in enumerate(row["answers"]):
            for rec in dataset.distractor_by_qa_id.get(row["qa_id"], []):
                if rec.get("answer_index") != i:
                    continue
                gold_norm = _normalize_for_compare(ans["text"])
                for slot in rec["distractors"]:
                    assert _normalize_for_compare(slot["text"]) != gold_norm


def test_normalized_text_absent_raises_clear_error(cfg, dataset):
    with pytest.raises(FileNotFoundError):
        dataset.normalized_text("DOC-DOES-NOT-EXIST")


def test_never_touch_files_unmodified_by_import(cfg):
    """Loading the dataset must never write to the released files."""
    import hashlib
    from benchmark.validation import NEVER_TOUCH_FILES

    def h(rel):
        p = cfg.dataset.root / rel
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

    before = {rel: h(rel) for rel in NEVER_TOUCH_FILES}
    load_dataset(cfg)
    after = {rel: h(rel) for rel in NEVER_TOUCH_FILES}
    assert before == after
