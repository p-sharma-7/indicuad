"""Resumability invariants for raw JSONL prediction writing (spec §8,
deliverable D/P): a completed (qa_id, setting) record is skipped on rerun
unless --overwrite, and no record is ever silently overwritten in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from benchmark.config import load_config, read_jsonl
from benchmark.models.base import RunRecord, load_completed_keys, raw_output_path, write_record


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    # Pathlib joins an absolute right-hand operand by replacing the left
    # side entirely, so this redirects results_dir() into a throwaway tmp
    # directory without touching the real results/indicuad_v2/.
    cfg.raw["paths"]["results_dir"] = str(tmp_path / "results_indicuad_v2_test")
    return cfg


def test_completed_keys_start_empty(cfg):
    path = raw_output_path(cfg, "fake_model", "full_contract")
    assert load_completed_keys(path) == set()


def test_write_then_resume_skips_completed(cfg):
    record = RunRecord(qa_id="IND-Q-00001", model_key="fake_model", context_condition="full_contract",
                        setting="direct_extraction", status="ok", raw_output="Governing Law of India")
    write_record(cfg, "fake_model", "full_contract", record)

    path = raw_output_path(cfg, "fake_model", "full_contract")
    completed = load_completed_keys(path)
    assert ("IND-Q-00001", "direct_extraction") in completed
    assert ("IND-Q-00001", "structured_extraction") not in completed


def test_failed_records_are_not_treated_as_completed(cfg):
    """A failed record must be retried on the next invocation, not
    permanently skipped — otherwise a transient OOM would silently and
    permanently exclude an item from the benchmark."""
    record = RunRecord(qa_id="IND-Q-00002", model_key="fake_model", context_condition="full_contract",
                        setting="direct_extraction", status="failed", failure_type="cuda_oom",
                        failure_message="mock OOM")
    write_record(cfg, "fake_model", "full_contract", record)
    path = raw_output_path(cfg, "fake_model", "full_contract")
    assert ("IND-Q-00002", "direct_extraction") not in load_completed_keys(path)


def test_rerun_appends_not_overwrites(cfg):
    """Simulates two separate invocations: the first writes a record, a
    'resumed' second invocation only appends genuinely new records — the
    first record must survive untouched in the file."""
    r1 = RunRecord(qa_id="IND-Q-00001", model_key="fake_model", context_condition="full_contract",
                    setting="direct_extraction", status="ok", raw_output="first run output")
    write_record(cfg, "fake_model", "full_contract", r1)

    path = raw_output_path(cfg, "fake_model", "full_contract")
    completed = load_completed_keys(path)
    assert ("IND-Q-00001", "direct_extraction") in completed

    # A resumed run would see this key as completed and skip it; simulate a
    # *different* qa_id being processed instead, as the real runner would.
    r2 = RunRecord(qa_id="IND-Q-00003", model_key="fake_model", context_condition="full_contract",
                    setting="direct_extraction", status="ok", raw_output="second run output")
    write_record(cfg, "fake_model", "full_contract", r2)

    all_records = read_jsonl(path)
    assert len(all_records) == 2
    assert all_records[0]["raw_output"] == "first run output"
    assert all_records[1]["raw_output"] == "second run output"


def test_overwrite_flow_starts_clean(cfg):
    r1 = RunRecord(qa_id="IND-Q-00001", model_key="fake_model", context_condition="full_contract",
                    setting="direct_extraction", status="ok", raw_output="stale output")
    write_record(cfg, "fake_model", "full_contract", r1)
    path = raw_output_path(cfg, "fake_model", "full_contract")
    assert path.is_file()

    # This mirrors what scripts/run_model.py / cmd_smoke do when --overwrite is passed.
    path.unlink()
    assert load_completed_keys(path) == set()
