"""Scoring invariants: tokenization, EM/P/R/F1/F2/Jaccard, normalization,
alignment (spec §10-§11, deliverable P)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.metrics import exact_match, jaccard, score_span, score_best_of, tokenize
from benchmark.normalize import (
    ALIGNMENT_EXACT, ALIGNMENT_NORMALIZED_EXACT, ALIGNMENT_NOT_FOUND, ALIGNMENT_INVALID_JSON,
    comparison_form, normalize_output, unwrap_direct, unwrap_structured,
)


def test_tokenize_basic():
    assert tokenize("Payment Terms & Schedule") == ["payment", "terms", "schedule"]


def test_exact_match_case_and_whitespace_insensitive():
    assert exact_match("  Payment   Terms ", "payment terms")
    assert not exact_match("Payment Terms", "Payment Schedule")


def test_score_span_identical_is_perfect():
    s = score_span("Governing Law", "Governing Law")
    assert s.exact_match
    assert s.precision == s.recall == s.f1 == s.f2 == s.jaccard == 1.0


def test_score_span_partial_overlap():
    s = score_span("the Governing Law of India", "Governing Law")
    assert not s.exact_match
    assert s.recall == 1.0
    assert 0 < s.precision < 1.0


def test_score_span_disjoint_is_zero():
    s = score_span("Confidentiality", "Payment Terms")
    assert s.f1 == 0.0
    assert s.jaccard == 0.0


def test_jaccard_empty_vs_empty_is_one():
    assert jaccard([], []) == 1.0


def test_score_best_of_picks_best_candidate():
    s = score_best_of(["Confidentiality", "Governing Law of India"], "Governing Law")
    assert s.recall == 1.0


def test_direct_cut_at_first_reemitted_prompt_marker():
    assert unwrap_direct("Governing Law of India\nQuestion: next one")[0] == ["Governing Law of India"]
    assert unwrap_direct("  Governing Law  ")[0] == ["Governing Law"]


def test_structured_first_balanced_object_and_non_string_answer():
    assert unwrap_structured('noise {"answer": "x {y}"} {"answer": "z"}')[0] == ["x {y}"]
    assert unwrap_structured('{"answer": ["x"]}')[0] == []
    assert unwrap_structured('{"other": "x"}')[0] == []


def test_unwrap_structured_valid_json():
    spans, notes = unwrap_structured('{"answer": "Governing Law"}')
    assert spans == ["Governing Law"]


def test_unwrap_structured_invalid_json_flagged():
    spans, notes = unwrap_structured("not json at all")
    assert spans == []
    assert "invalid_json" in notes


def test_normalize_output_exact_alignment():
    doc = "This Agreement is governed by the laws of India. The Governing Law is India."
    result = normalize_output("Governing Law is India", "direct_extraction", doc)
    assert result.alignment_status == ALIGNMENT_EXACT
    assert result.aligned_spans[0]["char_start"] >= 0


def test_normalize_output_normalized_exact_alignment_on_whitespace_defect():
    """Regression for corpus-specific defects the README documents
    (run-together words / line-break artefacts) — a prediction that adds
    normal spacing around a defect must still align via the normalized
    path, never via inventing/repairing the underlying text."""
    doc = "The parties agree to a PAYMENTSCHEDULE as set out in Annex B."
    result = normalize_output("PAYMENTSCHEDULE", "direct_extraction", doc)
    assert result.alignment_status == ALIGNMENT_EXACT


def test_normalize_output_not_found_when_fabricated():
    doc = "This Agreement is governed by the laws of India."
    result = normalize_output("a clause that does not appear anywhere", "direct_extraction", doc)
    assert result.alignment_status == ALIGNMENT_NOT_FOUND


def test_normalize_output_structured_invalid_json():
    doc = "irrelevant document text"
    result = normalize_output("this is not json", "structured_extraction", doc)
    assert result.alignment_status == ALIGNMENT_INVALID_JSON


def test_comparison_form_nfkc_and_lowercase():
    assert comparison_form("  Governing   LAW ") == "governing law"
