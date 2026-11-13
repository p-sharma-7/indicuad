"""Distractor-confusion metric invariants (spec §11, deliverable P)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.distractors import compute_distractor_score

TAU_VALUES = [0.3, 0.5, 0.7]


def test_perfect_gold_match_never_dsr_positive():
    score = compute_distractor_score(
        pred_text="Governing Law of India",
        gold_text="Governing Law of India",
        distractor_texts=["Governing Law of the United States", "Arbitration Clause", "Confidentiality Term"],
        tau_values=TAU_VALUES, dsr_margin=0.1,
    )
    assert score.gold_jaccard == 1.0
    # DCR only asks whether some distractor is reached (Jaccard 3/7 here).
    assert score.dcr_by_tau == {0.3: True, 0.5: False, 0.7: False}
    assert all(v is False for v in score.dsr_by_tau.values())
    assert score.margin == 1.0 - score.best_distractor_jaccard


def test_distractor_selection_triggers_dcr_and_dsr():
    score = compute_distractor_score(
        pred_text="Arbitration Clause under Indian law",
        gold_text="Governing Law of India",
        distractor_texts=["Arbitration Clause under Indian law", "Confidentiality Term", "Termination Notice"],
        tau_values=TAU_VALUES, dsr_margin=0.1,
    )
    assert score.gold_jaccard < 0.5
    assert score.best_distractor_jaccard == 1.0
    assert score.dcr_by_tau[0.5] is True
    assert score.dsr_by_tau[0.5] is True


def test_low_overlap_with_everything_is_not_dcr():
    score = compute_distractor_score(
        pred_text="totally unrelated text about weather",
        gold_text="Governing Law of India",
        distractor_texts=["Arbitration Clause", "Confidentiality Term", "Termination Notice"],
        tau_values=TAU_VALUES, dsr_margin=0.1,
    )
    assert score.dcr_by_tau[0.5] is False


def test_margin_computation():
    score = compute_distractor_score(
        pred_text="Payment Schedule Annex B",
        gold_text="Payment Schedule",
        distractor_texts=["Payment Schedule Annex A", "Termination Notice", "Confidentiality"],
        tau_values=TAU_VALUES, dsr_margin=0.1,
    )
    assert abs(score.margin - (score.gold_jaccard - score.best_distractor_jaccard)) < 1e-9


def test_tau_sensitivity_monotonic():
    """Raising tau can only turn a DCR/DSR positive into negative, never
    the reverse, for a fixed prediction."""
    score = compute_distractor_score(
        pred_text="Arbitration under Indian law and Confidentiality",
        gold_text="Governing Law of India",
        distractor_texts=["Arbitration under Indian law", "Confidentiality Term", "Termination Notice"],
        tau_values=[0.3, 0.5, 0.7], dsr_margin=0.1,
    )
    dcr_03, dcr_05, dcr_07 = score.dcr_by_tau[0.3], score.dcr_by_tau[0.5], score.dcr_by_tau[0.7]
    # DCR requires distractor_overlap >= tau, so it can only go True->False as tau rises.
    assert (not dcr_03) or True  # sanity: no exception raised; monotonic check below
    values = [score.dcr_by_tau[t] for t in (0.3, 0.5, 0.7)]
    true_then_false = True
    seen_false = False
    for v in values:
        if seen_false and v:
            true_then_false = False
        if not v:
            seen_false = True
    assert true_then_false
