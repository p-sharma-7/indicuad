"""Keyword-sweep matching for protocol §5 P2.

The sweep is a recall aid: it points an annotator at paragraphs worth
reading. That job fails in both directions — miss a clause and the sweep
was pointless, but bury the annotator in false leads and they stop
trusting it, which is worse.

Naive substring matching does exactly that. Measured over the frozen
corpus, matching `sweep_terms` as case-insensitive substrings produced:

    'IS '  2,043 matches, 438 real    (CAT-025 — matched every "is")
    'PAN'    525 matches,   5 real    (CAT-002 — matched "company")
    'LD'     370 matches,   2 real    (CAT-020 — matched "would", "held")
    'ESI'    175 matches,   2 real    (CAT-018)
    'cap'    128 matches,   0 real    (CAT-019 — matched "CAPACITY")

Three matching modes, chosen from the shape of the term:

  Acronym   all-caps, <= 4 chars ("LD", "PAN", "CIN", "ESI", "BIS", "IEC")
            case-SENSITIVE, both boundaries. "LD" must not match "would",
            and lowercasing is what let it.
  Short     <= 5 chars ("cap") — case-insensitive, both boundaries, so
            "cap" does not match "capacity".
  Stem      everything else ("terminat", "indemnif", "liability") —
            case-insensitive, leading boundary only, so a stem still
            reaches "termination" and "terminated".

Terms are authored in categories.jsonl, which protocol §7 requires a legal
reviewer to validate. That review was declined, so these terms have never
been checked by a qualified reader — treat sweep output accordingly.
"""

from __future__ import annotations

import re

ACRONYM_MAX = 4
SHORT_MAX = 5


def term_pattern(term: str) -> re.Pattern[str] | None:
    """Compiled matcher for one sweep term, or None if the term is empty."""
    stripped = term.strip()
    if not stripped:
        return None

    if stripped.isupper() and stripped.isalpha() and len(stripped) <= ACRONYM_MAX:
        # Case-sensitive: the whole point is that "LD" is not "would".
        return re.compile(rf"\b{re.escape(stripped)}\b")

    if len(stripped) <= SHORT_MAX:
        return re.compile(rf"\b{re.escape(stripped)}\b", re.I)

    return re.compile(rf"\b{re.escape(stripped)}", re.I)


def compile_terms(terms: list[str]) -> list[re.Pattern[str]]:
    return [p for p in (term_pattern(t) for t in terms) if p is not None]


def first_match(text: str, patterns: list[re.Pattern[str]]) -> re.Match[str] | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def hits(paragraphs: list[dict], terms: list[str]) -> list[str]:
    """Paragraph IDs matching any term. Candidates to check, not labels.

    Boilerplate is skipped — running headers are not part of the
    instrument, so a hit there is never a clause.
    """
    patterns = compile_terms(terms)
    return [
        paragraph["paragraph_id"]
        for paragraph in paragraphs
        if not paragraph.get("is_boilerplate")
        and first_match(paragraph["text"], patterns) is not None
    ]
