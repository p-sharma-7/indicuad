"""Resolve verbatim clause text to character offsets in the frozen corpus.

Annotation requires `answer_start` / `answer_end` offsets into
`work/normalized/<DOC>.txt`. Nobody can produce those by hand, so this
module takes the clause text as a human (or a model) would quote it and
finds it in the source.

Matching is whitespace-insensitive: the normalized text wraps clauses
across lines, so a quoted span almost never matches byte-for-byte even
when it is verbatim in every way that matters. Every other difference —
a changed word, a dropped negation, an em dash rewritten as a hyphen —
still fails to match, which is the point. The offsets returned always
slice the real text, so `text[start:end]` is source-exact even though the
query was not.

A span must match exactly once. Zero matches means the quote is wrong.
Several matches means the quote is ambiguous and needs `occurrence` set
explicitly, because silently taking the first one would put gold on the
wrong clause.

Used by propose_annotations.py and available to human annotators directly:

    python scripts/resolve_spans.py DOC-018 "The licensee shall deposit Rs."
"""

from __future__ import annotations

import re
import sys

from indicuad_common import CORPUS_DIR, NORMALIZED_DIR, read_jsonl

# The only category whose answer is allowed to sit in running furniture:
# a document's name is part of its masthead. See build_answer().
MASTHEAD_CATEGORY = "CAT-001"


class SpanError(ValueError):
    """A span that cannot be placed in the source text."""


def load_text(doc_id: str) -> str:
    path = NORMALIZED_DIR / f"{doc_id}.txt"
    if not path.is_file():
        raise SpanError(f"{doc_id}: no normalized text at {path}")
    return path.read_text(encoding="utf-8")


# Characters these templates use for an unfilled field: dot leaders, the
# ellipsis glyph, dashes and underscores. A run of them is a blank, and its
# exact length carries no meaning — one template writes "Rs………..", another
# "Rs. --------". Requiring an annotator to count dots would produce failed
# matches, not accuracy.
FILL_RUN = re.compile(r"[.…–—\-_]{2,}")


def _token_pattern(token: str) -> str:
    """Escape a token, but let any run of fill characters match any other."""
    pieces = []
    position = 0
    for match in FILL_RUN.finditer(token):
        pieces.append(re.escape(token[position:match.start()]))
        pieces.append(r"[.…–—\-_\s]{2,}")
        position = match.end()
    pieces.append(re.escape(token[position:]))
    return "".join(pieces)


def find_all(text: str, quote: str) -> list[tuple[int, int]]:
    """Every occurrence of `quote` in `text`, ignoring whitespace layout and
    the exact length of unfilled-field runs."""
    tokens = quote.split()
    if not tokens:
        raise SpanError("empty span text")
    pattern = re.compile(r"\s+".join(_token_pattern(t) for t in tokens))
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def resolve(doc_id: str, quote: str, occurrence: int | None = None) -> tuple[int, int]:
    """Character offsets of `quote` within `doc_id`'s normalized text."""
    text = load_text(doc_id)
    matches = find_all(text, quote)

    if not matches:
        raise SpanError(
            f"{doc_id}: span not found — {quote[:70]!r}. "
            "The quote is not verbatim in the source."
        )
    if len(matches) > 1 and occurrence is None:
        raise SpanError(
            f"{doc_id}: span occurs {len(matches)} times — {quote[:70]!r}. "
            f"Set 'occurrence' (1-{len(matches)}) to disambiguate."
        )

    index = 0 if occurrence is None else occurrence - 1
    if not 0 <= index < len(matches):
        raise SpanError(
            f"{doc_id}: occurrence {occurrence} out of range; {len(matches)} match(es)"
        )

    # The fill-run pattern may absorb trailing layout whitespace. A gold span
    # should not begin or end on it.
    start, end = matches[index]
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def paragraphs_for(doc_id: str, start: int, end: int, paragraphs: list[dict]) -> list[str]:
    """Paragraph IDs the span overlaps, in document order."""
    return [
        p["paragraph_id"]
        for p in paragraphs
        if p["doc_id"] == doc_id and p["char_start"] < end and start < p["char_end"]
    ]


def build_answer(
    doc_id: str,
    quote: str,
    paragraphs: list[dict],
    occurrence: int | None = None,
    allow_furniture: bool = False,
) -> dict:
    """A validated answer record: offsets, source-exact text, paragraph_id.

    Rejects spans landing on template furniture — a running header is not
    part of the instrument, so a gold span drawn over one is a defect.

    `allow_furniture` is the one exception, and it exists for CAT-001. A
    document's name is printed in its masthead, and a masthead that is
    repeated across sibling templates — seven PSU tenders each headed
    GENERAL CONDITIONS OF CONTRACT — is classified as furniture by a rule
    that is right about every other block. Refusing the span there would
    force CAT-001 to be recorded empty for documents whose title is legible
    on page one, which is a worse error than pointing gold at a block the
    retrieval index skips. Callers pass it per category, never per span.
    """
    start, end = resolve(doc_id, quote, occurrence)
    text = load_text(doc_id)
    covered = paragraphs_for(doc_id, start, end, paragraphs)

    if not covered:
        raise SpanError(f"{doc_id}: span [{start},{end}) lies outside every paragraph")

    by_id = {p["paragraph_id"]: p for p in paragraphs}
    furniture = [pid for pid in covered if by_id[pid].get("is_boilerplate")]
    if furniture and not allow_furniture:
        raise SpanError(
            f"{doc_id}: span overlaps boilerplate {furniture} — "
            "running headers and disclaimers are not part of the instrument"
        )

    return {
        "text": text[start:end],
        "answer_start": start,
        "answer_end": end,
        "paragraph_id": covered[0],
        "paragraph_ids": covered,
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        print("usage: resolve_spans.py <DOC-ID> <quoted span text> [occurrence]")
        return 2

    doc_id, quote = sys.argv[1], sys.argv[2]
    occurrence = int(sys.argv[3]) if len(sys.argv) > 3 else None
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")

    try:
        answer = build_answer(doc_id, quote, paragraphs, occurrence)
    except SpanError as error:
        print(f"ERROR  {error}")
        return 1

    print(f"answer_start : {answer['answer_start']}")
    print(f"answer_end   : {answer['answer_end']}")
    print(f"paragraph_id : {answer['paragraph_id']}")
    if len(answer["paragraph_ids"]) > 1:
        print(f"  spans {len(answer['paragraph_ids'])} paragraphs: {answer['paragraph_ids']}")
    print(f"text         : {answer['text']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
