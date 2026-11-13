"""Stage 4 — paragraph segmentation.

The paragraph is both the retrieval chunk and the unit gold spans point
into, so this output is frozen once annotation begins.

Algorithm, in order:
  1. Split the normalized text into blocks on blank lines.
  2. Identify running furniture — letterhead, standing disclaimers, page
     headers, signature blocks — as blocks whose text repeats elsewhere in
     the corpus without opening a numbered or lettered clause.
  3. Merge a block shorter than MIN_CHARS into following blocks — a bare
     heading line is a poor retrieval chunk and a poor annotation target;
     a heading plus its clause body is a good one. Furniture never
     participates in a merge (see below).
  4. Split any block longer than MAX_CHARS at line boundaries, preferring
     lines that open a new numbered/lettered clause.

Why furniture is held out of step 3: these templates carry a running page
header ("Law Department, Andhra Pradesh Template regarding Commercial
Contracts") that is shorter than MIN_CHARS, so an unrestricted merge folds
it forward onto whatever clause follows it. That produced paragraphs like
header + "This agreement is subject to the laws and regulations of the
Government" — the only Governing Law clause in the core split, with 48% of
its characters being header noise. The header dilutes BM25 term weights and
forces an annotator to draw a gold span around text that is not part of the
instrument. Holding furniture out keeps clause bodies clean.

Furniture paragraphs are kept in the output, flagged `is_boilerplate: true`,
never deleted: the paragraph set still tiles the document, the offsets stay
auditable, and the decision is reversible. Downstream stages skip them.

Every paragraph satisfies, exactly:
    normalized_text[char_start:char_end] == paragraph["text"]

Merges and splits only ever move boundaries to offsets that already exist
in the frozen text, so that invariant is structural, not best-effort.

Outputs:
    corpus/paragraphs.jsonl
    validation/paragraph_report.json
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from indicuad_common import (
    CORPUS_DIR,
    NORMALIZED_DIR,
    Stage,
    emit_report,
    paragraph_id_for,
    read_jsonl,
    write_jsonl,
)

TINY_CHARS = 25       # a bare page number; always absorbed, cap ignored
MIN_CHARS = 80        # below this, a block is merged forward
MERGE_CAP = 1500      # never merge past this size
MAX_CHARS = 1800      # above this, a block is split

BLANK_LINE = re.compile(r"\n{2,}")

# Lines opening a new clause: "1.", "(a)", "12.3", "(iv)", "A."
CLAUSE_START = re.compile(
    r"^\s*(?:\(\s*[a-zA-Z0-9ivxIVX]{1,4}\s*\)|[0-9]+(?:\.[0-9]+)*[.)]|[A-Z][.)])\s+"
)

# Furniture test, deliberately looser than CLAUSE_START: the trailing
# punctuation is optional because the Reg-30 disclosure forms number their
# responses "5 Whether domestic or international" with no period. Those are
# document content and must not be mistaken for furniture.
ENUMERATED = re.compile(
    r"^\s*(?:\(\s*[a-zA-Z0-9ivxIVX]{1,4}\s*\)|[0-9]+(?:\.[0-9]+)*[.)]?|[A-Za-z][.)])\s+"
)

# A block must recur at least this often in the corpus to count as running
# furniture. Two is enough: a page header appears on every page it heads,
# and a standing disclaimer appears once per template.
FURNITURE_MIN_REPEATS = 2

# Minimum length for a block to be called furniture on containment alone.
# PDF text layers do not agree on where the blank lines fall: one template
# emits "<header>\n\n<disclaimer>" where the rest emit them as a single
# block, so the disclaimer appears exactly once as a standalone block and
# the repeat test misses it. Matching a long fragment against a known
# furniture block recovers those. The floor keeps a short header key from
# matching inside a genuine clause.
FURNITURE_MIN_FRAGMENT = 40

# Within a single document, a block repeating only twice is ambiguous: it is
# either a page header on a two-page document, or a placeholder written twice
# (two parties described identically). Length separates them — a letterhead
# runs to dozens of characters, a placeholder like "[name]" does not. Three
# or more repeats is a header regardless of length. The floor is set just
# below the shortest real case seen - a 26-character notice block
# ("Address: Attention: Email:") repeated for two parties - and the
# operative-language guard below is what stops it suppressing a real clause.
FURNITURE_MIN_IN_DOC = 3
FURNITURE_TWICE_MIN_LEN = 24

# Running furniture never carries an obligation. Letterheads, page headers,
# standing disclaimers and signature blocks do not say "shall". A repeated
# block that DOES is shared clause text, not furniture — sibling GCCs from
# one PSU reuse whole clauses, and treating those as furniture would delete
# real contract language from the index and from annotation.
OPERATIVE = re.compile("\\b(shall|must|hereby agrees?|undertakes?|indemnif)", re.IGNORECASE)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def normalize_block(text: str) -> str:
    """Comparison key for furniture detection — case, punctuation and
    whitespace differences must not hide a repeat."""
    return _NON_ALNUM.sub(" ", re.sub(r"\s+", " ", text.lower())).strip()


DERIVED_NOTE = re.compile(r"DERIVED: pages (\d+)-(\d+) of (\S+\.pdf)", re.I)


def furniture_keys(documents: list[dict]) -> set[str]:
    """Normalized text of every block that recurs in the corpus without
    opening an enumerated item.

    Computed corpus-wide rather than per-document so that a disclaimer
    appearing once per template is still caught — within a single document
    it looks unique.
    """
    counts: Counter[str] = Counter()
    per_doc: dict[str, Counter[str]] = {}
    raw: dict[str, str] = {}
    present = {d["filename"] for d in documents}
    for document in documents:
        # A derived document that sits ALONGSIDE its parent duplicates text
        # already counted under that parent. Counting both would push genuine
        # clauses over the repeat threshold and suppress them as furniture —
        # which is what happened when the annexed agreements were split out.
        #
        # But only skip when the parent is actually in the corpus. A slice
        # taken from a source that was never ingested (the PWD conditions are
        # pages 1-30 of a 104-page book that is not itself a document here)
        # duplicates nothing, and skipping it hides its own running headers
        # from the counter — which left "TENDER& CONTRACT PWD" unflagged.
        derived = DERIVED_NOTE.search(document.get("license_notes") or "")
        if derived and derived.group(3) in present:
            continue
        text = (NORMALIZED_DIR / f"{document['doc_id']}.txt").read_text(encoding="utf-8")
        for start, end in block_offsets(text):
            # Count the block itself AND the chunks it splits into. A running
            # header inside an oversized block is not a block of its own, so
            # counting blocks alone never sees it repeat — it then survives
            # splitting as a short unflagged paragraph. Seen with
            # "TENDER& CONTRACT PWD" in the PWD conditions.
            pieces = [(start, end)]
            if end - start > MAX_CHARS:
                pieces += split_long_block(text, start, end)
            for piece_start, piece_end in pieces:
                block = text[piece_start:piece_end]
                key = normalize_block(block)
                if not key:
                    continue
                counts[key] += 1
                per_doc.setdefault(key, Counter())[document["doc_id"]] += 1
                raw.setdefault(key, block)
    # Repeating twice is not enough on its own. A template that defines two
    # parties with identical placeholder text ("[name] [address] ...") repeats
    # exactly twice inside one document, and treating that as furniture
    # suppressed real Parties clauses in the Letter of Intent. Running
    # furniture has one of two signatures instead: it appears in more than one
    # document, or it appears many times within a single one (once per page).
    keep = set()
    for key, count in counts.items():
        if count < FURNITURE_MIN_REPEATS or ENUMERATED.match(raw[key]):
            continue
        if OPERATIVE.search(raw[key]):
            continue
        docs_seen = per_doc.get(key, Counter())
        most = max(docs_seen.values(), default=0)
        if (
            len(docs_seen) >= 2
            or most >= FURNITURE_MIN_IN_DOC
            or (most >= 2 and len(key) >= FURNITURE_TWICE_MIN_LEN)
        ):
            keep.add(key)
    return keep


def superseded_page_ranges(documents: list[dict]) -> dict[str, tuple[int, int]]:
    """Parent doc_id -> page range that a derived document now covers."""
    by_filename = {d["filename"]: d["doc_id"] for d in documents}
    ranges: dict[str, tuple[int, int]] = {}
    for document in documents:
        match = DERIVED_NOTE.search(document.get("license_notes") or "")
        if not match:
            continue
        first, last, parent_file = int(match[1]), int(match[2]), match[3]
        parent = by_filename.get(parent_file)
        if parent:
            ranges[parent] = (first, last)
    return ranges


def is_furniture(block: str, keys: set[str]) -> bool:
    """Whether a block is running furniture, given the corpus furniture set."""
    key = normalize_block(block)
    if not key:
        # No letters and no digits — a horizontal rule or a row of dots used
        # as a separator. It cannot carry a clause, so it is furniture by
        # definition rather than by repetition.
        return bool(block.strip())
    if ENUMERATED.match(block) or OPERATIVE.search(block):
        return False
    if key in keys:
        return True
    return len(key) >= FURNITURE_MIN_FRAGMENT and any(key in known for known in keys)


def block_offsets(text: str) -> list[tuple[int, int]]:
    """Blank-line-delimited blocks as (start, end), whitespace trimmed."""
    spans = []
    pos = 0
    for match in BLANK_LINE.finditer(text):
        spans.append((pos, match.start()))
        pos = match.end()
    spans.append((pos, len(text)))

    trimmed = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((start, end))
    return trimmed


def merge_short_blocks(
    text: str, blocks: list[tuple[int, int]], furniture: list[bool]
) -> list[tuple[int, int, bool]]:
    """Fold undersized blocks forward into the next one.

    Furniture blocks are boundaries: they are never absorbed into a
    neighbour and never absorb one, however short they are. Without that
    rule a sub-MIN_CHARS running header is folded onto the clause beneath
    it and the two can no longer be told apart.
    """
    merged: list[tuple[int, int, bool]] = []
    index = 0
    while index < len(blocks):
        start, end = blocks[index]
        if furniture[index]:
            merged.append((start, end, True))
            index += 1
            continue
        while (
            end - start < MIN_CHARS
            and index + 1 < len(blocks)
            and not furniture[index + 1]
        ):
            # A tiny runt (a bare page number) is absorbed unconditionally:
            # leaving it standing produces a meaningless retrieval chunk,
            # and appending it to a large block barely changes that block.
            tiny = end - start < TINY_CHARS
            if not tiny and blocks[index + 1][1] - start > MERGE_CAP:
                break
            index += 1
            end = blocks[index][1]
        merged.append((start, end, False))
        index += 1

    # A trailing runt (last block, nothing to merge forward into) folds
    # backward instead, so no paragraph is a lone page number. Neither side
    # may be furniture.
    if len(merged) >= 2:
        last_start, last_end, last_furniture = merged[-1]
        prev_start, prev_end, prev_furniture = merged[-2]
        tiny = last_end - last_start < TINY_CHARS
        if (
            not last_furniture
            and not prev_furniture
            and last_end - last_start < MIN_CHARS
            and (tiny or last_end - prev_start <= MERGE_CAP)
        ):
            merged[-2:] = [(prev_start, last_end, False)]
    return merged


def line_offsets(text: str, start: int, end: int) -> list[tuple[int, int]]:
    lines = []
    pos = start
    while pos < end:
        newline = text.find("\n", pos, end)
        stop = end if newline == -1 else newline
        lines.append((pos, stop))
        pos = stop + 1
    return lines


def split_long_block(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Greedily pack lines into <= MAX_CHARS chunks, breaking at clauses."""
    if end - start <= MAX_CHARS:
        return [(start, end)]

    lines = line_offsets(text, start, end)
    if len(lines) <= 1:
        # A single enormous line (dense table row). Leave it whole rather
        # than cutting mid-sentence; the report flags oversized paragraphs.
        return [(start, end)]

    chunks: list[tuple[int, int]] = []
    chunk_start = lines[0][0]
    chunk_end = lines[0][1]

    for line_start, line_end in lines[1:]:
        would_be = line_end - chunk_start
        opens_clause = bool(CLAUSE_START.match(text[line_start:line_end]))
        # Break when the chunk is full, or at a clause opener once the
        # chunk is already substantial enough to stand alone.
        if would_be > MAX_CHARS or (opens_clause and chunk_end - chunk_start >= MIN_CHARS):
            chunks.append((chunk_start, chunk_end))
            chunk_start, chunk_end = line_start, line_end
        else:
            chunk_end = line_end

    chunks.append((chunk_start, chunk_end))
    return chunks


def pages_for_span(page_spans: list[dict], start: int, end: int) -> tuple[int, int]:
    touched = [
        p["page"]
        for p in page_spans
        if p["char_start"] < end and start < p["char_end"]
    ]
    if not touched:
        # Span sits inside a page separator; attribute to the nearest page.
        nearest = min(page_spans, key=lambda p: abs(p["char_start"] - start))
        return nearest["page"], nearest["page"]
    return min(touched), max(touched)


def segment_document(document: dict, furniture: set[str]) -> list[dict]:
    doc_id = document["doc_id"]
    text = (NORMALIZED_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")

    raw_blocks = block_offsets(text)
    flags = [is_furniture(text[s:e], furniture) for s, e in raw_blocks]
    blocks = merge_short_blocks(text, raw_blocks, flags)

    final: list[tuple[int, int, bool]] = []
    for start, end, furniture_block in blocks:
        # Furniture is emitted whole; splitting it would only produce more
        # of it, and it is excluded downstream in any case.
        if furniture_block:
            final.append((start, end, True))
            continue
        # Re-test after splitting. Some PDFs emit no blank line between the
        # disclaimer and the body, so the furniture is buried inside an
        # oversized block and only becomes a separable unit here.
        final.extend(
            (s, e, is_furniture(text[s:e], furniture))
            for s, e in split_long_block(text, start, end)
        )

    paragraphs = []
    for index, (start, end, furniture_block) in enumerate(final):
        page_start, page_end = pages_for_span(document["page_spans"], start, end)
        paragraphs.append(
            {
                "paragraph_id": paragraph_id_for(doc_id, index),
                "doc_id": doc_id,
                "paragraph_index": index,
                "page_start": page_start,
                "page_end": page_end,
                "char_start": start,
                "char_end": end,
                "char_count": end - start,
                "text": text[start:end],
                # Running header / letterhead / disclaimer / signature block.
                # Retained for offset completeness; excluded from the retrieval
                # index, the P2 sweep, and gold spans.
                "is_boilerplate": furniture_block,
                # Set in main() where the derived-document map is available.
                "is_superseded": False,
            }
        )
    return paragraphs


def main() -> int:
    stage = Stage("4", "Paragraph segmentation")

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    if not documents:
        stage.fail("corpus/documents.jsonl missing — run normalize_corpus.py first.")
        return 1

    existing = CORPUS_DIR / "paragraphs.jsonl"
    if existing.is_file():
        stage.warn(
            "corpus/paragraphs.jsonl already exists and will be regenerated. "
            "If annotation has begun, STOP: paragraph IDs must not change."
        )

    # Pass 1: which blocks are running furniture? Needs every document in
    # hand, so it cannot be folded into the per-document segmentation below.
    furniture = furniture_keys(documents)
    stage.info(f"running-furniture blocks identified: {len(furniture)} distinct texts")

    # Where an agreement was split out of a larger document, the parent still
    # holds the same pages. Those parent paragraphs are byte-identical
    # competitors to the child's gold paragraphs, so leaving both in the
    # retrieval index would depress recall for a reason that has nothing to
    # do with the retriever. Mark the parent's copy superseded; the rest of
    # the parent stays indexed as a hard negative.
    superseded = superseded_page_ranges(documents)
    if superseded:
        stage.info(
            "superseded page ranges: "
            + ", ".join(f"{d} pp.{a}-{b}" for d, (a, b) in sorted(superseded.items()))
        )

    all_paragraphs: list[dict] = []
    per_doc = []
    oversized = []

    # Pass 2: segment.
    for document in documents:
        paragraphs = segment_document(document, furniture)
        window = superseded.get(document["doc_id"])
        for paragraph in paragraphs:
            paragraph["is_superseded"] = bool(
                window
                and paragraph["page_start"] <= window[1]
                and paragraph["page_end"] >= window[0]
            )
        all_paragraphs.extend(paragraphs)

        sizes = [p["char_count"] for p in paragraphs]
        big = [p["paragraph_id"] for p in paragraphs if p["char_count"] > MAX_CHARS]
        boilerplate = [p["paragraph_id"] for p in paragraphs if p["is_boilerplate"]]
        oversized.extend(big)
        per_doc.append(
            {
                "doc_id": document["doc_id"],
                "paragraphs": len(paragraphs),
                "boilerplate_paragraphs": len(boilerplate),
                "min_chars": min(sizes) if sizes else 0,
                "median_chars": sorted(sizes)[len(sizes) // 2] if sizes else 0,
                "max_chars": max(sizes) if sizes else 0,
                "oversized_paragraph_ids": big,
                "boilerplate_paragraph_ids": boilerplate,
            }
        )
        stage.info(
            f"{document['doc_id']}  paragraphs={len(paragraphs):>4}  "
            f"median={sorted(sizes)[len(sizes)//2] if sizes else 0:>5}  "
            f"max={max(sizes) if sizes else 0:>5}  "
            f"boilerplate={len(boilerplate):>3}"
        )

    # ---- structural self-check before writing -----------------------------
    problems = []
    ids = Counter(p["paragraph_id"] for p in all_paragraphs)
    duplicates = [i for i, c in ids.items() if c > 1]
    if duplicates:
        problems.append(f"duplicate paragraph_ids: {duplicates[:10]}")

    by_doc: dict[str, list[dict]] = {}
    for paragraph in all_paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    slice_mismatches = 0
    overlaps = 0
    for doc_id, paragraphs in by_doc.items():
        text = (NORMALIZED_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
        previous_end = -1
        for paragraph in paragraphs:
            if text[paragraph["char_start"]:paragraph["char_end"]] != paragraph["text"]:
                slice_mismatches += 1
            if paragraph["char_start"] < previous_end:
                overlaps += 1
            previous_end = paragraph["char_end"]

    if slice_mismatches:
        problems.append(f"{slice_mismatches} paragraphs whose text != source slice")
    if overlaps:
        problems.append(f"{overlaps} overlapping paragraphs")

    # Suppressing a paragraph that carries an obligation would silently
    # delete annotatable content, which is far worse than leaving a header
    # in the index. Fail loudly instead of guessing.
    operative = re.compile(r"\b(shall|must|hereby agrees?|undertakes?|indemnif)", re.I)
    contaminated = [
        p["paragraph_id"]
        for p in all_paragraphs
        if p["is_boilerplate"] and operative.search(p["text"])
    ]
    if contaminated:
        problems.append(
            f"{len(contaminated)} paragraph(s) flagged boilerplate but containing "
            f"operative language: {contaminated[:10]}"
        )

    if problems:
        for problem in problems:
            stage.fail(problem)
        emit_report(
            "paragraph_report.json",
            {"status": "FATAL", "problems": problems},
        )
        return 1

    write_jsonl(CORPUS_DIR / "paragraphs.jsonl", all_paragraphs)

    sizes = [p["char_count"] for p in all_paragraphs]
    stage.info(f"wrote corpus/paragraphs.jsonl ({len(all_paragraphs)} paragraphs)")

    emit_report(
        "paragraph_report.json",
        {
            "status": "PASS",
            "paragraphs_total": len(all_paragraphs),
            "documents_total": len(documents),
            "unique_paragraph_ids": len(ids),
            "slice_exactness_verified": True,
            "overlapping_paragraphs": 0,
            "size_chars": {
                "min": min(sizes),
                "median": sorted(sizes)[len(sizes) // 2],
                "mean": round(sum(sizes) / len(sizes), 1),
                "max": max(sizes),
            },
            "oversized_paragraphs": oversized,
            "boilerplate": {
                "distinct_furniture_texts": len(furniture),
                "paragraphs_flagged": sum(1 for p in all_paragraphs if p["is_boilerplate"]),
                "rule": (
                    "A blank-line block whose normalized text recurs at least "
                    f"{FURNITURE_MIN_REPEATS} times corpus-wide and does not open an "
                    "enumerated item. Flagged, never deleted; excluded from the BM25 "
                    "index, the P2 sweep and gold spans. Furniture blocks are also held "
                    "out of the short-block merge so they cannot be folded onto a clause."
                ),
                "operative_language_check": "PASS",
            },
            "params": {
                "min_chars": MIN_CHARS,
                "merge_cap": MERGE_CAP,
                "max_chars": MAX_CHARS,
                "furniture_min_repeats": FURNITURE_MIN_REPEATS,
            },
            "per_document": per_doc,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
