"""Extract text from Word source documents (.docx and legacy .doc).

Several Startup India model contracts are published as Word files rather
than PDFs, and some carry a .docx extension while actually being legacy
Word 97-2003 binaries. Both are handled here so the corpus can use them.

**Why this is acceptable for a verbatim-span benchmark.** The extracted
text becomes the frozen reference that every gold offset points into, so a
span is exact by construction regardless of extraction method. The risk is
not corruption but *omission*: if extraction silently drops a passage, a
clause that exists in the source is invisible to annotation and the cell
is recorded empty. That is the same failure the bulk-approval incident
produced, so the extractor reports a coverage estimate and refuses files
whose output looks truncated.

    .docx   a zip; read word/document.xml, strip tags, keep paragraph breaks.
    .doc    OLE2 binary. Body text is stored UTF-16LE, so printable runs are
            recovered by scanning for [\\x20-\\x7e]\\x00 sequences. This loses
            runs shorter than MIN_RUN characters — usually table cells and
            single words — which is why short output is rejected rather than
            quietly accepted.

    python scripts/extract_word_text.py <file> [...]     print a report
"""

from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path

# Below this, a UTF-16 run is more likely field codes or style names than
# document text. Lowering it recovers a little more prose and a lot of noise.
MIN_RUN = 12

# A legacy .doc whose recovered text is a tiny fraction of its file size is
# probably mostly unrecovered. Word overhead is large, so this is generous.
MIN_YIELD_RATIO = 0.04

# Word writes body text either UTF-16LE (Windows) or single-byte cp1252
# (older Mac Word). Neither header nor extension tells you which, so both
# are tried and the one recovering more text wins.
UTF16_RUN = re.compile(rb"(?:[\x20-\x7e]\x00){%d,}" % MIN_RUN)
CP1252_RUN = re.compile(rb"[\x20-\x7e]{%d,}" % MIN_RUN)
TAG = re.compile(r"<[^>]+>")
PARA_END = re.compile(r"</w:p>")


class ExtractionError(ValueError):
    """The file could not be read, or the result looks truncated."""


def from_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    xml = PARA_END.sub("\n\n", xml)
    text = html.unescape(TAG.sub("", xml))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _assemble(runs: list[str]) -> str:
    """Deduplicate and join runs, preserving first-occurrence order.

    Word repeats some runs (headers, style samples); keeping the first
    occurrence keeps the body reading in order without duplicated blocks.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for run in runs:
        cleaned = run.strip()
        if len(cleaned) < MIN_RUN or cleaned in seen:
            continue
        seen.add(cleaned)
        kept.append(cleaned)
    return "\n\n".join(kept)


# Artefacts of the OLE2 container that the run scanner cannot distinguish
# from prose: the font table and the stream directory. They always precede
# the body, so they are stripped as a leading run rather than searched for
# throughout — a clause could legitimately mention "Microsoft", and deleting
# it mid-document would be a silent edit to the instrument.
_ARTEFACT = re.compile(
    r"^(?:"
    r"[A-Z][A-Za-z]*(?:[ -][A-Za-z0-9]+)*(?:MT|PS|PSMT|-BoldMT|-ItalicMT)|"
    r"(?:Times|Arial|Calibri|Cambria|Liberation|Droid|Lohit|Symbol|Wingdings|"
    r"Courier|Georgia|Verdana|Tahoma|Mangal|Nirmala|SimSun|MS Mincho)"
    r"[A-Za-z0-9 ]*|"
    r"WordDocument|ObjectPool|SummaryInformation|DocumentSummaryInformation|"
    r"CompObj|Data|1Table|0Table|Macros|MSWordDoc|Word\.Document(?:\.\d+)?|"
    r"Microsoft(?: Office)?(?: Word)?(?: \d+\.\d+)?|"
    r"(?:Normal|Default|List|Body|Table|Header|Footer|Heading|Title|Caption|"
    r"Footnote|Endnote|Balloon|Placeholder|Hyperlink)[A-Za-z ]*"
    r")$"
)


def strip_container_artefacts(text: str) -> str:
    """Drop the font table and stream names that head a legacy .doc scan.

    The run scanner reads every printable byte sequence in the OLE2
    container, so the extracted text opens with the document's font list
    ("Times New Roman", "Liberation Serif", "TimesNewRomanPS-BoldMT") and
    sometimes with Word's stream directory. Those are not part of the
    instrument. Five Startup India templates were dropped from this corpus
    because that debris reached the segmenter and tripped the boilerplate
    guard; stripping it at the source recovers them.

    Only the LEADING and TRAILING runs of artefact blocks are removed, and
    only blocks that match end to end — the font table heads the scan and
    the stream directory tails it. The first block that looks like prose
    ends each strip, so nothing inside the contract is touched.
    """
    blocks = text.split("\n\n")
    start = 0
    while start < len(blocks) and _ARTEFACT.match(blocks[start].strip()):
        start += 1
    end = len(blocks)
    while end > start and _ARTEFACT.match(blocks[end - 1].strip()):
        end -= 1
    return "\n\n".join(blocks[start:end]).strip()


def from_legacy_doc(path: Path) -> str:
    raw = path.read_bytes()

    candidates = [
        strip_container_artefacts(
            _assemble([m.decode("utf-16-le", errors="replace") for m in UTF16_RUN.findall(raw)])
        ),
        strip_container_artefacts(
            _assemble([m.decode("cp1252", errors="replace") for m in CP1252_RUN.findall(raw)])
        ),
    ]
    text = max(candidates, key=len)
    if not text:
        raise ExtractionError("no readable text runs found; not a Word binary")
    if len(text) < len(raw) * MIN_YIELD_RATIO:
        raise ExtractionError(
            f"recovered only {len(text):,} chars from a {len(raw):,}-byte file "
            f"({len(text)/len(raw):.1%}); likely truncated — do not use"
        )
    return text


def extract(path: Path) -> tuple[str, str]:
    """(text, format) for a Word file, choosing by content not extension."""
    head = path.read_bytes()[:8]
    if head[:2] == b"PK":
        return from_docx(path), "docx"
    if head == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return from_legacy_doc(path), "doc (legacy OLE2)"
    raise ExtractionError(f"unrecognised format, header {head!r}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    failed = 0
    for name in sys.argv[1:]:
        path = Path(name)
        try:
            text, fmt = extract(path)
        except (ExtractionError, OSError, zipfile.BadZipFile) as error:
            print(f"FAIL  {path.name:<22} {error}")
            failed += 1
            continue
        words = len(text.split())
        print(f"OK    {path.name:<22} {fmt:<18} {len(text):>7,} chars  {words:>6,} words")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
