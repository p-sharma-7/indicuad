"""Split gold answers that straddle a paragraph boundary (§6 rule 7).

A gold answer must lie inside one paragraph. Answers drift across a
boundary when the paragraph set is rebuilt — a clause that used to sit in
one block gets cut in two by a page break or a re-extraction, and the
offsets, which are still exact, now overflow the block they point into.

The repair keeps the annotator's judgement and only re-expresses it: the
answer is cut at the paragraph seams into one answer per paragraph, with
leading and trailing whitespace trimmed off each piece. No character of
the instrument is added or dropped, and `text` is re-sliced from the
frozen normalized file so it stays source-exact.

What this deliberately does NOT do is move a boundary or discard the
shorter piece. Either would silently change what the gold claims.

    split_straddling_spans.py [--apply]

Output: annotations/<chain head>, validation/span_split_report.json
"""

from __future__ import annotations

import argparse
import sys

from indicuad_common import (
    CORPUS_DIR,
    NORMALIZED_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the result")
    args = parser.parse_args()

    stage = Stage("A/S", "Split straddling gold spans")

    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    by_doc: dict[str, list[dict]] = {}
    for paragraph in paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)
    for blocks in by_doc.values():
        blocks.sort(key=lambda p: p["char_start"])

    path = current_annotations_path()
    cells = read_jsonl(path)
    texts: dict[str, str] = {}

    split_cells: list[str] = []
    pieces_made = 0

    for cell in cells:
        answers = cell.get("answers") or []
        if not answers:
            continue
        doc_id = cell["doc_id"]
        blocks = by_doc.get(doc_id, [])
        text = texts.setdefault(
            doc_id, (NORMALIZED_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
        )

        rebuilt: list[dict] = []
        changed = False
        for answer in answers:
            start, end = answer["answer_start"], answer["answer_end"]
            covered = [b for b in blocks if b["char_start"] < end and start < b["char_end"]]
            if len(covered) <= 1:
                rebuilt.append(answer)
                continue
            changed = True
            for block in covered:
                piece_start = max(start, block["char_start"])
                piece_end = min(end, block["char_end"])
                while piece_start < piece_end and text[piece_start].isspace():
                    piece_start += 1
                while piece_end > piece_start and text[piece_end - 1].isspace():
                    piece_end -= 1
                if piece_end <= piece_start:
                    continue
                rebuilt.append(
                    {
                        **answer,
                        "text": text[piece_start:piece_end],
                        "answer_start": piece_start,
                        "answer_end": piece_end,
                        "paragraph_id": block["paragraph_id"],
                        "paragraph_ids": [block["paragraph_id"]],
                    }
                )
                pieces_made += 1

        if changed:
            cell["answers"] = rebuilt
            cell["gold_paragraph_ids"] = sorted(
                {a["paragraph_id"] for a in rebuilt if a.get("paragraph_id")}
            )
            split_cells.append(cell["cell_id"])

    stage.info(f"cells with a straddling answer: {len(split_cells)}")
    for cell_id in split_cells:
        stage.info(f"    {cell_id}")
    stage.info(f"pieces produced: {pieces_made}")

    if args.apply:
        write_jsonl(path, cells)
        stage.info(f"wrote {path.name}")
    else:
        stage.info("dry run — pass --apply to write")

    emit_report(
        "span_split_report.json",
        {
            "status": "PASS" if args.apply else "DRY_RUN",
            "cells_split": split_cells,
            "pieces_produced": pieces_made,
            "note": (
                "An answer that crossed a paragraph boundary was cut at the seam into "
                "one answer per paragraph. The character range is unchanged apart from "
                "whitespace trimmed at each seam."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
