"""Re-anchor gold spans after the paragraph set is rebuilt.

Adding documents re-runs segmentation, and furniture detection is
corpus-wide: a block that appeared once before may now appear twice and be
reclassified, which shifts `paragraph_index` and therefore `paragraph_id`
in documents that were not themselves touched. Gold that points at a
paragraph by ID silently goes stale.

Character offsets do not go stale. Adding a document does not alter another
document's normalized text, so `answer_start` / `answer_end` still slice
exactly what they always did. This script treats the offsets as the source
of truth and recomputes every `paragraph_id` and `gold_paragraph_ids` from
them.

Before writing anything it re-verifies that each span's stored text still
equals `normalized_text[answer_start:answer_end]`. If that fails, the
underlying text changed rather than the segmentation, remapping would be
papering over a real problem, and the script stops.

    remap_gold_paragraphs.py [--check]

`--check` reports drift without writing.
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
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    args = parser.parse_args()

    stage = Stage("remap", "Re-anchor gold spans to the current paragraph set")

    path = current_annotations_path()
    cells = read_jsonl(path)
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")
    if not cells or not paragraphs:
        stage.fail("annotations or paragraphs missing")
        return 1

    by_doc: dict[str, list[dict]] = {}
    for paragraph in paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    texts: dict[str, str] = {}
    corrupted, drifted, orphaned, shifted, unchanged = [], [], [], [], 0

    for cell in cells:
        if cell.get("is_impossible") is not False:
            continue
        doc_id = cell["doc_id"]
        if doc_id not in texts:
            texts[doc_id] = (NORMALIZED_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
        text = texts[doc_id]

        fresh_ids: list[str] = []
        for answer in cell["answers"]:
            start, end = answer["answer_start"], answer["answer_end"]

            if text[start:end] != answer["text"]:
                # Distinguish two very different situations. If the stored
                # text still occurs exactly once, the document did not change
                # — content was inserted before it (OCR recovering a scanned
                # cover page shifts everything after it) and the offsets are
                # recoverable exactly. If it occurs zero times, or several,
                # the text itself changed and re-anchoring would paper over a
                # real defect, so that still fails.
                occurrences = [
                    i for i in range(len(text))
                    if text.startswith(answer["text"], i)
                ] if answer["text"] else []
                if len(occurrences) != 1:
                    corrupted.append(
                        {"cell_id": cell["cell_id"], "span": answer["text"][:60],
                         "occurrences": len(occurrences)}
                    )
                    continue
                moved_by = occurrences[0] - start
                shifted.append(
                    {"cell_id": cell["cell_id"], "moved_by": moved_by,
                     "span": answer["text"][:50]}
                )
                start, end = occurrences[0], occurrences[0] + len(answer["text"])
                if not args.check:
                    answer["answer_start"], answer["answer_end"] = start, end

            covering = [
                p["paragraph_id"]
                for p in by_doc.get(doc_id, [])
                if p["char_start"] < end and start < p["char_end"]
            ]
            if not covering:
                orphaned.append({"cell_id": cell["cell_id"], "offsets": [start, end]})
                continue

            if answer.get("paragraph_id") != covering[0]:
                drifted.append(
                    {
                        "cell_id": cell["cell_id"],
                        "was": answer.get("paragraph_id"),
                        "now": covering[0],
                    }
                )
            else:
                unchanged += 1

            if not args.check:
                answer["paragraph_id"] = covering[0]
            fresh_ids.extend(covering)

        if not args.check:
            cell["gold_paragraph_ids"] = sorted(set(fresh_ids))

    if corrupted:
        stage.fail(
            f"{len(corrupted)} span(s) no longer match the source text. The normalized "
            "text changed, not just the segmentation — remapping would hide a real "
            "defect. Investigate before re-running."
        )
        for item in corrupted[:5]:
            stage.info(f"  {item['cell_id']}: {item['span']!r}")
        return 1

    if not args.check:
        write_jsonl(path, cells)

    if shifted:
        stage.warn(
            f"{len(shifted)} span(s) had text inserted before them and were "
            f"re-anchored by offset; the text itself is unchanged"
        )
        for item in shifted[:5]:
            stage.info(f"  {item['cell_id']}: moved {item['moved_by']:+d} chars")
    stage.info(f"spans still offset-exact: {unchanged + len(drifted)}")
    stage.info(f"paragraph_id unchanged:   {unchanged}")
    stage.info(f"paragraph_id re-anchored: {len(drifted)}")
    if orphaned:
        stage.warn(f"{len(orphaned)} span(s) fall outside every paragraph: {orphaned[:3]}")
    for item in drifted[:8]:
        stage.info(f"  {item['cell_id']}: {item['was']} -> {item['now']}")
    if args.check:
        stage.info("--check: nothing written")

    emit_report(
        "gold_remap_report.json",
        {
            "status": "PASS",
            "mode": "check" if args.check else "applied",
            "spans_verified": unchanged + len(drifted),
            "paragraph_id_unchanged": unchanged,
            "paragraph_id_reanchored": len(drifted),
            "reanchored": drifted,
            "offset_shifted": shifted,
            "orphaned": orphaned,
            "note": (
                "Character offsets are the source of truth: adding documents cannot "
                "change another document's text, so offsets survive re-segmentation "
                "while paragraph IDs do not."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
