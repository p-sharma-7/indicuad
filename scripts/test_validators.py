"""Self-test for the validation suite.

A QC engine nobody has seen fail is not evidence of a clean corpus. This
injects each defect class the suite claims to catch and asserts it is
actually caught, using in-memory records built from the real corpus. It
writes nothing and never touches corpus or annotation files.

Run: python scripts/test_validators.py
"""

from __future__ import annotations

import sys

from indicuad_common import CORPUS_DIR, NORMALIZED_DIR, read_jsonl
from validators import (
    validate_duplicates,
    validate_leakage,
    validate_paragraphs,
    validate_queries,
    validate_schema,
    validate_source,
    validate_spans,
)

RESULTS: list[tuple[bool, str]] = []


def check(name: str, report: dict, expected_code: str) -> None:
    codes = {f.get("code") for f in (report.get("findings") or [])}
    ok = expected_code in codes
    RESULTS.append((ok, f"{name}: expected {expected_code}, got {sorted(codes) or 'none'}"))


def base_data() -> dict:
    return {
        "documents": read_jsonl(CORPUS_DIR / "documents.jsonl"),
        "paragraphs": read_jsonl(CORPUS_DIR / "paragraphs.jsonl"),
        "categories": read_jsonl(CORPUS_DIR / "categories.jsonl"),
        "annotations": [],
        "qa": [],
    }


def good_cell(paragraph: dict, text: str) -> dict:
    """A correct positive cell whose span is verbatim-exact."""
    start = paragraph["char_start"]
    end = min(paragraph["char_end"], start + 120)
    return {
        "qa_id": "IND-Q-TEST-1",
        "doc_id": paragraph["doc_id"],
        "category_id": "CAT-008",
        "question": "test",
        "question_register": "canonical",
        "intent_group_id": "IG-TEST",
        "is_impossible": False,
        "answers": [{
            "text": text[start:end],
            "answer_start": start,
            "answer_end": end,
            "paragraph_id": paragraph["paragraph_id"],
        }],
        "gold_paragraph_ids": [paragraph["paragraph_id"]],
    }


def main() -> int:
    print("=== Validation suite self-test ===\n")

    data = base_data()
    if not data["paragraphs"]:
        print("corpus not built; run the pipeline first")
        return 1

    paragraph = next(p for p in data["paragraphs"] if p["char_count"] > 200)
    text = (NORMALIZED_DIR / f"{paragraph['doc_id']}.txt").read_text(encoding="utf-8")

    # --- control: a correct cell must produce no span findings -------------
    clean = dict(data, qa=[good_cell(paragraph, text)])
    report = validate_spans(clean)
    RESULTS.append((
        not (report.get("findings") or []),
        f"control: a verbatim-exact span produces no findings "
        f"(got {[f['code'] for f in report.get('findings') or []]})",
    ))

    # --- control: an unannotated cell is not a defect ------------------------
    # is_impossible=null means "no verdict yet". Treating it as false would
    # report every scaffolded cell as a broken positive.
    blank = good_cell(paragraph, text)
    blank.update({"is_impossible": None, "answers": [], "gold_paragraph_ids": []})
    report = validate_spans(dict(data, qa=[blank]))
    RESULTS.append((
        not (report.get("findings") or []) and report.get("cells_unannotated") == 1,
        f"control: an unannotated cell produces no findings "
        f"(got {[f['code'] for f in report.get('findings') or []]}, "
        f"unannotated={report.get('cells_unannotated')})",
    ))

    # --- D. span text drifted from source ----------------------------------
    cell = good_cell(paragraph, text)
    cell["answers"][0]["text"] = cell["answers"][0]["text"].replace(" ", "  ", 1)
    check("span text mismatch", validate_spans(dict(data, qa=[cell])), "SPAN_TEXT_MISMATCH")

    # --- D. inverted offsets ------------------------------------------------
    cell = good_cell(paragraph, text)
    cell["answers"][0]["answer_start"], cell["answers"][0]["answer_end"] = (
        cell["answers"][0]["answer_end"], cell["answers"][0]["answer_start"],
    )
    check("inverted offsets", validate_spans(dict(data, qa=[cell])), "INVERTED_OFFSETS")

    # --- D. span pointing outside its declared paragraph --------------------
    other = next(
        p for p in data["paragraphs"]
        if p["doc_id"] == paragraph["doc_id"] and p["paragraph_id"] != paragraph["paragraph_id"]
    )
    cell = good_cell(paragraph, text)
    cell["answers"][0]["paragraph_id"] = other["paragraph_id"]
    check("span outside paragraph", validate_spans(dict(data, qa=[cell])), "SPAN_OUTSIDE_PARAGRAPH")

    # --- D. unknown paragraph reference -------------------------------------
    cell = good_cell(paragraph, text)
    cell["answers"][0]["paragraph_id"] = "PAR-DOC999-9999"
    check("unknown paragraph ref", validate_spans(dict(data, qa=[cell])), "UNKNOWN_PARAGRAPH_REF")

    # --- E. negative cell carrying an answer --------------------------------
    cell = good_cell(paragraph, text)
    cell["is_impossible"] = True
    check("negative with answers", validate_spans(dict(data, qa=[cell])), "NEGATIVE_WITH_ANSWERS")

    # --- F. registers disagreeing on gold ------------------------------------
    a = good_cell(paragraph, text)
    b = good_cell(paragraph, text)
    b["qa_id"], b["question_register"] = "IND-Q-TEST-2", "semi_formal"
    b["answers"][0]["answer_end"] -= 10
    b["answers"][0]["text"] = text[b["answers"][0]["answer_start"]:b["answers"][0]["answer_end"]]
    check("register gold divergence", validate_queries(dict(data, qa=[a, b])),
          "REGISTER_GOLD_DIVERGENCE")

    # --- F. duplicate qa_id ---------------------------------------------------
    a = good_cell(paragraph, text)
    b = good_cell(paragraph, text)
    b["question_register"] = "semi_formal"
    check("duplicate qa_id", validate_queries(dict(data, qa=[a, b])), "DUPLICATE_QA_ID")

    # --- G. split leakage ------------------------------------------------------
    a = good_cell(paragraph, text)
    b = good_cell(paragraph, text)
    a["split"], b["split"] = "core", "negatives"
    b["qa_id"] = "IND-Q-TEST-2"
    check("split leakage", validate_leakage(dict(data, qa=[a, b])), "SPLIT_LEAKAGE")

    # --- C. paragraph text drifted from frozen source --------------------------
    bad = [dict(p) for p in data["paragraphs"][:5]]
    bad[2]["text"] = bad[2]["text"] + " TAMPERED"
    check("paragraph text drift", validate_paragraphs(dict(data, paragraphs=bad)),
          "PARAGRAPH_TEXT_DRIFT")

    # --- C. overlapping paragraphs ---------------------------------------------
    bad = [dict(p) for p in data["paragraphs"][:5]]
    bad[3]["char_start"] = bad[2]["char_start"]
    bad[3]["text"] = text[bad[3]["char_start"]:bad[3]["char_end"]]
    check("overlapping paragraphs", validate_paragraphs(dict(data, paragraphs=bad)),
          "OVERLAPPING_PARAGRAPHS")

    # --- A. duplicate id / missing field ----------------------------------------
    bad = [dict(p) for p in data["paragraphs"][:4]]
    bad[1]["paragraph_id"] = bad[0]["paragraph_id"]
    check("duplicate paragraph id", validate_schema(dict(data, paragraphs=bad)), "DUPLICATE_ID")

    bad = [dict(d) for d in data["documents"][:3]]
    bad[1].pop("source_url")
    check("missing required field", validate_schema(dict(data, documents=bad)), "MISSING_FIELD")

    # --- B. source metadata drift --------------------------------------------------
    bad = [dict(d) for d in data["documents"]]
    bad[0]["license_notes"] = "Public domain (invented claim)"
    check("source metadata drift", validate_source(dict(data, documents=bad)), "METADATA_DRIFT")

    # --- B. source substitution -----------------------------------------------------
    bad = [dict(d) for d in data["documents"][:3]]
    bad[0]["filename"] = "not_in_manifest.pdf"
    check("source substitution", validate_source(dict(data, documents=bad)), "SOURCE_SUBSTITUTION")

    # --- H. duplicate document ---------------------------------------------------------
    bad = [dict(d) for d in data["documents"][:3]]
    bad[1]["text_sha256"] = bad[0]["text_sha256"]
    check("duplicate document", validate_duplicates(dict(data, documents=bad)), "DUPLICATE_DOCUMENT")

    print()
    passed = 0
    for ok, message in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
        passed += ok

    print(f"\n  {passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
