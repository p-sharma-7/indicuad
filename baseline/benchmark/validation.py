"""Dataset validation — must run first (spec §5).

Produces results/indicuad_v2/reports/dataset_validation.{json,md}. Every
check is read-only against the dataset.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.config import Config, read_json, read_jsonl, sha256_text, utc_now, write_json

# Files this harness must never write to. Hashed before and after validate()
# runs so a regression is caught automatically rather than assumed.
NEVER_TOUCH_FILES = [
    "qa/qa.json",
    "qa/qa_unanswerable.json",
    "qa/distractors.json",
    "corpus/documents.jsonl",
    "corpus/paragraphs.jsonl",
    "corpus/categories.jsonl",
]

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class Check:
    section: str
    name: str
    status: str  # PASS | FAIL | WARN | INFO
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"section": self.section, "name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class ValidationResult:
    checks: list[Check] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def add(self, section: str, name: str, status: str, detail: Any = None) -> None:
        self.checks.append(Check(section, name, status, detail))

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    @property
    def overall_status(self) -> str:
        return "FAIL" if self.n_fail > 0 else ("PASS_WITH_WARNINGS" if self.n_warn > 0 else "PASS")


def _hash_never_touch(cfg: Config) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for rel in NEVER_TOUCH_FILES:
        p = cfg.dataset.root / rel
        hashes[rel] = sha256_text(p.read_text(encoding="utf-8")) if p.is_file() else None
    return hashes


def _normalize_for_compare(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def _estimate_tokens(text: str) -> int:
    """Whitespace/punctuation-approximate token estimate, clearly labelled
    as an estimate (not a real model tokenizer) per spec §5.14."""
    return max(1, len(re.findall(r"\S+", text)))


def run_validation(cfg: Config, dataset=None) -> ValidationResult:
    """`dataset` may be pre-loaded (benchmark.dataset.Dataset); if None this
    loads it fresh via benchmark.dataset.load_dataset. Kept as a parameter
    so CLI callers don't pay the load cost twice."""
    from benchmark.dataset import load_dataset  # local import: avoid cycle

    result = ValidationResult(started_at=utc_now())
    pre_hashes = _hash_never_touch(cfg)

    # ---- 1. qa.json exists and parses -----------------------------------
    if not cfg.dataset.qa_release.is_file():
        result.add("release", "qa.json exists", "FAIL", "file not found")
        result.finished_at = utc_now()
        return result
    try:
        qa_rows = read_json(cfg.dataset.qa_release)
        result.add("release", "qa.json exists and parses", "PASS", f"{len(qa_rows)} rows")
    except Exception as exc:  # noqa: BLE001
        result.add("release", "qa.json exists and parses", "FAIL", str(exc))
        result.finished_at = utc_now()
        return result

    # ---- 2. exactly one record structure detected -------------------------
    field_sets = {frozenset(r.keys()) for r in qa_rows}
    if len(field_sets) == 1:
        result.add("release", "single record structure detected", "PASS", sorted(next(iter(field_sets))))
    else:
        result.add("release", "single record structure detected", "WARN",
                    {"distinct_field_sets": len(field_sets),
                     "field_sets": [sorted(fs) for fs in field_sets]})

    # ---- 3. all rows split == core -----------------------------------------
    non_core = [r["qa_id"] for r in qa_rows if r.get("split") != "core"]
    result.add("release", "every row split=core", "PASS" if not non_core else "FAIL",
                f"{len(non_core)} non-core rows" if non_core else "0 non-core rows")

    # ---- 4. all rows answerable --------------------------------------------
    bad_impossible = [r["qa_id"] for r in qa_rows if r.get("is_impossible") is not False]
    bad_empty = [r["qa_id"] for r in qa_rows if not r.get("answers")]
    result.add("release", "is_impossible is False on every row", "PASS" if not bad_impossible else "FAIL",
                f"{len(bad_impossible)} rows with is_impossible != False")
    result.add("release", "every row carries >=1 answer", "PASS" if not bad_empty else "FAIL",
                f"{len(bad_empty)} rows with empty answers[]")

    # ---- 5. counts ----------------------------------------------------------
    qa_ids = [r["qa_id"] for r in qa_rows]
    intent_groups = {r["intent_group_id"] for r in qa_rows}
    doc_ids = {r["doc_id"] for r in qa_rows}
    cat_ids = {r["category_id"] for r in qa_rows}
    # "651 gold spans" in the README is a per-intent-group count: all three
    # question_register rows of a group carry byte-identical answers[]
    # content (three paraphrasings of one question pointing at the same
    # gold span/s), so summing len(answers) over *all* 1521 rows triple-
    # counts to 1953 = 651 x 3 registers. One representative row per group
    # (registers verified elsewhere to be structurally identical in count)
    # gives the README-comparable figure; the raw per-row-instance total is
    # kept separately for transparency, not used in the cardinality check.
    first_row_per_group: dict[str, dict] = {}
    for r in qa_rows:
        first_row_per_group.setdefault(r["intent_group_id"], r)
    n_gold_spans_per_group = sum(len(r.get("answers", [])) for r in first_row_per_group.values())
    n_gold_span_answer_instances = sum(len(r.get("answers", [])) for r in qa_rows)
    registers = {r["question_register"] for r in qa_rows}
    counts = {
        "query_rows": len(qa_rows),
        "unique_qa_id": len(set(qa_ids)),
        "intent_groups": len(intent_groups),
        "documents_represented": len(doc_ids),
        "categories_represented": len(cat_ids),
        "gold_spans": n_gold_spans_per_group,
        "gold_span_answer_instances_all_registers": n_gold_span_answer_instances,
        "question_registers": sorted(registers),
    }
    result.add("release", "counts", "INFO", counts)

    # Verify the assumption the corrected gold_spans count above relies on:
    # every register within a group carries the same number of answers as
    # every other register in that group.
    group_answer_counts: dict[str, set[int]] = {}
    for r in qa_rows:
        group_answer_counts.setdefault(r["intent_group_id"], set()).add(len(r.get("answers", [])))
    uneven_groups = [g for g, counts_set in group_answer_counts.items() if len(counts_set) > 1]
    result.add("release", "every register within a group carries the same answer count",
                "PASS" if not uneven_groups else "FAIL",
                {"n_uneven_groups": len(uneven_groups), "examples": uneven_groups[:10]})
    result.add("release", "qa_id uniqueness", "PASS" if counts["unique_qa_id"] == counts["query_rows"] else "FAIL",
                counts)

    # ---- 6. cardinalities vs README expected --------------------------------
    exp = cfg.expected
    cardinality_checks = [
        ("query_rows", counts["query_rows"], exp["query_rows"]),
        ("intent_groups", counts["intent_groups"], exp["intent_groups"]),
        ("documents_represented", counts["documents_represented"], exp["core_documents"]),
        ("categories_represented", counts["categories_represented"], exp["categories"]),
        ("gold_spans", counts["gold_spans"], exp["gold_spans"]),
    ]
    discrepancies = [{"metric": m, "observed": o, "expected": e} for m, o, e in cardinality_checks if o != e]
    result.add("release", "cardinalities vs README expected", "PASS" if not discrepancies else "FAIL",
                discrepancies or "all match")

    # ---- 7/8/9. per-answer offset + paragraph checks -------------------------
    dataset = dataset or load_dataset(cfg)
    missing_normalized_docs = sorted({d for d in doc_ids if d not in dataset.documents})
    result.add("corpus", "every row doc_id has a core document record", "PASS" if not missing_normalized_docs else "FAIL",
                missing_normalized_docs)

    offset_failures: list[dict[str, Any]] = []
    missing_gold_paragraphs: list[dict[str, Any]] = []
    doc_text_cache: dict[str, str] = {}
    for row in qa_rows:
        doc_id = row["doc_id"]
        if doc_id not in doc_text_cache:
            path = cfg.dataset.normalized_dir / f"{doc_id}.txt"
            doc_text_cache[doc_id] = path.read_text(encoding="utf-8") if path.is_file() else None
        text = doc_text_cache[doc_id]
        if text is None:
            offset_failures.append({"qa_id": row["qa_id"], "doc_id": doc_id, "reason": "normalized text file missing"})
            continue
        for i, ans in enumerate(row.get("answers", [])):
            expected_text = ans.get("text", ans.get("answer_text"))
            start, end = ans.get("answer_start"), ans.get("answer_end")
            if start is None or end is None:
                offset_failures.append({"qa_id": row["qa_id"], "answer_index": i, "reason": "missing offsets"})
                continue
            sliced = text[start:end]
            if sliced != expected_text:
                offset_failures.append({
                    "qa_id": row["qa_id"], "answer_index": i, "reason": "slice mismatch",
                    "expected": expected_text, "sliced": sliced, "answer_start": start, "answer_end": end,
                })
        for pid in row.get("gold_paragraph_ids", []):
            if pid not in dataset.paragraphs:
                missing_gold_paragraphs.append({"qa_id": row["qa_id"], "paragraph_id": pid})

    result.add("gold_spans", "answer_start:answer_end slices frozen text exactly", "PASS" if not offset_failures else "FAIL",
                {"n_failures": len(offset_failures), "examples": offset_failures[:10]})
    result.add("gold_spans", "gold_paragraph_ids resolve in paragraph metadata", "PASS" if not missing_gold_paragraphs else "FAIL",
                {"n_failures": len(missing_gold_paragraphs), "examples": missing_gold_paragraphs[:10]})

    # ---- 10/11/12/13. distractors schema + join -----------------------------
    if not cfg.dataset.distractors.is_file():
        result.add("distractors", "qa/distractors.json exists", "FAIL", "file not found")
    else:
        distractor_records = read_json(cfg.dataset.distractors)
        d_field_sets = {frozenset(r.keys()) for r in distractor_records}
        slot_field_sets = {frozenset(s.keys()) for r in distractor_records for s in r.get("distractors", [])}
        result.add("distractors", "schema discovery", "INFO", {
            "n_records": len(distractor_records),
            "record_field_sets": [sorted(fs) for fs in d_field_sets],
            "slot_field_sets": [sorted(fs) for fs in slot_field_sets],
        })

        n_slots = sum(len(r.get("distractors", [])) for r in distractor_records)
        result.add("distractors", "slot count vs README expected", "PASS" if n_slots == exp["distractor_slots"] else "FAIL",
                    {"observed": n_slots, "expected": exp["distractor_slots"]})

        by_qa_id = dataset.distractor_by_qa_id
        join_missing: list[str] = []
        join_malformed: list[dict[str, Any]] = []
        join_multiple: list[dict[str, Any]] = []
        equal_to_gold: list[dict[str, Any]] = []
        source_span_bad: list[dict[str, Any]] = []

        for row in qa_rows:
            recs_for_row = by_qa_id.get(row["qa_id"], [])
            for i, ans in enumerate(row.get("answers", [])):
                matching = [r for r in recs_for_row if r.get("answer_index") == i]
                if not matching:
                    join_missing.append(f"{row['qa_id']}#{i}")
                elif len(matching) > 1:
                    join_multiple.append({"qa_id": row["qa_id"], "answer_index": i, "n_matches": len(matching)})
                else:
                    rec = matching[0]
                    slots = rec.get("distractors", [])
                    if len(slots) != 3:
                        join_malformed.append({"qa_id": row["qa_id"], "answer_index": i, "n_slots": len(slots)})
                    gold_norm = _normalize_for_compare(ans.get("text", ""))
                    for slot in slots:
                        if _normalize_for_compare(slot.get("text", "")) == gold_norm:
                            equal_to_gold.append({"qa_id": row["qa_id"], "answer_index": i, "distractor_text": slot.get("text")})
                        if slot.get("source") == "mined":
                            from_doc = slot.get("from_doc_id")
                            from_par = slot.get("from_paragraph_id")
                            if from_doc and from_doc not in dataset.documents:
                                source_span_bad.append({"qa_id": row["qa_id"], "reason": "from_doc_id not a core document", "from_doc_id": from_doc})
                            if from_par and from_par not in dataset.paragraphs:
                                source_span_bad.append({"qa_id": row["qa_id"], "reason": "from_paragraph_id not found", "from_paragraph_id": from_par})

        result.add("distractors", "every gold answer joins exactly one distractor record with 3 slots",
                    "PASS" if not (join_missing or join_malformed or join_multiple) else "FAIL",
                    {"missing": join_missing[:10], "n_missing": len(join_missing),
                     "malformed": join_malformed[:10], "n_malformed": len(join_malformed),
                     "multiple": join_multiple[:10], "n_multiple": len(join_multiple)})
        result.add("distractors", "no distractor equals its gold answer after normalization",
                    "PASS" if not equal_to_gold else "FAIL",
                    {"n_failures": len(equal_to_gold), "examples": equal_to_gold[:10]})
        result.add("distractors", "mined distractor source doc/paragraph resolves", "PASS" if not source_span_bad else "WARN",
                    {"n_failures": len(source_span_bad), "examples": source_span_bad[:10]})

    # ---- 14/15. document length distribution --------------------------------
    doc_lengths = []
    for doc_id, text in doc_text_cache.items():
        if text is not None:
            doc_lengths.append({"doc_id": doc_id, "chars": len(text), "estimated_tokens": _estimate_tokens(text)})
    doc_lengths.sort(key=lambda d: d["chars"])
    total_chars = sum(d["chars"] for d in doc_lengths)
    result.add("corpus", "document length distribution", "INFO", {
        "n_documents": len(doc_lengths),
        "total_chars": total_chars,
        "min_chars": doc_lengths[0]["chars"] if doc_lengths else None,
        "max_chars": doc_lengths[-1]["chars"] if doc_lengths else None,
        "median_chars": doc_lengths[len(doc_lengths) // 2]["chars"] if doc_lengths else None,
        "note": "estimated_tokens is a whitespace-split approximation, not a model tokenizer count",
        "per_document": doc_lengths,
    })
    result.add("corpus", "total core corpus characters vs README expected",
                "PASS" if total_chars == exp["core_characters"] else "FAIL",
                {"observed": total_chars, "expected": exp["core_characters"]})

    # ---- 16. read-only invocation of the dataset's own reproduce.py --------
    reproduce_path = cfg.dataset.root / "scripts" / "reproduce.py"
    if reproduce_path.is_file():
        try:
            proc = subprocess.run(
                [sys.executable, str(reproduce_path), "--verify"],
                cwd=str(cfg.dataset.root), capture_output=True, text=True, timeout=120,
            )
            result.add("upstream_reproduce", "scripts/reproduce.py --verify (read-only)",
                        "PASS" if proc.returncode == 0 else "FAIL",
                        {"returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-2000:]})
        except Exception as exc:  # noqa: BLE001
            result.add("upstream_reproduce", "scripts/reproduce.py --verify (read-only)", "FAIL", str(exc))
    else:
        result.add("upstream_reproduce", "scripts/reproduce.py --verify (read-only)", "WARN", "reproduce.py not found")

    # ---- 17. never-touch integrity ------------------------------------------
    post_hashes = _hash_never_touch(cfg)
    changed = [rel for rel in NEVER_TOUCH_FILES if pre_hashes.get(rel) != post_hashes.get(rel)]
    result.add("integrity", "no released dataset file was modified by this validation run",
                "PASS" if not changed else "FAIL", {"changed": changed})

    result.finished_at = utc_now()
    return result


def render_markdown(result: ValidationResult) -> str:
    lines = [
        "# IndiCUAD V2 — Dataset Validation Report",
        "",
        f"Generated: {result.finished_at}",
        f"Overall status: **{result.overall_status}**  ({result.n_fail} FAIL, {result.n_warn} WARN, "
        f"{sum(1 for c in result.checks if c.status == 'PASS')} PASS, "
        f"{sum(1 for c in result.checks if c.status == 'INFO')} INFO)",
        "",
    ]
    sections: dict[str, list[Check]] = {}
    for c in result.checks:
        sections.setdefault(c.section, []).append(c)
    for section, checks in sections.items():
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| Status | Check | Detail |")
        lines.append("|---|---|---|")
        for c in checks:
            detail_str = str(c.detail)
            if len(detail_str) > 300:
                detail_str = detail_str[:300] + "…"
            detail_str = detail_str.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {c.status} | {c.name} | {detail_str} |")
        lines.append("")
    return "\n".join(lines)


def write_reports(cfg: Config, result: ValidationResult) -> tuple[Path, Path]:
    reports_dir = cfg.results_dir("reports")
    json_path = reports_dir / "dataset_validation.json"
    md_path = reports_dir / "dataset_validation.md"
    write_json(json_path, {
        "overall_status": result.overall_status,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "n_fail": result.n_fail,
        "n_warn": result.n_warn,
        "checks": [c.to_dict() for c in result.checks],
    })
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(result))
    return json_path, md_path
