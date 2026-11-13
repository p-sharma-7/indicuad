"""Deterministic validators for the IndiCUAD corpus.

Every check here is deterministic and evidence-producing: it either passes,
or it names the exact records that failed. No check ever repairs semantic
content, and none of them touch gold annotations.

Findings carry an error class from the preparation spec §22:

    AUTO_FIX            deterministic, safely repairable
    NEEDS_HUMAN_REVIEW  semantic or legal judgement required
    FATAL               stop the pipeline

Artifacts that do not exist yet report status NOT_PRESENT rather than
failing, so the suite is meaningful at every stage of the build.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from indicuad_common import (
    ANNOTATIONS_DIR,
    CORPUS_DIR,
    NORMALIZED_DIR,
    QA_DIR,
    current_annotations_path,
    read_jsonl,
    read_manifest,
)
# Single source of truth: the same pattern the segmenter uses to tell an
# enumerated clause from running furniture.
from segment_paragraphs import ENUMERATED, OPERATIVE

INFO = "INFO"
AUTO_FIX = "AUTO_FIX"
NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
FATAL = "FATAL"

# The one category whose gold may sit in running furniture: a document's
# name is part of its masthead. Kept in step with resolve_spans.
MASTHEAD_CATEGORY = "CAT-001"

REGISTERS = ["canonical", "semi_formal", "naive_layperson"]


class Findings:
    """Accumulates classified findings for one validator."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, error_class: str, code: str, message: str, **context) -> None:
        self.items.append(
            {"error_class": error_class, "code": code, "message": message, **context}
        )

    @property
    def fatal(self) -> list[dict]:
        return [i for i in self.items if i["error_class"] == FATAL]

    @property
    def auto_fixable(self) -> list[dict]:
        return [i for i in self.items if i["error_class"] == AUTO_FIX]

    @property
    def human(self) -> list[dict]:
        return [i for i in self.items if i["error_class"] == NEEDS_HUMAN_REVIEW]

    def status(self) -> str:
        if self.fatal:
            return "FAIL"
        if self.auto_fixable:
            return "FAIL"
        if self.human:
            return "NEEDS_HUMAN_REVIEW"
        return "PASS"

    def report(self, **extra) -> dict:
        return {
            "status": self.status(),
            "finding_counts": {
                FATAL: len(self.fatal),
                AUTO_FIX: len(self.auto_fixable),
                NEEDS_HUMAN_REVIEW: len(self.human),
            },
            "findings": self.items[:500],
            "findings_truncated": len(self.items) > 500,
            **extra,
        }


def _load():
    return {
        "documents": read_jsonl(CORPUS_DIR / "documents.jsonl"),
        "paragraphs": read_jsonl(CORPUS_DIR / "paragraphs.jsonl"),
        "categories": read_jsonl(CORPUS_DIR / "categories.jsonl"),
        "annotations": read_jsonl(current_annotations_path()),
        "qa": read_jsonl(QA_DIR / "qa.jsonl"),
    }


# ---------------------------------------------------------------------------
# A. Schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "documents": ["doc_id", "filename", "source_url", "source_entity",
                  "license_notes", "text_sha256", "page_count"],
    "paragraphs": ["paragraph_id", "doc_id", "paragraph_index", "char_start",
                   "char_end", "text", "is_boilerplate"],
    "categories": ["category_id", "category_name", "description",
                   "canonical_question", "semi_formal_question",
                   "naive_layperson_question", "boundary_rule"],
}

ID_FIELD = {"documents": "doc_id", "paragraphs": "paragraph_id",
            "categories": "category_id"}


def validate_schema(data=None) -> dict:
    data = data or _load()
    findings = Findings()
    counts = {}

    for name in ("documents", "paragraphs", "categories"):
        records = data[name]
        counts[name] = len(records)
        if not records:
            findings.add(FATAL, "MISSING_ARTIFACT", f"corpus/{name}.jsonl is empty or absent")
            continue

        for index, record in enumerate(records):
            for field in REQUIRED_FIELDS[name]:
                if field not in record:
                    findings.add(
                        AUTO_FIX, "MISSING_FIELD",
                        f"{name}[{index}] missing required field {field!r}",
                        artifact=name, record_index=index, field=field,
                    )

        id_field = ID_FIELD[name]
        ids = [r.get(id_field) for r in records]
        for value, count in Counter(ids).items():
            if count > 1:
                findings.add(
                    AUTO_FIX, "DUPLICATE_ID",
                    f"{name}: {id_field}={value!r} appears {count} times",
                    artifact=name, id=value, occurrences=count,
                )

    for paragraph in data["paragraphs"]:
        for field in ("char_start", "char_end", "paragraph_index"):
            if not isinstance(paragraph.get(field), int):
                findings.add(
                    AUTO_FIX, "INVALID_TYPE",
                    f"{paragraph.get('paragraph_id')}: {field} is not an integer",
                    paragraph_id=paragraph.get("paragraph_id"), field=field,
                )

    if len(data["categories"]) not in (0, 26):
        findings.add(
            NEEDS_HUMAN_REVIEW, "CATEGORY_COUNT",
            f"protocol defines 26 categories, found {len(data['categories'])}",
        )

    return findings.report(record_counts=counts)


# ---------------------------------------------------------------------------
# B. Source metadata preservation
# ---------------------------------------------------------------------------

def validate_source(data=None) -> dict:
    data = data or _load()
    findings = Findings()
    manifest = {r["filename"]: r for r in read_manifest()}

    for document in data["documents"]:
        row = manifest.get(document["filename"])
        if row is None:
            findings.add(
                FATAL, "SOURCE_SUBSTITUTION",
                f"{document['doc_id']}: filename not present in the source manifest",
                doc_id=document["doc_id"], filename=document["filename"],
            )
            continue
        for field in ("source_url", "source_entity", "license_notes"):
            if document.get(field) != row[field]:
                findings.add(
                    AUTO_FIX, "METADATA_DRIFT",
                    f"{document['doc_id']}: {field} differs from the manifest",
                    doc_id=document["doc_id"], field=field,
                    manifest_value=row[field], corpus_value=document.get(field),
                )

    covered = {d["filename"] for d in data["documents"]}
    for filename in manifest:
        if filename not in covered:
            findings.add(
                FATAL, "MISSING_DOCUMENT",
                f"manifest row {filename!r} has no normalized document",
                filename=filename,
            )

    return findings.report(
        manifest_rows=len(manifest), documents=len(data["documents"])
    )


# ---------------------------------------------------------------------------
# C. Paragraphs
# ---------------------------------------------------------------------------

def validate_paragraphs(data=None) -> dict:
    data = data or _load()
    findings = Findings()
    doc_ids = {d["doc_id"] for d in data["documents"]}

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for paragraph in data["paragraphs"]:
        by_doc[paragraph["doc_id"]].append(paragraph)

    slice_checked = 0
    for doc_id, paragraphs in by_doc.items():
        if doc_id not in doc_ids:
            findings.add(
                FATAL, "ORPHAN_PARAGRAPH",
                f"paragraphs reference unknown doc_id {doc_id!r}", doc_id=doc_id,
            )
            continue

        text_path = NORMALIZED_DIR / f"{doc_id}.txt"
        if not text_path.is_file():
            findings.add(
                FATAL, "MISSING_SOURCE_TEXT",
                f"{doc_id}: normalized text missing; offsets cannot be verified",
                doc_id=doc_id,
            )
            continue
        text = text_path.read_text(encoding="utf-8")

        paragraphs.sort(key=lambda p: p["paragraph_index"])
        previous_end = -1
        for paragraph in paragraphs:
            pid = paragraph["paragraph_id"]
            start, end = paragraph["char_start"], paragraph["char_end"]

            if start < 0 or end > len(text) or end <= start:
                findings.add(
                    AUTO_FIX, "INVALID_OFFSETS",
                    f"{pid}: offsets [{start},{end}) invalid for a {len(text)}-char document",
                    paragraph_id=pid,
                )
                continue

            if text[start:end] != paragraph["text"]:
                findings.add(
                    FATAL, "PARAGRAPH_TEXT_DRIFT",
                    f"{pid}: text does not match source[{start}:{end}] — "
                    "the frozen text or the segmentation changed",
                    paragraph_id=pid,
                )
            slice_checked += 1

            if start < previous_end:
                findings.add(
                    AUTO_FIX, "OVERLAPPING_PARAGRAPHS",
                    f"{pid}: starts at {start}, before previous paragraph ended at {previous_end}",
                    paragraph_id=pid,
                )
            previous_end = end

        indices = [p["paragraph_index"] for p in paragraphs]
        if indices != list(range(len(indices))):
            findings.add(
                AUTO_FIX, "NON_CONTIGUOUS_INDEX",
                f"{doc_id}: paragraph_index is not 0..n-1", doc_id=doc_id,
            )

    return findings.report(
        paragraphs=len(data["paragraphs"]),
        documents_covered=len(by_doc),
        slice_exactness_checked=slice_checked,
    )


# ---------------------------------------------------------------------------
# D/E. Spans and negatives
# ---------------------------------------------------------------------------

def validate_spans(data=None) -> dict:
    data = data or _load()
    findings = Findings()

    cells = data["annotations"] or data["qa"]
    if not cells:
        return {
            "status": "NOT_PRESENT",
            "note": "no annotations or qa rows yet; span validation is vacuous",
            "findings": [], "finding_counts": {},
        }

    paragraphs = {p["paragraph_id"]: p for p in data["paragraphs"]}
    texts: dict[str, str] = {}
    for document in data["documents"]:
        path = NORMALIZED_DIR / f"{document['doc_id']}.txt"
        if path.is_file():
            texts[document["doc_id"]] = path.read_text(encoding="utf-8")

    spans_checked = 0
    unannotated = 0
    for cell in cells:
        ident = cell.get("qa_id") or f"{cell.get('doc_id')}x{cell.get('category_id')}"
        answers = cell.get("answers") or []
        gold_ids = cell.get("gold_paragraph_ids") or []

        # is_impossible is tri-state: null means the cell has no verdict yet.
        # That is the normal state of a scaffolded workspace, not a defect —
        # treating null as false would report every empty cell as a broken
        # positive. build_qa.py is what refuses to ship null cells.
        if cell.get("is_impossible") is None:
            unannotated += 1
            continue

        # E. Negatives must be empty on both sides.
        if cell.get("is_impossible"):
            if answers:
                findings.add(
                    AUTO_FIX, "NEGATIVE_WITH_ANSWERS",
                    f"{ident}: is_impossible=true but answers is non-empty", cell=ident,
                )
            if gold_ids:
                findings.add(
                    AUTO_FIX, "NEGATIVE_WITH_GOLD_PARAGRAPHS",
                    f"{ident}: is_impossible=true but gold_paragraph_ids is non-empty",
                    cell=ident,
                )
            continue

        if not answers:
            findings.add(
                NEEDS_HUMAN_REVIEW, "POSITIVE_WITHOUT_ANSWERS",
                f"{ident}: is_impossible=false but no answer spans given", cell=ident,
            )
            continue

        text = texts.get(cell.get("doc_id", ""))
        if text is None:
            findings.add(
                FATAL, "MISSING_SOURCE_TEXT",
                f"{ident}: no normalized text for {cell.get('doc_id')}", cell=ident,
            )
            continue

        for position, answer in enumerate(answers):
            start, end = answer.get("answer_start"), answer.get("answer_end")

            if not isinstance(start, int) or not isinstance(end, int):
                findings.add(
                    AUTO_FIX, "INVALID_OFFSET_TYPE",
                    f"{ident}[{position}]: answer offsets are not integers", cell=ident,
                )
                continue
            if end < start:
                findings.add(
                    AUTO_FIX, "INVERTED_OFFSETS",
                    f"{ident}[{position}]: answer_end {end} < answer_start {start}",
                    cell=ident,
                )
                continue
            if start < 0 or end > len(text):
                findings.add(
                    AUTO_FIX, "OUT_OF_RANGE_OFFSETS",
                    f"{ident}[{position}]: [{start},{end}) outside a {len(text)}-char document",
                    cell=ident,
                )
                continue

            # The load-bearing invariant. No fuzzy matching, ever.
            if text[start:end] != answer.get("text"):
                findings.add(
                    FATAL, "SPAN_TEXT_MISMATCH",
                    f"{ident}[{position}]: source[{start}:{end}] != answer.text verbatim",
                    cell=ident,
                    source_slice=text[start:end][:160],
                    answer_text=(answer.get("text") or "")[:160],
                )
            spans_checked += 1

            pid = answer.get("paragraph_id")
            if pid is None:
                findings.add(
                    AUTO_FIX, "MISSING_PARAGRAPH_REF",
                    f"{ident}[{position}]: answer has no paragraph_id", cell=ident,
                )
            elif pid not in paragraphs:
                findings.add(
                    AUTO_FIX, "UNKNOWN_PARAGRAPH_REF",
                    f"{ident}[{position}]: paragraph_id {pid!r} does not exist", cell=ident,
                )
            else:
                paragraph = paragraphs[pid]
                if not (paragraph["char_start"] <= start and end <= paragraph["char_end"]):
                    findings.add(
                        NEEDS_HUMAN_REVIEW, "SPAN_OUTSIDE_PARAGRAPH",
                        f"{ident}[{position}]: span [{start},{end}) is not contained in "
                        f"{pid} [{paragraph['char_start']},{paragraph['char_end']})",
                        cell=ident, paragraph_id=pid,
                    )

        for pid in gold_ids:
            if pid not in paragraphs:
                findings.add(
                    AUTO_FIX, "UNKNOWN_GOLD_PARAGRAPH",
                    f"{ident}: gold_paragraph_ids references unknown {pid!r}", cell=ident,
                )

    return findings.report(
        cells_total=len(cells),
        cells_with_verdict=len(cells) - unannotated,
        cells_unannotated=unannotated,
        spans_checked=spans_checked,
    )


# ---------------------------------------------------------------------------
# F. Query register consistency
# ---------------------------------------------------------------------------

def validate_queries(data=None) -> dict:
    data = data or _load()
    findings = Findings()
    qa = data["qa"]
    if not qa:
        return {
            "status": "NOT_PRESENT",
            "note": "qa/qa.jsonl not built yet",
            "findings": [], "finding_counts": {},
        }

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in qa:
        key = row.get("intent_group_id") or f"{row.get('doc_id')}x{row.get('category_id')}"
        groups[key].append(row)

    for qa_id, count in Counter(r.get("qa_id") for r in qa).items():
        if count > 1:
            findings.add(AUTO_FIX, "DUPLICATE_QA_ID",
                         f"qa_id {qa_id!r} appears {count} times", qa_id=qa_id)

    for group_id, rows in groups.items():
        registers = [r.get("question_register") for r in rows]

        for register, count in Counter(registers).items():
            if count > 1:
                findings.add(
                    AUTO_FIX, "DUPLICATE_REGISTER",
                    f"{group_id}: register {register!r} appears {count} times",
                    intent_group_id=group_id,
                )

        unknown = [r for r in registers if r not in REGISTERS]
        if unknown:
            findings.add(
                AUTO_FIX, "UNKNOWN_REGISTER",
                f"{group_id}: unrecognised register(s) {unknown}",
                intent_group_id=group_id,
            )

        # All registers of one cell must carry the same gold answer.
        def gold_key(row: dict):
            return (
                bool(row.get("is_impossible")),
                tuple(sorted(row.get("gold_paragraph_ids") or [])),
                tuple(sorted(
                    (a.get("answer_start"), a.get("answer_end"), a.get("text"))
                    for a in (row.get("answers") or [])
                )),
            )

        distinct = {gold_key(r) for r in rows}
        if len(distinct) > 1:
            findings.add(
                FATAL, "REGISTER_GOLD_DIVERGENCE",
                f"{group_id}: registers disagree on the gold answer — "
                "all registers of one cell must share it",
                intent_group_id=group_id,
                registers=registers,
            )

        if len({r.get("doc_id") for r in rows}) > 1 or len({r.get("category_id") for r in rows}) > 1:
            findings.add(
                FATAL, "INTENT_GROUP_SPANS_CELLS",
                f"{group_id}: one intent group covers more than one (doc, category) cell",
                intent_group_id=group_id,
            )

    questions = Counter(
        (r.get("doc_id"), r.get("category_id"), (r.get("question") or "").strip().lower())
        for r in qa
    )
    for key, count in questions.items():
        if count > 1:
            findings.add(
                AUTO_FIX, "DUPLICATE_QUESTION",
                f"identical question text repeated {count}x within cell {key[0]}x{key[1]}",
                doc_id=key[0], category_id=key[1],
            )

    return findings.report(qa_rows=len(qa), intent_groups=len(groups))


# ---------------------------------------------------------------------------
# G. Split leakage
# ---------------------------------------------------------------------------

def validate_leakage(data=None) -> dict:
    data = data or _load()
    findings = Findings()

    doc_split = {d["doc_id"]: d.get("split") for d in data["documents"]}
    unassigned = [d for d, s in doc_split.items() if s in (None, "")]

    observed: dict[str, set] = defaultdict(set)
    for row in data["qa"]:
        if row.get("split"):
            observed[row["doc_id"]].add(row["split"])

    for doc_id, splits in observed.items():
        if len(splits) > 1:
            findings.add(
                FATAL, "SPLIT_LEAKAGE",
                f"{doc_id}: appears in more than one split {sorted(splits)} — "
                "splits are assigned by document",
                doc_id=doc_id,
            )
        declared = doc_split.get(doc_id)
        if declared and splits and declared not in splits:
            findings.add(
                AUTO_FIX, "SPLIT_MISMATCH",
                f"{doc_id}: documents.jsonl says {declared!r} but qa rows say {sorted(splits)}",
                doc_id=doc_id,
            )

    status_extra = {}
    if unassigned:
        findings.add(
            NEEDS_HUMAN_REVIEW, "SPLIT_UNASSIGNED",
            f"{len(unassigned)} document(s) have no split yet — triage is pending human review",
            documents=unassigned[:50],
        )
        status_extra["unassigned_documents"] = len(unassigned)

    return findings.report(**status_extra)


# ---------------------------------------------------------------------------
# H. Duplicates
# ---------------------------------------------------------------------------

def validate_duplicates(data=None) -> dict:
    data = data or _load()
    findings = Findings()

    # Identical whole documents are a collection error worth knowing about.
    by_hash: dict[str, list[str]] = defaultdict(list)
    for document in data["documents"]:
        by_hash[document["text_sha256"]].append(document["doc_id"])
    duplicate_documents = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    for digest, ids in duplicate_documents.items():
        findings.add(
            NEEDS_HUMAN_REVIEW, "DUPLICATE_DOCUMENT",
            f"documents {ids} have byte-identical normalized text",
            doc_ids=ids, text_sha256=digest,
        )

    # Repeated paragraphs are EXPECTED (running headers, template signature
    # blocks, clauses repeated in an annexure — protocol §6 rule 8 requires
    # annotating every occurrence). Reported as information, not a defect.
    text_counts: dict[str, list[str]] = defaultdict(list)
    for paragraph in data["paragraphs"]:
        text_counts[paragraph["text"].strip()].append(paragraph["paragraph_id"])
    repeated = {t: ids for t, ids in text_counts.items() if len(ids) > 1}

    cross_document = 0
    for ids in repeated.values():
        if len({i.split("-")[1] for i in ids}) > 1:
            cross_document += 1

    # A repeated block that is NOT flagged boilerplate and NOT an enumerated
    # clause is furniture the segmenter failed to catch — most likely folded
    # onto a clause body, which is the defect this flag exists to prevent.
    boilerplate_ids = {
        p["paragraph_id"] for p in data["paragraphs"] if p.get("is_boilerplate")
    }
    by_id = {p["paragraph_id"]: p for p in data["paragraphs"]}
    # Superseded paragraphs are the parent's copy of an agreement that was
    # split into its own document. They are repeated BY CONSTRUCTION, are
    # already held out of the retrieval index, and flagging them as furniture
    # would be wrong — they are clause text, not letterhead.
    superseded_ids = {
        p["paragraph_id"] for p in data["paragraphs"] if p.get("is_superseded")
    }
    unflagged = []
    shared_clauses = []
    for ids in repeated.values():
        # Discount copies that are already accounted for. If what remains is a
        # single paragraph, it is the canonical one and the repeat is fully
        # explained — a split-out agreement leaves exactly this shape: the
        # child's clause plus the parent's superseded copy of it.
        live = [
            pid for pid in ids
            if pid not in boilerplate_ids and pid not in superseded_ids
        ]
        if len(live) <= 1:
            continue
        for pid in live:
            text = by_id[pid]["text"]
            # A repeated block carrying an obligation is shared CLAUSE text,
            # not furniture the segmenter missed. Sibling general-conditions
            # documents from one PSU reuse whole clauses verbatim; that is a
            # property of the corpus and is reported below as a count, not
            # raised as a defect. Furniture never says "shall".
            if OPERATIVE.search(text):
                shared_clauses.append(pid)
                continue
            if not ENUMERATED.match(text):
                unflagged.append(pid)
    if unflagged:
        findings.add(
            NEEDS_HUMAN_REVIEW, "UNFLAGGED_REPEATED_BLOCK",
            f"{len(unflagged)} repeated non-enumerated paragraph(s) are not flagged "
            f"is_boilerplate: {sorted(unflagged)[:10]}",
            paragraph_ids=sorted(unflagged)[:50],
        )

    # Gold spans must never point into template furniture — with one
    # exception, CAT-001. A document's name is printed in its masthead, and a
    # masthead repeated across sibling templates (seven PSU tenders each
    # headed GENERAL CONDITIONS OF CONTRACT) is classified as furniture by a
    # rule that is right about every other block. Refusing the span there
    # would force CAT-001 to be recorded empty for documents whose title is
    # legible on page one. The exception is granted in
    # resolve_spans.build_answer and is mirrored here so the two agree; it is
    # scoped to that one category and to nothing else.
    cells = data["annotations"] or data["qa"]
    gold_in_boilerplate = sorted(
        {
            pid
            for cell in cells
            if cell.get("category_id") != MASTHEAD_CATEGORY
            for pid in cell.get("gold_paragraph_ids") or []
            if pid in boilerplate_ids
        }
    )
    masthead_in_boilerplate = sorted(
        {
            pid
            for cell in cells
            if cell.get("category_id") == MASTHEAD_CATEGORY
            for pid in cell.get("gold_paragraph_ids") or []
            if pid in boilerplate_ids
        }
    )
    if gold_in_boilerplate:
        findings.add(
            FATAL, "GOLD_SPAN_IN_BOILERPLATE",
            f"{len(gold_in_boilerplate)} gold paragraph(s) are flagged boilerplate: "
            f"{gold_in_boilerplate[:10]}",
            paragraph_ids=gold_in_boilerplate[:50],
        )

    if masthead_in_boilerplate:
        findings.add(
            INFO, "MASTHEAD_GOLD_IN_BOILERPLATE",
            f"{len(masthead_in_boilerplate)} CAT-001 gold paragraph(s) sit in furniture "
            "under the masthead exception — a title repeated across sibling templates: "
            f"{masthead_in_boilerplate[:10]}",
            paragraph_ids=masthead_in_boilerplate[:50],
        )

    return findings.report(
        duplicate_documents=len(duplicate_documents),
        masthead_gold_in_boilerplate=len(masthead_in_boilerplate),
        repeated_paragraph_groups=len(repeated),
        repeated_paragraph_records=sum(len(v) for v in repeated.values()),
        repeated_groups_spanning_documents=cross_document,
        boilerplate_paragraphs=len(boilerplate_ids),
        unflagged_repeated_blocks=len(unflagged),
        shared_clause_paragraphs=len(shared_clauses),
        gold_spans_in_boilerplate=len(gold_in_boilerplate),
        note=(
            "Repeated paragraphs are expected in this corpus and are reported, not "
            "removed: protocol §6 rule 8 requires every occurrence of a duplicated "
            "CLAUSE to be annotated. Running furniture (letterhead, page headers, "
            "disclaimers, signature blocks) is not a clause, is flagged "
            "is_boilerplate by the segmenter, and is excluded from the retrieval "
            "index, the P2 sweep and gold spans."
        ),
    )


ALL_VALIDATORS = {
    "schema_report.json": validate_schema,
    "source_metadata_report.json": validate_source,
    "paragraph_integrity_report.json": validate_paragraphs,
    "span_report.json": validate_spans,
    "query_consistency_report.json": validate_queries,
    "split_leakage_report.json": validate_leakage,
    "duplicate_report.json": validate_duplicates,
}
