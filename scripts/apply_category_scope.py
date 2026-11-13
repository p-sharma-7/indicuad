"""Restrict each category to the instrument types it can apply to (§A-4).

Asking "what is the warranty duration?" of a power of attorney is not a
test of anything. The instrument type makes the question inapplicable, not
merely unanswered, and counting it as an abstention case inflates the
negative class with questions no annotator or system should be asked.

This adds two fields and marks each core cell applicable or not:

    documents.jsonl   doc_type              one of DOC_TYPES
    categories.jsonl  applies_to_doc_types  list of doc_type values
    annotations       applicable            bool

**Cells are marked, never deleted.** The full 26-category grid stays in
the annotations file so the decision is auditable and reversible; only
`build_qa.py` skips inapplicable cells when emitting query rows.

**The negatives split keeps all 26 categories.** Those documents exist to
test whether a system invents clauses that are not there, so their
unanswerable questions are the instrument, not noise. Scoping them would
remove the measurement.

Applicability is derived from each category's own boundary rule where it
states one — CAT-018 already says "Employment documents only" — and
otherwise from whether an instrument of that type can carry the clause at
all. It is a legal-ontology judgement and was NOT reviewed by counsel; the
§7 review was declined. Treat it as revisable.
"""

from __future__ import annotations

import sys

from indicuad_common import (
    CORPUS_DIR,
    Stage,
    current_annotations_path,
    emit_report,
    read_jsonl,
    write_jsonl,
)

# Instrument families present in the corpus.
DOC_TYPES = [
    "employment",      # employment, founder, hiring
    "services",        # consultancy, agency, SLA, service agreements
    "confidentiality",  # NDA, confidentiality & IP assignment
    "supply_works",    # procurement, works, O&M, net metering, GeM/PWD terms
    "property",        # lease, leave & license
    "finance_security",  # guarantee, bank guarantee, indemnity, hire-purchase
    "entity",          # partnership deed
    "agency_authority",  # power of attorney
    "disclosure",      # Reg-30 filings (negatives split)
]

DOC_TYPE = {
    "DOC-015": "employment",       "DOC-049": "employment",
    "DOC-050": "employment",
    "DOC-051": "services",         "DOC-054": "services",
    "DOC-057": "services",
    "DOC-047": "confidentiality",  "DOC-048": "confidentiality",
    "DOC-060": "confidentiality",
    "DOC-044": "supply_works",     "DOC-045": "supply_works",
    "DOC-052": "supply_works",     "DOC-053": "supply_works",
    "DOC-018": "property",           "DOC-062": "property",
    "DOC-019": "finance_security", "DOC-020": "finance_security",
    "DOC-022": "finance_security", "DOC-017": "finance_security",
    "DOC-016": "entity",
    "DOC-023": "agency_authority",
    # PSU and institutional general conditions of contract. Those written for
    # works and for goods are supply_works; those written for services are
    # services, following each document's own heading.
    "DOC-064": "supply_works",     "DOC-065": "services",
    "DOC-067": "supply_works",     "DOC-068": "supply_works",
    "DOC-069": "supply_works",     "DOC-070": "services",
    "DOC-081": "supply_works",
    # Power offtake. A PPA and a PSA sell a commodity against a delivery
    # obligation, which is the supply_works family rather than services.
    "DOC-071": "supply_works",     "DOC-072": "supply_works",
    "DOC-073": "supply_works",     "DOC-074": "supply_works",
    # Build-and-operate and build-only infrastructure agreements.
    "DOC-076": "supply_works",     "DOC-079": "supply_works",
    "DOC-080": "supply_works",     "DOC-086": "supply_works",
    # Services: an IT master service agreement, a technology licence, and
    # three manpower/facility contracts.
    "DOC-085": "services",         "DOC-083": "services",
    "DOC-087": "services",         "DOC-088": "services",
    "DOC-089": "services",         "DOC-096": "services",
    # A specimen agreement engaging a manager - an employment instrument, and
    # one of the few in the corpus that can answer CAT-018.
    "DOC-097": "employment",
}

ALL = set(DOC_TYPES) - {"disclosure"}

# Which instrument types can carry each clause. Universal categories are
# every type; the rest are narrowed only where the instrument genuinely
# cannot contain the clause.
APPLIES = {
    "CAT-001": ALL, "CAT-002": ALL, "CAT-003": ALL, "CAT-004": ALL,
    "CAT-006": ALL, "CAT-016": ALL, "CAT-019": ALL,
    # A term and a notice period presuppose an ongoing relationship. A power
    # of attorney and a one-off indemnity have neither.
    "CAT-005": ALL - {"agency_authority"},
    "CAT-007": ALL - {"agency_authority"},
    "CAT-013": ALL - {"agency_authority"},
    "CAT-014": ALL - {"agency_authority"},
    "CAT-015": ALL - {"agency_authority"},
    # Restrictive covenants bind a person or firm engaged in a business
    # relationship. They have no place in a security instrument, a lease,
    # or a grant of authority.
    "CAT-008": {"employment", "services", "confidentiality", "entity"},
    # supply_works added on the evidence: the PSU works conditions bar the
    # contractor from engaging the employer's retired gazetted officers for
    # one or two years (BCPL clause 12.1, NIT Meghalaya clause 40). That is a
    # restriction on hiring the counterparty's former staff, whatever it is
    # headed. The consistency check below caught the omission against the gold.
    "CAT-009": {"employment", "services", "confidentiality", "entity", "supply_works"},
    "CAT-012": {"employment", "services", "confidentiality", "supply_works"},
    # Confidentiality regimes appear in relational contracts.
    "CAT-010": {"employment", "services", "confidentiality", "supply_works", "entity"},
    "CAT-011": {"employment", "services", "confidentiality", "supply_works", "entity"},
    # Money.
    "CAT-017": ALL - {"agency_authority", "confidentiality"},
    "CAT-018": {"employment"},          # boundary rule says employment only
    # employment added on the evidence: a contract of service prices its own
    # termination, and pay in lieu of notice is a pre-agreed sum payable on
    # ending the contract - which the boundary rule brings in as a
    # termination fee. Caught by the check against DOC-097 clause 8.
    "CAT-020": {"services", "supply_works", "property", "finance_security", "employment"},
    "CAT-021": {"services", "supply_works", "property", "finance_security"},
    # Performance categories belong to instruments that deliver something.
    "CAT-022": {"services", "supply_works", "finance_security"},
    # finance_security included because a hire-purchase agreement delivers
    # the asset: DOC-017 clause 1 fixes the day hire commences. The
    # consistency check below caught this omission against the gold.
    # property included because a lease delivers possession on a date:
    # DOC-062 makes rent payable only from actual possession, and gives the
    # lessee a termination right if possession is late. Caught by the check.
    "CAT-023": {"services", "supply_works", "finance_security", "property"},
    "CAT-024": {"services", "supply_works"},
    "CAT-025": {"services", "supply_works"},
    # Data handling arises wherever a party processes the other's data.
    "CAT-026": {"employment", "services", "confidentiality", "supply_works"},
}


def main() -> int:
    stage = Stage("A-4", "Category scope by instrument type")

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    categories = read_jsonl(CORPUS_DIR / "categories.jsonl")
    path = current_annotations_path()
    cells = read_jsonl(path)

    unmapped = [
        d["doc_id"] for d in documents
        if d.get("split") == "core" and d["doc_id"] not in DOC_TYPE
    ]
    if unmapped:
        stage.fail(f"core documents with no doc_type: {unmapped}")
        stage.info("Add them to DOC_TYPE — guessing a type would distort every count.")
        return 1

    for document in documents:
        document["doc_type"] = DOC_TYPE.get(
            document["doc_id"],
            "disclosure" if document.get("split") == "negatives" else "excluded",
        )
    write_jsonl(CORPUS_DIR / "documents.jsonl", documents)

    for category in categories:
        category["applies_to_doc_types"] = sorted(APPLIES[category["category_id"]])
    write_jsonl(CORPUS_DIR / "categories.jsonl", categories)

    types = {d["doc_id"]: d["doc_type"] for d in documents}
    applicable = inapplicable = negatives_kept = 0
    contradictions = []

    for cell in cells:
        if cell["split"] == "negatives":
            # Deliberately unscoped: unanswerable questions ARE the probe.
            cell["applicable"] = True
            negatives_kept += 1
            continue
        ok = types.get(cell["doc_id"]) in APPLIES[cell["category_id"]]
        cell["applicable"] = ok
        if ok:
            applicable += 1
        else:
            inapplicable += 1
            # A positive cell proves the category CAN apply to this
            # instrument. If scoping says otherwise, the scoping is wrong.
            if cell.get("is_impossible") is False:
                contradictions.append(cell["cell_id"])

    if contradictions:
        stage.fail(
            f"{len(contradictions)} cell(s) hold a real clause but were scoped out — "
            "the applicability map contradicts the annotation, so the map is wrong: "
            f"{contradictions[:10]}"
        )
        return 1

    write_jsonl(path, cells)

    per_doc = {}
    for cell in cells:
        if cell["split"] == "core" and cell["applicable"]:
            per_doc[cell["doc_id"]] = per_doc.get(cell["doc_id"], 0) + 1
    mean = sum(per_doc.values()) / max(len(per_doc), 1)

    stage.info(f"core cells applicable:   {applicable}")
    stage.info(f"core cells scoped out:   {inapplicable}")
    stage.info(f"negatives kept in full:  {negatives_kept}")
    stage.info(f"mean applicable categories per core document: {mean:.1f} of 26")
    stage.info("no positive cell was scoped out — map is consistent with the gold")

    emit_report(
        "category_scope_report.json",
        {
            "status": "PASS",
            "doc_types": DOC_TYPES,
            "core_cells_applicable": applicable,
            "core_cells_scoped_out": inapplicable,
            "negatives_cells_unscoped": negatives_kept,
            "mean_applicable_per_core_doc": round(mean, 2),
            "note": (
                "Cells are marked, not deleted; build_qa.py skips inapplicable ones. "
                "The negatives split keeps all 26 categories because its unanswerable "
                "questions are the hallucination probe. Applicability is a legal-"
                "ontology judgement that has NOT been reviewed by counsel."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
