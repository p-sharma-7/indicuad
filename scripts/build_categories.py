"""Stage 5 — build corpus/categories.jsonl (26 categories).

Category names, attention tiers, and boundary rules are transcribed from
the frozen ANNOTATION_PROTOCOL §4. They are NOT invented here; only
markdown emphasis markers are dropped in transcription.

Category ID ordering
--------------------
The protocol does not publish a full ID table, but its worked examples
fix three IDs:

    CAT-008 = Non-Compete            (§2, worked example 1)
    CAT-017 = Payment Terms & Schedule (§2, worked example 3)
    CAT-022 = Warranty Duration      (§2, worked example 2)

The ordering below satisfies all three anchors exactly, so the protocol's
examples stay literally correct, and otherwise follows contract reading
order: identity and dates, restrictive covenants, confidentiality and IP,
termination, dispute resolution, money, liability, delivery, data.

Human-authored vs. drafted
--------------------------
  boundary_rule        transcribed verbatim from the protocol
  attention_tier       from the protocol
  description          DRAFTED here
  *_question (x3)      DRAFTED here (permitted: query drafting)
  negative_examples    DRAFTED, seeded from the protocol's confusion column
  sweep_terms          DRAFTED (mechanical aid for the P2 keyword sweep)

Protocol §7 requires a lawyer or law-faculty reviewer to read this file
BEFORE P1. Every record therefore carries
review_status = "DRAFT_PENDING_LEGAL_REVIEW".

Output: corpus/categories.jsonl
"""

from __future__ import annotations

import sys

from indicuad_common import CORPUS_DIR, Stage, emit_report, write_jsonl

CANONICAL_TEMPLATE = (
    'Highlight the parts (if any) of this contract related to "{name}". '
    "Details: {detail}"
)

# (number, name, attention_tier, boundary_rule, description, canonical detail,
#  semi_formal, naive_layperson, negative_examples, sweep_terms)
CATEGORIES: list[tuple] = [
    (
        1, "Document Name", "low_friction",
        "Title line only. Not the recital describing the agreement.",
        "The title of the instrument as it appears at the head of the document.",
        "What is the title or name of this contract?",
        "What kind of agreement is this document?",
        "what is this paper called",
        ["The recital sentence describing what the agreement does",
         "A subject line in a covering letter"],
        ["agreement", "deed", "contract", "memorandum", "indenture", "letter of",
         "power of attorney", "undertaking", "guarantee"],
    ),
    (
        2, "Parties", "low_friction",
        "Each party's full identifying clause (name + constitution + address), "
        "one span per party.",
        "The full identifying clause for each contracting party.",
        "Who are the parties to this contract?",
        "Which companies or people are signing this agreement?",
        "who is this contract between",
        ["A third party merely referred to in a clause",
         "The witness or attesting signatory",
         "A guarantor named only in an annexure"],
        ["between", "party of the first part", "hereinafter referred to as",
         "represented by", "having its registered office", "son of", "aged about",
         "resident of", "CIN", "PAN"],
    ),
    (
        3, "Agreement Date", "low_friction",
        "The date the contract is dated, not the effective date if different.",
        "The date on which the contract is dated or executed.",
        "On what date is this contract dated?",
        "When was this agreement signed or dated?",
        "what date is on this contract",
        ["The effective or commencement date where it is stated separately",
         "The date of a referenced purchase order"],
        ["made at", "made on", "dated", "this day of", "executed on",
         "entered into on", "on this"],
    ),
    (
        4, "Effective Date", "low_friction",
        "Only if explicitly distinguished from Agreement Date; else leave empty.",
        "The date the contract takes effect, where explicitly distinguished "
        "from the date it is dated.",
        "What is the effective date of this contract, if stated separately "
        "from the agreement date?",
        "When does this agreement actually start taking effect?",
        "when does this actually kick in",
        ["The agreement date restated",
         "A delivery or milestone start date"],
        ["effective date", "shall come into force", "commence with effect from",
         "w.e.f.", "with effect from", "shall be effective"],
    ),
    (
        5, "Expiration Date", "low_friction",
        "End of the initial term. Not renewal terms.",
        "The end of the initial term of the contract.",
        "When does the initial term of this contract expire?",
        "When does this agreement come to an end?",
        "when does this run out",
        ["A renewal or extension term",
         "A warranty or defect-liability period",
         "A survival period after termination"],
        ["shall expire", "expiry", "initial term", "period of", "valid until",
         "valid up to", "term of this agreement", "shall remain in force"],
    ),
    (
        6, "Governing Law", "low_friction",
        "Choice-of-law sentence only. If governing law and jurisdiction are one "
        "sentence, mark it here and under Dispute Resolution.",
        "The choice-of-law provision naming the law that governs the contract.",
        "Which law governs this contract?",
        "Which country's or state's laws apply to this agreement?",
        "whose laws apply here",
        ["An arbitration seat or venue, unless in the same sentence",
         "A reference to a statute regulating the subject matter"],
        ["governed by", "governing law", "laws of India", "in accordance with the laws",
         "construed in accordance", "subject to the laws"],
    ),
    (
        7, "Notice Period", "low_friction",
        "The period required for a notice of termination or non-renewal.",
        "The length of notice required to terminate or decline renewal.",
        "What notice period is required to terminate or not renew this contract?",
        "How much advance notice must a party give to end this agreement?",
        "how much warning do i have to give to quit",
        ["A notice address or mode-of-service clause",
         "A cure period for remedying a breach"],
        ["notice period", "days' notice", "days notice", "month's notice",
         "prior written notice", "advance notice", "by giving", "serving notice"],
    ),
    (
        8, "Non-Compete", "high_contest",
        "Restriction on competing. If it restricts dealing exclusively with one "
        "party, that is exclusivity, not this.",
        "A restriction on a party competing with the counterparty during or "
        "after the contract.",
        "Is there a restriction on competing with the counterparty during or "
        "after the term of this contract?",
        "Can this person or company be stopped from working with a competitor?",
        "if i leave can i join a rival",
        ["An exclusivity clause requiring dealing only with one party",
         "A non-solicitation of employees clause",
         "A confidentiality obligation"],
        ["non-compete", "not compete", "shall not compete", "competing business",
         "carry on any business", "similar business", "directly or indirectly engage",
         "rival", "competitor"],
    ),
    (
        9, "No-Solicit of Employees", "moderate",
        "Restriction on soliciting/hiring the counterparty's staff. Not customer "
        "non-solicit.",
        "A restriction on soliciting, poaching, or hiring the counterparty's "
        "employees.",
        "Is there a restriction on soliciting or hiring the other party's "
        "employees?",
        "Can one side be stopped from hiring the other side's staff?",
        "can they poach our workers",
        ["A non-solicitation of customers or clients",
         "A general non-compete restriction"],
        ["solicit", "non-solicit", "entice", "induce", "poach", "employ any employee",
         "offer employment", "hire any"],
    ),
    (
        10, "Confidentiality Scope", "high_contest",
        "The obligation itself and the definition of confidential information.",
        "The confidentiality obligation and the definition of what counts as "
        "confidential information.",
        "What confidential information must be protected, and what is the "
        "obligation of confidentiality?",
        "What information must be kept secret under this agreement?",
        "what am i not allowed to tell anyone",
        ["Carve-outs and exceptions to confidentiality",
         "A personal data handling clause",
         "An intellectual property assignment"],
        ["confidential", "confidentiality", "secrecy", "non-disclosure",
         "proprietary information", "trade secret", "shall not disclose",
         "keep confidential"],
    ),
    (
        11, "Confidentiality Exceptions", "high_contest",
        "Carve-outs only (public domain / independently developed / required by "
        "law / already known). Never the general obligation — that is "
        "Confidentiality Scope.",
        "Carve-outs from the confidentiality obligation.",
        "What are the exceptions to the confidentiality obligation?",
        "When is a party allowed to disclose confidential information anyway?",
        "when am i allowed to share it",
        ["The general confidentiality obligation itself",
         "The definition of confidential information"],
        ["public domain", "publicly available", "independently developed",
         "required by law", "compelled by", "already known", "rightfully received",
         "shall not apply", "except", "other than"],
    ),
    (
        12, "IP Ownership Assignment", "high_contest",
        "IP created by one party vesting in the other. Not a licence grant.",
        "Assignment or vesting of intellectual property created by one party "
        "in the other.",
        "Does intellectual property created under this contract vest in or get "
        "assigned to one of the parties?",
        "Who owns the designs, software, or inventions created under this "
        "contract?",
        "who owns what i make",
        ["A licence or right-to-use grant that does not transfer ownership",
         "A confidentiality obligation over technical information"],
        ["intellectual property", "copyright", "patent", "work product",
         "work made for hire", "shall vest", "assign", "assignment", "moral rights",
         "invention", "design", "trademark"],
    ),
    (
        13, "Termination for Convenience", "high_contest",
        "Termination without cause, by notice alone.",
        "A right to terminate without cause, on notice alone.",
        "Can a party terminate this contract for convenience, without cause?",
        "Can either side end this agreement just by giving notice?",
        "can they cancel for no reason",
        ["Termination on breach, default, or insolvency",
         "Expiry of the term by effluxion of time"],
        ["terminate", "termination", "for convenience", "without cause",
         "without assigning any reason", "at its sole discretion", "at will",
         "by giving notice"],
    ),
    (
        14, "Termination for Cause", "high_contest",
        "Termination on breach, insolvency, or default. If one clause covers "
        "both, mark it under both.",
        "A right to terminate on breach, default, or insolvency.",
        "Can a party terminate this contract for cause, such as breach, "
        "default, or insolvency?",
        "Can this agreement be ended if one side breaks the rules or goes "
        "bankrupt?",
        "what happens if they mess up can we cancel",
        ["Termination without cause on notice alone",
         "A liquidated damages or penalty clause"],
        ["breach", "default", "insolvenc", "winding up", "liquidation",
         "bankrupt", "material breach", "failure to perform", "terminate forthwith",
         "cure period", "remedy the breach"],
    ),
    (
        15, "Survival Period", "moderate",
        "Which obligations survive termination and for how long.",
        "Obligations that continue after termination or expiry, and their "
        "duration.",
        "Which obligations survive termination or expiry of this contract, and "
        "for how long?",
        "Which duties continue even after this agreement ends?",
        "what still applies after it is over",
        ["The initial term or expiry date",
         "A warranty duration"],
        ["survive", "survival", "notwithstanding termination",
         "shall continue to be binding", "after the expiry", "remain in effect",
         "continue in force"],
    ),
    (
        16, "Dispute Resolution & Arbitration Seat", "moderate",
        "Arbitration clause, seat, venue, governing rules, court-jurisdiction "
        "fallback.",
        "How disputes are resolved: arbitration, seat and venue, governing "
        "rules, and any court-jurisdiction fallback.",
        "How are disputes under this contract resolved, and what is the "
        "arbitration seat or court jurisdiction?",
        "What happens if the parties end up in a dispute?",
        "where do we fight it out if there is a problem",
        ["A choice-of-law sentence standing alone",
         "A notice provision"],
        ["arbitration", "arbitral", "arbitrator", "Arbitration and Conciliation Act",
         "seat of arbitration", "venue", "jurisdiction", "courts at",
         "conciliation", "mediation", "dispute", "tribunal"],
    ),
    (
        17, "Payment Terms & Schedule", "moderate",
        "Amounts, milestones, invoicing mechanics, due dates, late-payment "
        "interest.",
        "Payment amounts, milestones, invoicing mechanics, due dates, and "
        "late-payment interest.",
        "What are the payment amounts, milestones, and due dates?",
        "How and when does payment have to be made under this contract?",
        "when do i get paid and how much",
        ["A bare contract value figure in a disclosure with no payment "
         "obligation",
         "Salary and statutory benefits in an employment document"],
        ["payment", "invoice", "shall pay", "due date", "instalment", "installment",
         "milestone", "interest", "advance", "retention", "GST", "consideration",
         "price", "billing", "within 30 days"],
    ),
    (
        18, "Remuneration & Statutory Benefits", "moderate",
        "Salary, allowances, PF, ESI, gratuity, bonus. Employment documents only.",
        "Salary, allowances, and statutory employment benefits.",
        "What remuneration and statutory benefits are payable to the employee?",
        "What salary and benefits does the employee receive?",
        "what is the pay and what perks do i get",
        ["Vendor or contractor payment terms",
         "A reimbursement of expenses clause in a services contract"],
        ["salary", "remuneration", "wages", "allowance", "provident fund", "PF",
         "ESI", "gratuity", "bonus", "leave", "increment", "CTC", "emoluments"],
    ),
    (
        19, "Cap on Liability", "high_contest",
        "A ceiling on liability, or a time limit for bringing claims.",
        "A ceiling on a party's liability, or a limitation period for claims.",
        "Is there a cap on liability or a time limit for bringing claims under "
        "this contract?",
        "Is there a maximum amount either side can be made to pay?",
        "how much can they make me pay at most",
        ["Liquidated damages payable on breach or delay",
         "An indemnity obligation with no stated ceiling"],
        ["liability", "limitation of liability", "shall not exceed", "aggregate liability",
         "cap", "indirect", "consequential", "in no event", "maximum liability"],
    ),
    (
        20, "Liquidated Damages", "high_contest",
        "A pre-agreed sum payable on breach/delay, or a termination fee. Not a "
        "general damages clause.",
        "A pre-agreed sum payable on breach or delay, or a termination fee.",
        "Are liquidated damages, penalties, or a termination fee payable on "
        "breach or delay?",
        "Is there a fixed penalty if something is late or goes wrong?",
        "is there a fine if it is late",
        ["A general right to claim damages at law",
         "A cap on overall liability"],
        ["liquidated damages", "LD", "penalty", "pre-estimate", "per week of delay",
         "0.5% of", "termination fee", "compensation for delay", "price reduction"],
    ),
    (
        21, "Insurance", "moderate",
        "Requirement to maintain insurance for the counterparty's benefit.",
        "An obligation to take out or maintain insurance for the "
        "counterparty's benefit.",
        "Is either party required to maintain insurance?",
        "Does anyone have to buy insurance under this contract?",
        "do i need insurance for this",
        ["An indemnity clause with no insurance requirement",
         "A bank guarantee or performance security"],
        ["insurance", "insured", "policy", "workmen's compensation",
         "third party liability", "CAR policy", "shall insure", "keep insured"],
    ),
    (
        22, "Warranty Duration", "moderate",
        "The duration of a warranty. Warranty scope itself is not this category.",
        "The length of any warranty or defect-liability period.",
        "What is the duration of any warranty against defects or errors in "
        "goods or services provided?",
        "How long is the warranty period under this contract?",
        "how long is it under warranty",
        ["The scope of what the warranty covers",
         "An O&M or maintenance period",
         "The initial term of the contract"],
        ["warranty", "warrant", "guarantee period", "defect liability",
         "defects liability", "DLP", "months from the date of", "free of defects",
         "workmanship"],
    ),
    (
        23, "Delivery & Milestone Obligations", "moderate",
        "Delivery dates, milestones, completion schedules, acceptance criteria.",
        "Delivery dates, milestones, completion schedules, and acceptance "
        "criteria.",
        "What are the delivery dates, milestones, or completion schedules?",
        "When do the goods or work have to be delivered or finished?",
        "when does the stuff have to be done by",
        ["A payment milestone with no delivery obligation",
         "The contract expiry date"],
        ["delivery", "deliver", "completion", "milestone", "schedule",
         "time is of the essence", "commissioning", "acceptance", "handover",
         "timeline", "within weeks", "date of completion"],
    ),
    (
        24, "O&M Period", "moderate",
        "Operation and maintenance commitment and its term.",
        "An operation and maintenance commitment and how long it lasts.",
        "Is there an operation and maintenance obligation, and what is its term?",
        "Does anyone have to run or maintain this after it is built, and for "
        "how long?",
        "who looks after it afterwards and for how long",
        ["A warranty or defect-liability period",
         "A delivery or commissioning milestone"],
        ["operation and maintenance", "O&M", "maintenance period",
         "comprehensive maintenance", "AMC", "annual maintenance", "upkeep",
         "servicing", "five years", "shall maintain"],
    ),
    (
        25, "Technical Standards Compliance", "moderate",
        "Named standards: BIS, IS, IEC, MNRE, ISO, statutory technical "
        "specifications.",
        "Compliance with named technical standards or statutory technical "
        "specifications.",
        "Which technical standards or specifications must be complied with?",
        "What quality or technical standards does the work have to meet?",
        "what rules does the equipment have to follow",
        ["A general compliance-with-law clause",
         "A quality warranty with no named standard"],
        ["BIS", "IS ", "IEC", "MNRE", "ISO", "ALMM", "standard", "specification",
         "conformity", "certified", "test report", "as per IS", "technical specification"],
    ),
    (
        26, "Personal Data Handling", "moderate",
        "Text governing collection, storage, processing, or transfer of personal "
        "data. Extraction only — never judge DPDP compliance.",
        "Provisions governing collection, storage, processing, or transfer of "
        "personal data.",
        "How is personal data collected, stored, processed, or transferred "
        "under this contract?",
        "What does this contract say about handling people's personal "
        "information?",
        "what happens to my personal details",
        ["A general confidentiality obligation over business information",
         "An IT security or systems-access clause with no personal data"],
        ["personal data", "personal information", "data protection", "DPDP",
         "data principal", "data fiduciary", "privacy", "sensitive personal",
         "Aadhaar", "consent", "processing of data", "data subject"],
    ),
]


def build() -> list[dict]:
    records = []
    for (
        number, name, tier, boundary_rule, description, detail,
        semi_formal, naive, negatives, sweep,
    ) in CATEGORIES:
        records.append(
            {
                "category_id": f"CAT-{number:03d}",
                "category_name": name,
                "attention_tier": tier,
                "description": description,
                "canonical_question": CANONICAL_TEMPLATE.format(name=name, detail=detail),
                "semi_formal_question": semi_formal,
                "naive_layperson_question": naive,
                "boundary_rule": boundary_rule,
                "negative_examples": negatives,
                "sweep_terms": sweep,
                "boundary_rule_source": "ANNOTATION_PROTOCOL §4 (transcribed verbatim)",
                "review_status": "DRAFT_PENDING_LEGAL_REVIEW",
                "reviewed_by": None,
                "reviewed_at": None,
            }
        )
    return records


ANCHORS = {"CAT-008": "Non-Compete", "CAT-017": "Payment Terms & Schedule",
           "CAT-022": "Warranty Duration"}

EXPECTED_TIER_COUNTS = {"low_friction": 7, "moderate": 11, "high_contest": 8}


def main() -> int:
    stage = Stage("5", "Category configuration")
    records = build()

    problems = []
    if len(records) != 26:
        problems.append(f"expected 26 categories, built {len(records)}")

    names = [r["category_name"] for r in records]
    if len(set(names)) != len(names):
        problems.append("duplicate category names")

    ids = [r["category_id"] for r in records]
    if len(set(ids)) != len(ids):
        problems.append("duplicate category ids")

    tier_counts: dict[str, int] = {}
    for record in records:
        tier_counts[record["attention_tier"]] = tier_counts.get(record["attention_tier"], 0) + 1
    if tier_counts != EXPECTED_TIER_COUNTS:
        problems.append(f"attention tier counts {tier_counts} != protocol {EXPECTED_TIER_COUNTS}")

    by_id = {r["category_id"]: r["category_name"] for r in records}
    for cid, expected in ANCHORS.items():
        if by_id.get(cid) != expected:
            problems.append(
                f"protocol anchor violated: {cid} should be {expected!r}, got {by_id.get(cid)!r}"
            )

    if problems:
        for problem in problems:
            stage.fail(problem)
        emit_report("category_report.json", {"status": "FATAL", "problems": problems})
        return 1

    write_jsonl(CORPUS_DIR / "categories.jsonl", records)
    stage.info(f"wrote corpus/categories.jsonl (26 categories)")
    for tier, count in EXPECTED_TIER_COUNTS.items():
        stage.info(f"  {tier:<14} {count}")
    stage.info("protocol ID anchors satisfied: " + ", ".join(sorted(ANCHORS)))
    stage.warn("categories.jsonl requires lawyer / faculty review before P1 (protocol §7).")

    emit_report(
        "category_report.json",
        {
            "status": "PASS_PENDING_LEGAL_REVIEW",
            "categories_total": len(records),
            "attention_tier_counts": tier_counts,
            "protocol_id_anchors": ANCHORS,
            "boundary_rules_source": "transcribed verbatim from ANNOTATION_PROTOCOL §4",
            "drafted_fields_requiring_review": [
                "description", "canonical_question", "semi_formal_question",
                "naive_layperson_question", "negative_examples", "sweep_terms",
            ],
            "blocks_p1": True,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
