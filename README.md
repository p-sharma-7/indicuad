# IndiCUAD

A CUAD-format clause-extraction benchmark built from **43 Indian SME and
public-sector contracts**, with gold spans resolved to character offsets against
a frozen corpus.

The release is **`qa/qa.json`** — 1,521 query rows over 507 cells, 651 gold
spans, every cell verified individually by a person.

```bash
python scripts/reproduce.py --verify     # 47 checks, read-only, offline, ~10s
```

Every figure on this page was recomputed from the artefacts on **2026-08-17** by
that command. See **[REPRODUCE.md](REPRODUCE.md)** for the full reproduction
guide, the three-tier harness, and the open items.

The paper's baselines (evaluation harness, every model's outputs, scores and
analysis) are in **[baseline/](baseline/README.md)**.

> **Scope.** The corpus is 43 core documents. No negatives-split document, cell
> or query contributes to any figure here, and none survives in the release: all
> 1,521 rows carry `split=core`.

---

## Current state

| | |
|---|---|
| Documents | **43**, 4,432,753 characters of frozen text |
| Paragraphs | **9,174** — 7,455 indexed, 1,719 flagged as running furniture |
| Categories | **26** — 7 low-friction / 11 moderate / 8 high-contest |
| Grid | 43 × 26 = **1,118 cells**; 173 scoped out by instrument type; **945 askable** |
| Positive cells | **507** (53.7% of askable) |
| Gold spans | **651**, all offset-exact, all inside a single paragraph |
| Query rows | **1,521** = 507 intent groups × 3 question registers |
| Effective N | **507** — registers are one observation asked three ways |
| Annotation | 945 / 945 cells decided; **100% individually human-verified** (P5) |
| Distractors | **1,953 / 1,953** slots filled — 651 mined, 1,302 perturbed |
| Blind study | **44 core cells** (of 60 sampled), two independent human annotators, **complete** |
| Validator self-tests | **16 / 17** passing — one known failure, below |

`qa/qa.json` sha256 begins `bb3202dd…`. The superseded pre-P5 file
`qa/qa.jsonl` begins `4a786383…`.

### What each row carries

`qa_id` · `doc_id` · `category_id` · `question` · `question_register` ·
`intent_group_id` · `answers[]` · `is_impossible` · `gold_paragraph_ids` ·
`doc_origin` · `tier` · `split` · `annotator_ids` · `annotation_pass` ·
`model_assisted` · `adjudicated` · `jaccard_agreement` · `annotator_notes` ·
`context_paragraph_ids` · `context_char_spans`

`answers[].answer_start` and `answer_end` are **document-level** offsets into
`work/normalized/{doc_id}.txt`. Anything that slices a context window out has to
rebase spans against `context_char_spans`.

> **Planned but never implemented — do not assume these exist:**
> `hard_negative_paragraph_ids`, `lexical_overlap_f1`, `template_family_id`,
> `legal_review_status`, `cuad_label`, `verdict_source`. `cuad_label` matters
> most: no mapping records which IndiCUAD categories correspond to which CUAD
> labels, so the zero-shot CUAD-transfer experiment cannot be run
> programmatically.

---

## How it was built

Three kinds of agent act in this pipeline and they are not interchangeable: a
**script** applies a rule, a **model** proposes a judgement, a **person**
decides one.

The first annotation pass was performed by **Claude Opus 5 through the
Anthropic API** under a written protocol, one call per document × category, with
the model emitting a `paragraph_id` and a verbatim quote — never a character
offset. Three mechanical gates then decide whether that becomes gold: the
paragraph must exist, the quote must be an exact substring of it, and the
computed offsets must re-slice the same bytes from the frozen text. A near miss
is **discarded, never repaired**.

Every cell was subsequently read one at a time by a person (the P5 pass, 15
August 2026), with a second human reviewer on 372 of them.

Full stage table with counts and instruments: [REPRODUCE.md](REPRODUCE.md#pipeline-stages).

### The prompts

| File | Governs|
|---|---|---|
| `prompts/extract_spans.v3.md` | the annotation pass |
| `prompts/perturb_distractors.v1.md` | distractor slots 2–3 |

---

## Measured quality

Two independent human annotators judged the same 60 blind cells (seed
`20260816`); neither had seen any of them during P5.

| | All 60 cells | 44 core cells |
|---|---:|---:|
| Annotator vs annotator κ | 0.8020 | **0.7027** |
| Dataset vs annotator κ | 0.6284 / 0.5066 | **0.4956 / 0.3936** |
| Dataset precision | 0.824 / 0.676 | 0.824 / 0.676 |
| Dataset recall | 0.848 / 0.852 | 0.875 / 0.885 |

**Quote the core-only column.** 16 of the 60 sampled cells were negatives that
no longer exist in this dataset. Precision is identical and recall improves —
the dataset's behaviour did not change — but dropping those cells lifts chance
agreement from 0.495 to 0.541, and κ discounts by chance. The published 0.802
was partly carried by cells the release does not contain. Full explanation:
[REPRODUCE.md](REPRODUCE.md#agreement-figures-read-before-quoting-them).

The error is asymmetric toward **over-calling** — false positives exceed false
negatives against both annotators. That is the expected direction given the
protocol's explicit instruction that recall beats precision, and should be
reported as a designed trade-off rather than a defect.

---

## Design decisions worth knowing

**Offsets are structural, not best-effort.** Every paragraph satisfies
`normalized_text[char_start:char_end] == text` exactly, verified before the file
is written and again by `reproduce.py --verify` across all 9,174 paragraphs.
Segmentation only ever places boundaries at offsets that already exist in the
frozen text.

**No quotation is ever retyped.** Spans are sliced from the frozen document, so
the corpus's own defects survive intact — `PAYMENTSCHEDULE`, `daysthereafter`,
`thePerformance`, a line break inside "Company's Act 2013". A cleaned-up quote
would no longer match its source, which is why the prompt names these defects
explicitly and tells the model not to helpfully repair them.

**Running furniture is separated from clause text, not deleted.** The AP Law
Department templates carry a 70-character running page header which was being
folded onto the clause below it — the core split's only Governing Law clause
once arrived 48% furniture by character. Furniture is now identified before
merging, held out of merges, re-tested after splits, and flagged
`is_boilerplate`. It stays in `paragraphs.jsonl` so the paragraph set still
tiles each document and the call is reversible; the index, the sweep and gold
spans all skip it. `segment_paragraphs.py` fails if any paragraph flagged
boilerplate contains operative language, so this can never silently suppress an
obligation. One documented exception: a CAT-001 document title may sit in
furniture under the masthead rule, which applies to 3 gold paragraphs.

**Text-layer defects are flagged, never repaired.** Normalization touches only
line endings, zero-width characters, non-breaking spaces and trailing
whitespace, each counted per document. 9 documents carry flagged defects —
Cyrillic homoglyphs (`SEВІ` for `SEBI`) and one corrupt Devanagari column —
reported rather than rewritten, because rewriting would alter legal text. They
are worth knowing about because they defeat plain-Latin keyword sweeps.

**Category IDs are anchored to the protocol.** The protocol does not publish a
full ID table, but its worked examples fix `CAT-008` = Non-Compete, `CAT-017` =
Payment Terms & Schedule, `CAT-022` = Warranty Duration. The ordering satisfies
all three so the protocol's examples stay literally correct. The other 23
assignments are open to review.

**Scoping audits the schema, not the reverse.** Asking a power of attorney about
warranty duration is not a hard question but a meaningless one. Each category is
restricted to the instrument types that can carry it, and a consistency check
refuses to scope out a cell that holds a real clause. That check caught the map
being wrong twice — CAT-009 missing PSU works conditions that bar hiring the
employer's retired officers, and CAT-020 missing employment contracts, which
price their own termination.

**Nothing is deleted.** Every removal has an archive file: `corpus/excluded_*.jsonl`,
`qa/qa_unanswerable.json`, and a `.bak` beside every rewritten artefact.

---


## Repository map

```
qa/qa.json              THE RELEASE — 1,521 rows
qa/qa_unanswerable.json the 1,704 dropped rows, annotation intact
qa/distractors.json       651 records × 3 slots, all filled
corpus/                   documents · paragraphs · categories (+ excluded_*)
work/normalized/          the frozen text every offset refers to
prompts/                  the two prompt files
annotations/blind/        the two completed blind returns + sealed key
validation/               one JSON report per validator
scripts/reproduce.py      the entry point — verify / rebuild / annotate
baseline/                 evaluation harness + model outputs for the paper
```

### Licence and redistribution

Annotations are original work. The source contracts are public documents from
Indian government portals and exchange filings; NSE archive terms permit viewing
but **not redistribution**. The intended release ships `qa.json` with
`document_source_metadata.csv` plus offsets and a downloader that reconstructs the corpus locally
and verifies hashes. **That downloader has not been written.**
