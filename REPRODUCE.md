# Reproducing IndiCUAD

**IndiCUAD** is a CUAD-format clause-extraction benchmark built from **43 Indian
SME and public-sector contracts**. The release is `qa/qa.json`: **1,521 query
rows** over **507 cells** (document × clause category), carrying **651 gold
spans** resolved to character offsets against a frozen corpus.

Every figure in this file was recomputed from the artefacts on
**2026-08-17** by `scripts/reproduce.py --verify`, not copied from the build
documents — several of which predate the corpus as it now stands.

> **Scope.** The corpus is 43 core documents. No negatives-split document, cell
> or query contributes to any figure here, and none survives in the release:
> all 1,521 rows carry `split=core`.

---

## Quickstart

```bash
python scripts/reproduce.py --verify
```

Read-only, offline, about ten seconds. No API key, no network, no third-party
package — stdlib only. Exit code `0` means every check passed. This is the tier
an artifact evaluator should run, and the only one that can fail in a way that
means the dataset is wrong.

Current state on a clean checkout: **47 of 47 checks pass**, with two advisory
notes (`SNAPSHOT.txt` drift, and the disputed §7 legal-review status).

---

## The three tiers

The pipeline is **not uniformly reproducible**, and this harness is built to
say so rather than to paper over it.

| Tier | Command | Deterministic | Cost | What it establishes |
|---|---|---|---|---|
| **Verify** | `reproduce.py --verify` | yes | free | The released artefacts are internally consistent and match their published figures |
| **Rebuild** | `reproduce.py --rebuild` | yes | free | The derived files re-derive byte-identically from the corpus and annotations |
| **Annotate** | `reproduce.py --annotate` | **no** | paid API | Produces a *re-annotation* to compare against — never a reproduction |

Both write-tiers accept `--dry-run` to print the stage list without executing.

### Why the third tier can never reproduce

The prompted stage is non-deterministic by construction. `temperature`, `top_p`
and `top_k` return HTTP 400 on `claude-opus-5` and are not sent; there is no
seed; and `propose_v3.py` deliberately varies `effort` across samples
(`high → high → medium`) because varying it *is* the self-consistency mechanism.

So: the reproducible *object* is the procedure and its gates. The reproducible
*artefact* is the frozen corpus plus the released annotations and their hashes.
Re-running the model is a comparison experiment, not a reproduction, and any
paper text should say so.

---

## What `--verify` checks

47 assertions in eight groups. The ones that matter most are in **gold spans** —
if those fail, the dataset is broken.

**corpus** — 43 core documents; frozen text present for each; 4,432,753
characters; 9,174 paragraphs (1,719 furniture, 7,455 indexed); and the invariant
the whole offset scheme rests on: `frozen[char_start:char_end] == paragraph.text`
for every paragraph.

**schema** — 26 categories; tiers 7 `low_friction` / 11 `moderate` / 8
`high_contest`; three question registers and a boundary rule on each.

**grid** — 43 × 26 = 1,118 core cells; 173 scoped out by instrument type; 945
askable; the arithmetic closes.

**release** — 1,521 rows over 507 cells and 507 intent groups; exactly three
rows per group, one per register; every row `split=core`; `is_impossible` false
throughout; `qa_id` unique; 651 gold spans.

**gold spans** — for every answer: `frozen[answer_start:answer_end] ==
answer.text`; the `paragraph_id` exists; the span lies wholly inside that
paragraph. Spans occurring twice within their paragraph are counted and
reported.

**context** — all 1,521 rows carry a context window; every `context_char_span`
reproduces its paragraphs verbatim; every gold span falls inside its window.

**distractors** — 651 records, 1,953 slots, all filled (651 mined + 1,302
perturbed); no distractor appears in its own source document; none equals its
gold; none duplicates a sibling.

**blind study** — recomputes Cohen's κ from the two annotation CSVs against the
labels and asserts it reproduces the published run exactly. It also reports the
core-only recomputation, which differs materially — see
[Agreement figures](#agreement-figures-read-before-quoting-them).

**hashes** — prints sha256 for the five key artefacts and diffs `SNAPSHOT.txt`.

### Sample output

```
  gold spans
  ------------------------------------------------------------------
  ok   span text == frozen[start:end]                      0
  ok   span's paragraph_id exists                          0
  ok   span lies inside its paragraph                      0
       spans occurring twice in their paragraph            0

  blind study
  ------------------------------------------------------------------
  ok   cells judged                                        60
  ok   reproduces published κ (annotators)                 0.8020
  <!-- ok   reproduces published κ (vs Akshay Kumar)                    0.6284 -->
  ok   reproduces published κ (vs Kanishka)                   0.5066
       [core only] n                                       44
       [core only] annotator vs annotator κ                0.7027
       [core only] dataset vs Akshay Kumar / Kanishka κ               0.3956 / 0.2936

  ==================================================================
  all 47 checks passed
```

---

## Pipeline stages

Three kinds of agent act here and they are not interchangeable: a **script**
applies a rule, a **model** proposes a judgement, a **person** decides one.

| # | Stage | Instrument | Carried forward | Set aside | Agent |
|---|---|---|---:|---:|---|
| 1 | Collection | `document_source_metadata.csv` | 43 documents | — | human |
| 2 | Extract · OCR · normalize · freeze | `extract_pdf_text.py`, `ocr_pages.py`, `normalize_corpus.py` | 4,432,753 chars | 0 rewritten | script |
| 3 | Segmentation | `segment_paragraphs.py` | 9,174 paragraphs | 1,719 furniture | script |
| 4 | Near-duplicate screen | `near_duplicate_screen.py` | 43 compared | 1 pair flagged | script |
| 5 | Category schema | `build_categories.py`, `approve_categories.py` | 26 categories | — | human |
| 6 | Grid construction | `build_annotation_workspace.py` | 1,118 cells | — | script |
| 7 | Scoping by instrument type | `apply_category_scope.py` | 945 cells | 173 cells | human |
| 8 | Keyword sweep | `sweep.py` | candidates only | ~2,800 false leads | script |
| 9 | **Prompted proposal** | `propose_v3.py` + `prompts/extract_spans.v3.md` | 945 cells decided | spans failing 3 gates | **Claude Opus 5** |
| 10 | Span resolution | `resolve_span()`, `split_straddling_spans.py` | 651 spans | 8 redone, 11 ambiguous | script |
| 11 | Release build | `build_qa.py` | 2,835 rows | — | script |
| 12 | P5 verification | *procedure never written down* | 945 cells | 1 gold replaced | 3 humans |
| 13 | Blind study | `blind_subsample_v2.py` | 44 core cells | measures only | 2 humans, blind |
| 14 | Context windows | `add_context_paragraphs.py` | every row | — | script |
| 15 | Drop unanswerable rows | `drop_unanswerable.py` | 1,521 rows · 507 cells | 1,314 rows · 438 cells | human |
| 16 | Distractor slots | `mine_distractors.py`, `merge_perturbations.py` | 1,953 of 1,953 | 4 merge gates | script + model |

Stages 7, 11, 14, 15 and 16 are what `--rebuild` re-runs. Stage 9 is what
`--annotate` re-runs.

### The prompt files

| File | Governs |
|---|---|
| `prompts/extract_spans.v3.md` | Stage 9 — one call per document × category; `propose_v3.py` hashes it into every proposal row |
| `prompts/perturb_distractors.v1.md` | Stage 16 — distractor slots 2 and 3 |

Any rerun against them is a re-annotation (the model call is not deterministic).

`extract_spans.v3.md` is parsed by `propose_v3.split_prompt()` on its H2
headers and then `.format()`-ed. Two hazards, both of which have already bitten
it: the header strings must not appear anywhere above their own block, and a
bare `{` in block A or B raises `ValueError`. Both are documented in the file
header.

---

## How the prompt was used

This section is for a reviewer who wants to re-run the labelling and produce a
*comparable* dataset. Exact reproduction is impossible (no seed, no
temperature) — what follows is how to get a
run that can be measured against the release.

### One call = one cell

The unit of work is a **cell**: one document × one clause category. Nothing is
batched, and no conversation is carried between calls — every cell is judged
from a clean context.

```
43 documents × 26 categories × k samples = 1,118 calls at k=1
```

Each request is assembled from three blocks:

| Block | Contents | Varies |
|---|---|---|
| **system** | The task rules, the three gates, `is_impossible` semantics, one worked example. 8,290 chars. | never |
| **USER block A** | The entire contract, every paragraph tagged `<<PAR-DOC015-0002>>`, furniture marked `[boilerplate]`. | per document |
| **USER block B** | One category card — id, name, description, boundary rule, negative examples. 954 chars. | per call |

The model returns a `CategoryExtraction` object enforced by
`output_config.format` as a JSON schema (pydantic `extra="forbid"` → 
`additionalProperties: false`), so the reply cannot be prose or malformed JSON.

The model **never emits character offsets** — it emits a `paragraph_id` and a
verbatim `quote`, and `resolve_span()` computes offsets. Everything then passes
the three gates in Figure 2: unknown paragraph, non-verbatim quote, or offsets
that don't re-slice the frozen text are discarded, never repaired.

### Caching is the whole cost model — do not break it

The system block and block A carry `cache_control`, so a document's text is
paid for once and read back by its other 25 category calls. That is the
difference between a $25 input bill and a $165 one:

| | Tokens | Rate | Cost |
|---|---:|---|---:|
| Cache write (once per document) | ~1,256,000 | $6.25/M | **$7.85** |
| Cache read (25 of every 26 calls) | ~31,409,000 | $0.50/M | **$15.70** |
| Block B (uncached, varies per call) | ~267,000 | $5.00/M | **$1.33** |
| **Input subtotal** | | | **~$25** |
| Output @ 600–1,500 tok/call | 0.7–1.7M | $25.00/M | $17–42 |
| **Total, k=1** | | | **~$42–67** |

*Estimated at 4 chars/token against Claude Opus 5 list rates. Legal text with
placeholder runs and OCR artifacts may tokenize worse — verify with
`client.messages.count_tokens()` before budgeting, never with `tiktoken`.*
At k=3 the cache write is unchanged and only reads and output scale: ~$110–186.

**Three ways a reviewer accidentally destroys the discount:**

1. **Parallelising a document's 26 calls.** A cache entry is readable only
   after the first response *starts streaming*. Fire all 26 at once and every
   one pays full price. `propose_v3.py` is sequential per document by design —
   if you parallelise, parallelise across *documents*, never within one.
2. **Letting the cache expire mid-document.** `cache_control:
   {"type": "ephemeral"}` is a **5-minute** TTL. Fourteen core documents exceed
   100k characters (DOC-086 is 622,509), and a single high-effort call on one
   of those can approach that window on its own. For a re-run, change it to
   `{"type": "ephemeral", "ttl": "1h"}` — the write cost doubles to ~$12.56,
   which is cheap insurance against re-paying $165 of uncached input.
3. **Editing the system block between documents.** Caching is a prefix match:
   one byte anywhere in the prefix invalidates everything after it. Freeze the
   prompt for the whole run.

### Decoding settings, and why there is no seed

```python
model       = "claude-opus-5"
max_tokens  = 16000
thinking    = {"type": "adaptive"}
output_config = {"effort": <ladder>, "format": {"type": "json_schema", ...}}
# temperature / top_p / top_k are NOT sent — they return HTTP 400 on this model
```

There is no seed parameter, and the sampling controls that would approximate
one are rejected by the API. `--k` (repeated sampling) and the effort ladder
(`high → high → medium`, cycled across samples) are the only variation knobs;
at `--k 1` every call runs at `high`. Self-consistency across k samples is the
substitute for determinism.

Two settings a reviewer should reconsider rather than copy:

- **`effort`.** Current guidance for this model is to start at `xhigh` for
  agentic and extraction-heavy work and sweep downward — `low` and `medium` are
  unusually strong and are the primary cost lever. The `high/high/medium`
  ladder in the repo is a defensible default, not a tuned one.
- **`max_tokens=16000`.** Thinking is **on by default** on this model and
  `max_tokens` caps thinking *plus* response together. On a long contract at
  high effort this can truncate; the truncated JSON then fails schema
  validation, and `run_document` logs the cell as `FAILED` and moves on. Raise
  it (and stream) or count your `FAILED` lines.

### Failure modes that are silent unless you look

- **Refusals.** `propose_v3.py` raises on `stop_reason == "refusal"`; the
  caller catches it, prints `FAILED`, and continues. Contract text touching
  indemnity, security or personal data can trip a classifier. A cell that
  refuses simply has no proposal — it is not distinguishable from an empty
  verdict unless you read the log.
- **Cache misses.** Every output row records `usage.cache_read`. If that is
  zero across a document's 2nd–26th calls, the caching is broken and you are
  paying 6× — stop the run and fix it before continuing.
- **Discarded spans.** Every row records a `discarded` list with a reason
  (`unknown_paragraph_id`, `quote_not_verbatim`). A high discard rate means the
  paragraph block and the frozen text have drifted apart, not that the model is
  bad.

### The runbook

```bash
export ANTHROPIC_API_KEY=...          # or: ant auth login
python scripts/propose_v3.py --doc DOC-015          # one document first — 26 calls
python scripts/propose_v3.py --all --k 1 --out work/rerun.jsonl
```

Do the single-document run first and inspect `work/proposals_frozen.jsonl`:
confirm `cache_read` is non-zero from the second call onward, confirm spans are
resolving, confirm nothing says `FAILED`. Then commit to the full run.

Hash the output before scoring it. `prompt_sha256` is already recorded on every
row, so the run is self-identifying — but the audit compares human verdicts
against a *frozen* file, and a file that is still being appended to is not one.

### What "similar" means, and how to measure it

**Do not diff the re-annotation against `qa/qa.json` and count mismatches.**
Two independent runs of this task disagree on some cells even when both are
defensible — that is a property of clause extraction, not a defect.

Score it the way the blind study scores humans: treat the release label as one
annotator and the re-annotation as another, and compute Cohen's κ on the
present/absent verdict, keyed on `intent_group_id`.

**The benchmark to compare against is already measured.** Two independent human
annotators, working blind from the same documents and boundary rules, agree
with each other at **κ = 0.7027** on the 44 core cells (§ *Agreement figures*).
That is the practical ceiling for this task:

| Re-annotation vs. release | Reading |
|---|---|
| κ ≈ 0.70 or above | Indistinguishable from two humans doing the task independently. The pipeline reproduces. |
| κ ≈ 0.40–0.70 | Comparable to the release's own agreement with each human annotator (0.396 / 0.294). Expected. |
| κ below 0.30 | Something is wrong with the setup, not the model — check the prompt loaded, the gates fired, and the discard rate. |

Report the κ, its n, and the sample composition — never a bare "matched X%".

---

## Running `--rebuild`

```bash
python scripts/reproduce.py --rebuild --dry-run   # list the stages
python scripts/reproduce.py --rebuild             # execute
```

Runs stages 7, 11, 14, 15, 16 and the validators, in dependency order. Every
destructive step leaves a `.bak` or an archive file beside the original. Then
re-run `--verify` to confirm the rebuild reproduces the release.

> ### ⚠ Seven scripts write on invocation and ignore `--help`
>
> `apply_category_scope.py`, `build_qa.py`, `run_validation.py`,
> `build_annotation_workspace.py`, `normalize_corpus.py`,
> `segment_paragraphs.py` and `build_categories.py` use no `argparse`. They
> **ignore argv entirely**, so `--help`, `--dry-run` and `--version` all
> silently run the stage for real.
>
> Worse, against an incomplete checkout they do not fail — they rewrite their
> report with zeros. Invoking `apply_category_scope.py --help` in a tree
> missing `annotations/annotations_reviewed.jsonl` overwrites
> `validation/category_scope_report.json` with `core_cells_applicable: 0`,
> destroying a recorded figure and reporting `status: PASS`. This contradicts
> the fail-loud discipline claimed in METHODOLOGY §3.8 and is worth fixing at
> source.
>
> `--verify` catches the damage on the next run (`expected 173, got 0`).

### `--rebuild` is currently blocked in this checkout

Stages 7 and 11 read `annotations/annotations_reviewed.jsonl`, which is **not
present**. `apply_category_scope.py` finds zero cells and `build_qa.py` reports
*"no annotation cells found — run build_annotation_workspace.py first"*. Until
that file is restored, only stages 14–16 can rebuild, and `--verify` is the
usable tier.

---

## Agreement figures: read before quoting them

The blind two-annotator study drew 60 cells at seed `20260816`. **16 of them are
negatives-split cells** that no longer exist in the release. Restricting to the
44 core cells — the only ones this dataset now contains — every coefficient
falls:

| Measure | All 60 cells (published) | 44 core cells |
|---|---:|---:|
| Annotator vs annotator, raw | 0.9000 | 0.8636 |
| Chance agreement | 0.4950 | 0.5413 |
| **Annotator vs annotator, κ** | **0.8020** | **0.7027** |
| Dataset vs Akshay Kumar, κ | 0.6284 | 0.3956 |
| Dataset vs Kanishka, κ | 0.5066 | 0.2936 |
| Dataset precision (Akshay Kumar / Kanishka) | 0.824 / 0.676 | 0.824 / 0.676 |
| Dataset recall (Akshay Kumar / Kanishka) | 0.848 / 0.852 | 0.875 / 0.885 |
| Label composition | 34 present · 26 absent | 34 present · 10 absent |

Precision is **identical** and recall slightly improves — the dataset's
behaviour did not change. What changed is the base rate. The 16 negatives cells
were near-unanimous true absences; dropping them takes the absent class from 26
to 10 and lifts chance agreement from 0.495 to 0.541. κ discounts by chance, so
it falls although no prediction changed.

**The 60-cell κ figures were partly carried by cells the dataset no longer
contains, and cannot be quoted alongside a core-only description of the
corpus.**

By attention tier on the core subset the effect is sharpest where n is smallest:
on the 13 core `low_friction` cells the dataset scores κ **−0.114** and
**−0.258** against the two annotators. On n = 13 that estimates nothing
reliably, but it cannot be reported as supporting the tier design either.

---

## Open items

Ordered by what blocks reproduction most.

1. **`annotations/annotations_reviewed.jsonl` is missing**, which blocks
   `--rebuild` at stages 7 and 11. Also absent and cited by the methodology as
   reproduction-required: `annotations/BLIND_SUBSAMPLE.jsonl`,
   `annotations/REVIEW_SHEET.md`, `annotations/gemini/`, `decisions.json`, and
   `ANNOTATION_PROTOCOL` Appendix A.

2. **The sealed blind key does not match the scored returns.** The local key
   holds 59 cells; the two CSVs hold 60; the sets share no identifiers.
   `blind_subsample_v2.py --score` raises `KeyError` here. The figures are
   sound — `--verify` recomputes them from the CSVs and reproduces the published
   run exactly — but the key must be recovered from the machine where `--emit`
   ran, or the study re-emitted at its recorded seed.

3. **`SNAPSHOT.txt` is stale.** 6 of its 50 entries have drifted, and it does
   not hash `qa/qa.json` at all — it pins the superseded `qa/qa.jsonl`.
   Current: `qa.json` `bb3202dd…`, `qa.jsonl` `4a786383…`.

4. **The validators still read `qa/qa.jsonl`, not `qa/qa.json`.**
   `span_report.json` and `query_consistency_report.json` report 3,225 rows over
   1,335 intent groups — the pre-P5, pre-drop file. Re-point them before quoting
   validator status for the current release.

5. **P5 has no written procedure.** The pass that took every cell from
   bulk-approved to individually human-read exists only in the data
   (`annotation_pass`, `annotator_ids`, `annotator_notes`). No script, no review
   order, no rule for when a reviewer finding was applied to the gold.

6. **Protocol §7 legal review — status disputed.** All 26 categories record
   `review_status: WAIVED_NOT_REVIEWED` with `reviewed_by: "Not reviewed —
   external counsel declined"`, and METHODOLOGY §7.4 and PROJECT_CONTEXT §6 say
   the same. **The project lead states a legal expert did review the categories
   before they were finalised.** If so the data is simply stale and must be
   corrected at source — `--verify` flags it on every run until it is:

   ```bash
   python scripts/approve_categories.py --reviewer "Name, Role" --date YYYY-MM-DD --notes "..."
   ```

   The script refuses to record an approval without a named reviewer, by
   design. Correcting this also affects the METHODOLOGY §3.2.1 argument, which
   currently explains the weak `low_friction` agreement partly by the boundary
   rules being less determinate than they look — a harder reading to sustain if
   a qualified practitioner signed those rules off first.

7. **Documentation figures predate the corpus.** `PROJECT_CONTEXT.md` §2 still
   reports the 10 Aug build (962 cells, 1,742 rows, 81.6% abstention). Live:
   507 cells, 1,521 rows, no abstention rows at all.

8. **`validation/` was partly recomputed on 2026-08-17.** Seven reports were
   regenerated against the live post-`drop_excluded` corpus and now reflect 58
   documents / 9,428 paragraphs rather than the pre-drop 72 / 10,443.
   `final_qc_report.json` reads `FATAL` because `source_metadata_report.json`
   fails on the missing files of item 1, not because of a data defect.

---

## Environment

Python 3.13, stdlib only for `--verify`. `--annotate` additionally needs
`anthropic`, `pydantic`, and `ANTHROPIC_API_KEY`.

Two known environment traps, both recorded in PROJECT_CONTEXT §12 and both still
live: **`pymupdf`'s native DLL is blocked by Windows Application Control**, so
the PDF stages (`inspect_sources.py`, `extract_pdf_text.py`) fail with an
`ImportError` — everything downstream of normalization works; and
**`normalize_corpus.py` wipes tier/split assignments**, so `apply_triage.py`
must run after it or every document loses its split.

## Layout

```
corpus/      documents.jsonl · paragraphs.jsonl · categories.jsonl
             excluded_*.jsonl — archived, not deleted
qa/          qa.json          THE RELEASE — 1,521 rows
             qa_unanswerable.json  the 1,704 dropped rows, annotation intact
             distractors.json   651 records × 3 slots
prompts/     extract_spans.v3.md · perturb_distractors.v1.md
work/normalized/   the frozen text every offset refers to
annotations/blind/ the two completed blind returns + sealed key
validation/  one JSON report per validator
scripts/     reproduce.py is the entry point; the rest are stages
```
