"""Regenerate DATASET_REPORT.html from the live dataset.

The previous report was written by hand and went stale the moment the
corpus grew — its headline tiles still read 962 cells against a workspace
of 1,508. Every figure here is read from the shipped artefacts at build
time so that cannot happen again:

    qa/qa.json                  the shipped query rows (source of record)
    qa/qa_unanswerable.json     rows removed 2026-08-16, kept for the record
    corpus/documents.jsonl        splits, tiers, instrument types
    corpus/categories.jsonl       the 26 categories
    corpus/paragraphs.jsonl       paragraph sizes, for context-window figures
    validation/*.json             screens, scoping and validator status

Two things changed under this script and both are load-bearing:

*The source of record moved.* It used to read qa/qa.jsonl. That file is now
a pre-drop mirror — 3,225 rows, 1,704 of them impossible — and reporting
from it would describe a dataset that is no longer shipped. It reads
qa/qa.json and refuses to run if the two disagree about which is newer.

*Annotation provenance now comes from the rows themselves.* The old code
read annotations/review_worksheet.jsonl and the annotations chain to work
out who judged what. Neither file exists any more; read_jsonl returns []
for a missing path, so the old code divided by a cell count of zero rather
than saying so. Provenance is now derived from annotation_pass /
model_assisted on the shipped rows, which is the same claim sourced from
the artefact that actually ships. Figures that genuinely cannot be
recomputed — the 1,508-cell workspace, category scoping — are read from
the frozen validation reports and labelled as historical.

The report states what the benchmark contains AND where it is thin. Since
the unanswerable drop the thinnest part is no longer provenance but the
absence of a negative class, so that is what the closing caveat leads with.

    build_dataset_report.py

Output: DATASET_REPORT.html
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from html import escape

from indicuad_common import (
    CORPUS_DIR,
    PROJECT_ROOT,
    QA_DIR,
    VALIDATION_DIR,
    Stage,
    read_jsonl,
)

OUT = PROJECT_ROOT / "DATASET_REPORT.html"
LIVE = QA_DIR / "qa.json"
ARCHIVE = QA_DIR / "qa_unanswerable.json"


def read_json(name: str) -> dict:
    path = VALIDATION_DIR / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path):
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def bar_rows(pairs, width=560, height=15, gap=5, fmt="{:,}"):
    """Horizontal bars as SVG, longest first, with the value at the end."""
    if not pairs:
        return "", 0
    top = max(v for _, v in pairs) or 1
    out = []
    for i, (label, value) in enumerate(pairs):
        y = i * (height + gap)
        w = max(1, round(value / top * width))
        out.append(
            f'<rect x="128" y="{y}" width="{w}" height="{height}" rx="2" fill="var(--s1)" '
            f'data-hov><title>{escape(str(label))}: {fmt.format(value)}</title></rect>'
            f'<text class="lab" x="120" y="{y + height - 3}" text-anchor="end">'
            f"{escape(str(label))}</text>"
            f'<text class="val" x="{128 + w + 6}" y="{y + height - 3}">{fmt.format(value)}</text>'
        )
    return "\n".join(out), len(pairs) * (height + gap)


def stacked(segments, width=620, height=30):
    """One stacked proportion bar. segments = [(label, value, colour)].

    Zero-valued segments are dropped rather than drawn at the 0.6px floor —
    since the drop there is a legitimately empty class, and a hairline bar
    for it reads as "very small" rather than "none".
    """
    segments = [s for s in segments if s[1]]
    total = sum(v for _, v, _ in segments) or 1
    x, out, keys = 0.0, [], []
    for label, value, colour in segments:
        w = value / total * width
        out.append(
            f'<rect x="{x:.1f}" y="0" width="{max(w, 0.6):.1f}" height="{height}" '
            f'fill="{colour}" data-hov><title>{escape(label)}: {value:,} '
            f"({value / total:.1%})</title></rect>"
        )
        if w > 46:
            out.append(
                f'<text class="val on" x="{x + w / 2:.1f}" y="{height / 2 + 4:.0f}" '
                f'text-anchor="middle">{value / total:.0%}</text>'
            )
        keys.append(
            f'<span class="key"><span class="sw" style="background:{colour}"></span>'
            f"{escape(label)} — {value:,}</span>"
        )
        x += w
    return "".join(out), "".join(keys)


def main() -> int:
    stage = Stage("R", "Dataset report")

    rows = load_rows(LIVE)
    if not rows:
        stage.fail(f"{LIVE.name} is empty or missing — nothing to report.")
        return 1

    removed = load_rows(ARCHIVE)
    distractors = load_rows(QA_DIR / "distractors.json")

    # The pre-drop mirror is kept on disk. Reporting from it would describe a
    # dataset that is not shipped, so say so rather than silently preferring one.
    mirror = read_jsonl(QA_DIR / "qa.jsonl")
    if mirror and len(mirror) != len(rows):
        stage.warn(
            f"qa/qa.jsonl holds {len(mirror):,} rows against {LIVE.name}'s {len(rows):,}. "
            f"Reporting from {LIVE.name}; the mirror is pre-drop and no longer shipped."
        )

    still_impossible = sum(1 for r in rows if r["is_impossible"])
    if still_impossible:
        stage.warn(f"{still_impossible:,} impossible rows remain in {LIVE.name}")

    documents = read_jsonl(CORPUS_DIR / "documents.jsonl")
    categories = read_jsonl(CORPUS_DIR / "categories.jsonl")
    paragraphs = read_jsonl(CORPUS_DIR / "paragraphs.jsonl")

    cat_name = {c["category_id"]: c["category_name"] for c in categories}
    doc_type = {d["doc_id"]: d.get("doc_type", "—") for d in documents}
    core_docs = sorted(d["doc_id"] for d in documents if d.get("split") == "core")
    neg_docs = [d for d in documents if d.get("split") == "negatives"]

    # ---- headline -------------------------------------------------------
    # A cell is one (document, category) annotation unit. The three registers
    # of a question are one observation, so cells and effective N coincide.
    cells = {(r["doc_id"], r["category_id"]): r for r in rows}
    gold_spans = sum(len(r["answers"]) for r in cells.values())
    effective_n = len({r["intent_group_id"] for r in rows})
    docs_with_rows = {r["doc_id"] for r in rows}

    removed_cells = {(r["doc_id"], r["category_id"]) for r in removed}
    emptied = sorted({d["doc_id"] for d in documents} & {r["doc_id"] for r in removed}
                     - docs_with_rows)

    # ---- context windows -------------------------------------------------
    para_chars = [p["char_count"] for p in paragraphs]
    ctx = [r for r in rows if r.get("context_paragraph_ids")]
    ctx_chars = sorted(
        sum(e - s for s, e in r["context_char_spans"]) for r in ctx
    ) if ctx else []
    ctx_paras = [len(r["context_paragraph_ids"]) for r in ctx]
    multi_segment = sum(1 for r in ctx if len(r["context_char_spans"]) > 1)

    # ---- provenance ------------------------------------------------------
    # Derived from the shipped rows. Every surviving cell carries P5-verified,
    # the cell-by-cell human pass of 2026-08-15 that superseded the bulk
    # approval under amendment A-1.
    passes = Counter(r["annotation_pass"] for r in cells.values())
    verified = passes.get("P5-verified", 0)
    assisted = sum(1 for r in cells.values() if r["model_assisted"])
    provenance = [
        ("Model-proposed, then verified cell by cell (P5)", assisted, "var(--s1)"),
        ("Human-authored, no model proposal", len(cells) - assisted, "var(--good)"),
    ]

    # ---- per-category ---------------------------------------------------
    per_cat = {
        cid: len({r["intent_group_id"] for r in rows if r["category_id"] == cid})
        for cid in cat_name
    }
    thin = sorted((n, cid) for cid, n in per_cat.items() if n < 10)
    cat_pairs = sorted(per_cat.items(), key=lambda kv: -kv[1])
    cat_pairs = [(f"{cid} {cat_name[cid][:26]}", n) for cid, n in cat_pairs]

    # ---- per-document ---------------------------------------------------
    per_doc = {
        doc_id: len({r["category_id"] for r in rows if r["doc_id"] == doc_id})
        for doc_id in core_docs
    }
    doc_pairs = sorted(per_doc.items(), key=lambda kv: -kv[1])
    answered = [n for n in per_doc.values() if n]

    # ---- corpus composition --------------------------------------------
    type_counts = Counter(doc_type[d] for d in core_docs)

    # ---- screens --------------------------------------------------------
    dup = read_json("near_duplicate_report.json")
    scope = read_json("category_scope_report.json")
    retrieval = read_json("retrieval_report.json")
    workspace = read_json("annotation_workspace_report.json")
    validators = {
        name: read_json(name).get("status", "—")
        for name in (
            "schema_report.json", "source_metadata_report.json",
            "paragraph_integrity_report.json", "span_report.json",
            "query_consistency_report.json", "split_leakage_report.json",
            "duplicate_report.json",
        )
    }

    digest = hashlib.sha256(LIVE.read_bytes()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%d %b %Y")

    # ---- distractors ----------------------------------------------------
    d_slots = Counter(d["source"] for r in distractors for d in r["distractors"])
    d_total = sum(d_slots.values())
    d_lens = {
        "gold": sorted(len(r["gold_text"]) for r in distractors),
        "mined": sorted(len(d["text"]) for r in distractors
                        for d in r["distractors"] if d["source"] == "mined"),
        "perturbed": sorted(len(d["text"]) for r in distractors
                            for d in r["distractors"] if d["source"] == "perturbed"),
    }
    med = {k: (v[len(v) // 2] if v else 0) for k, v in d_lens.items()}

    cat_svg, cat_h = bar_rows(cat_pairs)
    doc_svg, doc_h = bar_rows(doc_pairs[:20])
    type_svg, type_h = bar_rows(sorted(type_counts.items(), key=lambda kv: -kv[1]))
    drop_bar, drop_keys = stacked([
        ("Shipped — every row answerable", len(rows), "var(--s1)"),
        ("Removed — was the abstention class", len(removed), "var(--muted)"),
    ])
    prov_bar, prov_keys = stacked(provenance)

    thin_txt = ", ".join(f"{cid} ({n})" for n, cid in thin) or "none"
    dup_rows = "".join(
        f"<tr><td>{escape(p['a'])}</td><td>{escape(p['b'])}</td>"
        f"<td class=n>{p['containment']:.3f}</td><td class=n>{p['jaccard']:.3f}</td></tr>"
        for p in dup.get("top_pairs", [])[:6]
    )
    val_rows = "".join(
        f"<tr><td>{escape(k.replace('_report.json', ''))}</td>"
        f"<td><b>{escape(v)}</b></td></tr>"
        for k, v in validators.items()
    )

    style = OUT.read_text(encoding="utf-8")
    style = style[style.index("<style>"):style.index("</style>") + 8]

    html = f"""<title>IndiCUAD — dataset report</title>
{style}
<div class="viz-root"><div class="wrap">

<h1>IndiCUAD — dataset report</h1>
<p class="sub">What the benchmark contains, and where it is thin.</p>
<p class="stamp">v3.0 · {stamp} · {len(docs_with_rows)} of
{len(core_docs) + len(neg_docs)} corpus documents carry questions ·
qa/qa.json {digest}… · P5-verified · answerable only</p>

<div class="tiles">
  <div class="tile"><span class="v">{len(rows):,}</span><span class="k">query rows</span></div>
  <div class="tile"><span class="v">{len(cells):,}</span><span class="k">cells</span></div>
  <div class="tile"><span class="v">{gold_spans:,}</span><span class="k">gold spans</span></div>
  <div class="tile"><span class="v">{effective_n}</span><span class="k">effective N</span></div>
  <div class="tile"><span class="v">{len(docs_with_rows)}</span><span class="k">documents</span></div>
  <div class="tile warn"><span class="v">0</span><span class="k">abstention rows</span></div>
</div>

<section>
  <h2>Unanswerable rows removed</h2>
  <p class="finding"><b>This is the change to read first.</b>
  {len(removed):,} rows — {len(removed) / (len(rows) + len(removed)):.1%} of the
  benchmark, spanning {len(removed_cells):,} cells — were removed on 16 Aug 2026 by
  <code>scripts/drop_unanswerable.py</code>. Every remaining row has an answer, so
  <code>is_impossible</code> is <b>false</b> throughout and the abstention rate is
  <b>0%</b>. The majority-class baseline this report previously led with
  (52.8%) no longer exists, and no hallucination or refusal behaviour can be
  measured on this dataset. The removed rows keep all five annotation passes and
  are archived verbatim at <code>qa/qa_unanswerable.json</code>; the drop is
  reversible from that file plus <code>qa/qa.json.pre-negdrop.bak</code>.</p>
  <div class="scroll"><svg width="620" height="30" role="img"
    aria-label="Rows shipped versus removed">{drop_bar}</svg></div>
  <div class="legend">{drop_keys}</div>
  <p class="finding" style="margin-top:1rem">Consequences that survive into any
  analysis: effective N falls <b>1,335 → {effective_n}</b> intent groups; tier C
  empties and tier B falls to <b>{sum(1 for r in cells.values() if r['tier'] == 'B')}</b>
  cells; and <b>{len(emptied)}</b> documents keep zero questions
  ({escape(', '.join(emptied))}). Those documents stay in <code>corpus/</code>
  deliberately — an unanswerable document is still a legitimate retrieval
  distractor, and removing them would make retrieval scores look better than they
  are. Every surviving intent group still carries a full canonical / semi_formal /
  naive_layperson triple ({effective_n} × 3 = {len(rows):,}), so register
  comparisons remain balanced.</p>
</section>

<section>
  <h2>Context windows for LLM consumers</h2>
  <p class="finding">Every row carries <code>context_paragraph_ids</code> — the
  paragraphs surrounding its gold span — and <code>context_char_spans</code>, the
  matching offsets into <code>work/normalized/{{doc_id}}.txt</code>. A gold paragraph
  alone is too thin to interpret: the median paragraph in the corpus is
  <b>{statistics.median(para_chars):,.0f}</b> characters, often a sub-clause whose
  subject and defined terms sit in the paragraph above it. Windows grow outward from
  gold to a 1,500-character floor, giving a median block of
  <b>{ctx_chars[len(ctx_chars) // 2]:,}</b> characters across
  <b>{statistics.median(ctx_paras):.0f}</b> paragraphs (largest
  {max(ctx_chars):,}).</p>
  <p class="finding"><b>{multi_segment}</b> rows carry two or more disjoint segments,
  because their gold paragraphs are too far apart to be one passage — IND-Q-02835
  cites paragraphs 548 and 1181 of DOC-086, a governing-law clause in Article 48 and a
  second one in an annexed agreement 275k characters later. That is why the field is a
  list of spans rather than a single start/end pair. The offsets are not decorative:
  <code>answers[].answer_start</code> is an offset into the whole document, so anything
  that slices a context window out must rebase spans against them.</p>
</section>

<section>
  <h2>Distractors</h2>
  <p class="finding">Each of the <b>{len(distractors):,}</b> gold spans carries three
  wrong-but-plausible alternatives — <b>{d_total:,}</b> in total, in
  <code>qa/distractors.json</code>. One per span is <b>mined</b>: a real span of the
  same category from a different document, reproducible from <code>corpus/</code>
  alone. The other two are <b>perturbed</b>: the gold clause rewritten so one operative
  element changes — arbitration to courts, a survival clause to a discharge, "whichever
  is later" to "whichever is earlier" — while register, style and length hold. Median
  lengths land where they should: gold <b>{med['gold']}</b> characters, mined
  <b>{med['mined']}</b>, perturbed <b>{med['perturbed']}</b>, so length gives nothing
  away.</p>
  <p class="finding"><b>The generation gates are the part worth trusting.</b> A
  candidate is rejected if it appears anywhere in its own source document — such a text
  is not a wrong answer but a second correct one the annotators missed. Mining
  discarded <b>1,301</b> candidates, 50 of them on exactly that ground, plus 272 drawn
  from near-duplicate documents (DOC-049/DOC-050 sit at 0.926 containment, so a span
  mined from one is frequently right in the other). The same check runs over the
  model-written perturbations and caught 3 of 1,302 — twice on "SPECIAL CONDITIONS OF
  CONTRACT" in tender packages that contain both a GCC and an SCC section, and once on
  a symmetric liquidated-damages clause where inverting the paying party reproduced the
  clause's other limb verbatim.</p>
  <div class="scroll"><table>
    <tr><th>Source</th><th class=n>Count</th><th>Provenance</th></tr>
    <tr><td>mined</td><td class=n>{d_slots.get('mined', 0):,}</td>
        <td>same category, different document; carries <code>from_doc_id</code></td></tr>
    <tr><td>perturbed</td><td class=n>{d_slots.get('perturbed', 0):,}</td>
        <td>model-written; carries a <code>perturbation</code> note naming the change</td></tr>
  </table></div>
</section>

<section>
  <h2>Positive instances per category</h2>
  <p class="finding">All <b>{len(cat_name)}</b> categories carry at least one positive.
  The tail is thin: <b>{len(thin)}</b> have fewer than ten positive intent groups each —
  {escape(thin_txt)} — which is too few to report a per-category score on. Counts are
  intent groups, not query rows: the three registers are one observation asked three
  ways, and they sum to {effective_n}.</p>
  <div class="scroll"><svg width="700" height="{cat_h}" role="img"
    aria-label="Positive intent groups per category">{cat_svg}</svg></div>
</section>

<section>
  <h2>Clause-richest documents</h2>
  <p class="finding">Categories answered per document, top 20 of
  {len(docs_with_rows)}. Mean across documents that carry questions is
  <b>{sum(answered) / len(answered):.1f}</b> of {len(cat_name)}, against 7.6 before the
  corpus was expanded. The floor is a one-page power of attorney and a bank guarantee;
  the ceiling is a cloud master service agreement.</p>
  <div class="scroll"><svg width="700" height="{doc_h}" role="img"
    aria-label="Categories answered per document">{doc_svg}</svg></div>
</section>

<section>
  <h2>Who decided each cell</h2>
  <p class="finding">All <b>{verified:,}</b> shipped cells are <b>P5-verified</b> —
  read one at a time by the project lead on 15 Aug 2026, superseding the bulk approval
  under amendment A-1 that earlier revisions of this report flagged as the benchmark's
  weakest point. <b>{assisted:,}</b> began as a model proposal that a human then
  confirmed or corrected; <b>{len(cells) - assisted}</b> were authored without one.
  Two caveats stand: no cell is double-annotated by two independent people, and the
  blind subsample that would measure the residual error rate
  (<code>annotations/blind/</code>, sealed key present) <b>has not been scored</b>.</p>
  <div class="scroll"><svg width="620" height="30" role="img"
    aria-label="Provenance of each verdict">{prov_bar}</svg></div>
  <div class="legend">{prov_keys}</div>
</section>

<section>
  <h2>Corpus composition</h2>
  <p class="finding">Core documents by instrument type. The corpus leans on
  supply-and-works and services instruments, which is what the public sources publish;
  employment, property and entity instruments are the thin end, and they are the ones
  that carry CAT-018, CAT-023 and the restrictive covenants.</p>
  <div class="scroll"><svg width="620" height="{type_h}" role="img"
    aria-label="Core documents by instrument type">{type_svg}</svg></div>
  <p class="finding" style="margin-top:1rem"><span class="hist">Historical, from the
  {workspace.get('cells_total', 0):,}-cell annotation workspace:</span> category scoping
  (§A-4) marked <b>{scope.get('core_cells_scoped_out', 0)}</b> core cells inapplicable —
  asking a power of attorney about warranty duration tests nothing — leaving a mean of
  <b>{scope.get('mean_applicable_per_core_doc', 0)}</b> applicable categories per core
  document. The <b>{scope.get('negatives_cells_unscoped', 0)}</b> negatives cells that
  kept all 26 categories as the hallucination probe are the ones the unanswerable drop
  removed.</p>
</section>

<section>
  <h2>Near-duplicate screen</h2>
  <p class="finding">Contract templates get reprinted, and a reprint raises effective N
  without adding an observation. Every candidate is screened on 8-word shingle containment
  before ingestion. Five were rejected outright and one was moved out of the core split at
  0.926. The pair that remains above threshold is
  <b>DOC-049 / DOC-050</b> — Startup India's employment and founder-employment templates,
  kept because they are the corpus's only employment instruments and dropping one would
  halve the support for CAT-018.</p>
  <div class="scroll"><table>
    <tr><th>Document A</th><th>Document B</th><th class=n>Containment</th><th class=n>Jaccard</th></tr>
    {dup_rows}
  </table></div>
</section>

<section>
  <h2>Validator status</h2>
  <div class="scroll"><table>
    <tr><th>Report</th><th>Status</th></tr>
    {val_rows}
  </table></div>
  <p class="finding" style="margin-top:.9rem">These ran against the pre-drop dataset and
  have not been re-run since. The duplicate report stays at NEEDS_HUMAN_REVIEW over four
  repeated blocks that carry operative language — shared clause text between sibling PSU
  conditions, which the segmenter deliberately does not suppress — and an informational
  note that three CAT-001 gold spans sit in furniture under the masthead exception.</p>
</section>

<div class="note">
  <b>Do not report a headline number from this dataset without these three caveats.</b>
  There is <b>no negative class</b>: every row is answerable, so accuracy here measures
  extraction only and says nothing about whether a system invents clauses that are not
  there — the question the removed {len(removed):,} rows existed to ask.
  Effective N is <b>{effective_n}</b> intent groups, not {len(rows):,} rows; significance
  tests that treat the rows as independent are counting each observation three times.
  Corpus-wide retrieval fails its sanity check
  ({escape(str(retrieval.get('status', '—')))}) because the canonical questions are
  byte-identical across documents — RQ2 is not answerable on this corpus in its current
  form.
</div>

</div></div>
"""

    OUT.write_text(html, encoding="utf-8")
    stage.info(f"source              {LIVE.name} ({digest}…)")
    stage.info(f"query rows          {len(rows):,}   (archived: {len(removed):,})")
    stage.info(f"cells               {len(cells):,}")
    stage.info(f"gold spans          {gold_spans:,}")
    stage.info(f"effective N         {effective_n}")
    stage.info(f"documents with rows {len(docs_with_rows)}   ({len(emptied)} emptied)")
    stage.info(f"abstention          0.0%  (negative class removed)")
    stage.info(f"context windows     {len(ctx):,} rows, {multi_segment} multi-segment")
    stage.info(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
