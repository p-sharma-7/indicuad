"""Render the proposal worksheet as a document a human can actually read.

`annotations/review_worksheet.jsonl` is the file the reviewer edits, but
JSONL is a poor surface for legal judgement: the reviewer needs the
category definition, the boundary rule, the proposed span and the reason
side by side. This renders exactly that, grouped by document, and marks
the cells where the proposal is flagged contestable.

Reading happens here; verdicts still get recorded in the worksheet.

Outputs:
    annotations/REVIEW_SHEET.md    for the repo
    annotations/review_sheet.html  for a reviewer without an editor
"""

from __future__ import annotations

import html
import sys
from collections import defaultdict

from indicuad_common import ANNOTATIONS_DIR, CORPUS_DIR, Stage, read_jsonl

TYPE_LABEL = {
    "PRIMARY_LABEL": "proposed label",
    "RECALL_CANDIDATE": "candidate only — not a label",
    "NO_CANDIDATE_FOUND": "nothing found — you must confirm",
}

VERDICT_HELP = {
    "PRIMARY_LABEL": "`ACCEPT` if the span is right · `EDIT` to correct it · `EMPTY` if there is no clause",
    "RECALL_CANDIDATE": "`EDIT` with the span you choose · `EMPTY` if this is not the category",
    "NO_CANDIDATE_FOUND": "`EMPTY` to confirm · `EDIT` if you find a clause. `ACCEPT` is refused here.",
}


def main() -> int:
    stage = Stage("A/S", "Render review sheet")

    rows = read_jsonl(ANNOTATIONS_DIR / "review_worksheet.jsonl")
    if not rows:
        stage.fail("no review_worksheet.jsonl — run review_proposals.py first.")
        return 1

    proposals = {p["cell_id"]: p for p in read_jsonl(ANNOTATIONS_DIR / "annotations_proposed.jsonl")}
    categories = {c["category_id"]: c for c in read_jsonl(CORPUS_DIR / "categories.jsonl")}
    documents = {d["doc_id"]: d for d in read_jsonl(CORPUS_DIR / "documents.jsonl")}

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[row["doc_id"]].append(row)

    def contestable(row: dict) -> bool:
        text = row.get("rationale", "").lower()
        return "reviewer:" in text or "boundary question" in text

    flagged = sum(1 for r in rows if contestable(r))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["proposal_type"]] += 1

    out: list[str] = []
    out.append("# IndiCUAD — model-assisted proposal review\n")
    out.append(
        f"**{len(rows)} cells** to review across **{len(by_doc)} core documents** — "
        f"{counts['PRIMARY_LABEL']} proposed labels, "
        f"{counts['RECALL_CANDIDATE']} candidates, "
        f"{counts['NO_CANDIDATE_FOUND']} nothing-found. "
        f"**{flagged}** are flagged contestable.\n"
    )
    out.append(
        "> These are proposals, not annotations. Nothing here is gold until you set a "
        "verdict in `annotations/review_worksheet.jsonl` and run "
        "`review_proposals.py --promote --reviewer \"Name, Role\"`.\n"
    )
    out.append(
        "> `ACCEPT` on a *nothing found* row is refused by the tool. Confirming a cell "
        "empty is a judgement only you can make — those cells are what the benchmark "
        "measures hallucination against.\n"
    )
    out.append(
        "\nThis covers the 7 low-friction categories plus recall pointers. "
        "**133 core cells and all 572 negatives cells are not here** — they are "
        "annotated by hand.\n"
    )

    out.append("\n## Settle these first\n")
    out.append(
        "Three documents state liability as an *uncapped indemnity or total "
        "exclusion*, not a monetary cap: `DOC-017xCAT-019`, `DOC-019xCAT-019`, "
        "`DOC-022xCAT-019`. Decide once whether CAT-019 covers exclusions, then apply "
        "it to all three — an inconsistent call makes that category's numbers "
        "meaningless.\n"
    )

    for doc_id in sorted(by_doc):
        document = documents[doc_id]
        out.append(f"\n---\n\n## {doc_id} — {document['filename']}\n")
        out.append(f"*Tier {document['tier']} · {document['char_count']:,} chars*\n")

        for row in sorted(by_doc[doc_id], key=lambda r: r["category_id"]):
            category = categories[row["category_id"]]
            marker = " ⚠️" if contestable(row) else ""

            out.append(
                f"\n### {row['category_id']} — {row['category_name']}{marker}\n"
            )
            out.append(
                f"`{row['cell_id']}` · {row['attention_tier']} · "
                f"**{TYPE_LABEL[row['proposal_type']]}**\n"
            )
            out.append(f"\n*Boundary rule:* {category['boundary_rule']}\n")

            spans = row.get("proposed_spans") or []
            if spans:
                answers = proposals[row["cell_id"]]["answers"]
                out.append("\n")
                for span, answer in zip(spans, answers):
                    clean = " ".join(span.split())
                    out.append(
                        f"> {clean}\n>\n> — `{answer['paragraph_id']}` "
                        f"[{answer['answer_start']}, {answer['answer_end']})\n\n"
                    )
            else:
                out.append("\n*No span proposed.*\n\n")

            out.append(f"*Why:* {row['rationale']}\n")
            out.append(f"\n*Verdict:* {VERDICT_HELP[row['proposal_type']]}\n")

    path = ANNOTATIONS_DIR / "REVIEW_SHEET.md"
    path.write_text("".join(out), encoding="utf-8")

    html_path = ANNOTATIONS_DIR / "review_sheet.html"
    html_path.write_text(
        render_html(by_doc, proposals, categories, documents, contestable, counts, flagged),
        encoding="utf-8",
    )

    stage.info(f"wrote {path.name} — {len(rows)} cells, {flagged} flagged contestable")
    stage.info(f"wrote {html_path.name}")
    return 0


CSS = """
:root{
  --ground:#F5F6F9; --surface:#FFFFFF; --sunk:#EDEFF4;
  --ink:#191C23; --ink-soft:#5B6274; --ink-faint:#868DA0;
  --rule:#DCE0E8; --rule-soft:#E9ECF2;
  --accent:#33468F; --accent-soft:#E7EAF5;
  --flag:#8F5300; --flag-soft:#FBF1E0; --flag-rule:#E0B978;
  --label:#1F6B5A; --label-soft:#E2EFEB;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#111319; --surface:#181B22; --sunk:#1F232C;
    --ink:#E7EAF0; --ink-soft:#9BA4B5; --ink-faint:#727B8D;
    --rule:#2A2F3A; --rule-soft:#22262F;
    --accent:#93A5EA; --accent-soft:#1D2338;
    --flag:#E3A85C; --flag-soft:#291F12; --flag-rule:#6B4E22;
    --label:#6FC3AD; --label-soft:#13251F;
  }
}
:root[data-theme="dark"]{
  --ground:#111319; --surface:#181B22; --sunk:#1F232C;
  --ink:#E7EAF0; --ink-soft:#9BA4B5; --ink-faint:#727B8D;
  --rule:#2A2F3A; --rule-soft:#22262F;
  --accent:#93A5EA; --accent-soft:#1D2338;
  --flag:#E3A85C; --flag-soft:#291F12; --flag-rule:#6B4E22;
  --label:#6FC3AD; --label-soft:#13251F;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:16px; line-height:1.65;
}
.wrap{max-width:52rem;margin:0 auto;padding:3rem 1.5rem 6rem}
h1{
  font-family:Georgia,"Iowan Old Style","Times New Roman",serif;
  font-weight:600; font-size:2rem; line-height:1.2; margin:0 0 .75rem;
  text-wrap:balance; letter-spacing:-.01em;
}
h2{
  font-family:Georgia,"Iowan Old Style",serif; font-weight:600;
  font-size:1.3rem; margin:0 0 .2rem; text-wrap:balance;
}
h3{font-size:1rem;font-weight:650;margin:0;text-wrap:balance}
.lede{color:var(--ink-soft);margin:0 0 2rem;font-size:1.05rem}
.lede b{color:var(--ink);font-variant-numeric:tabular-nums}
.notice{
  background:var(--surface); border:1px solid var(--rule);
  border-left:3px solid var(--accent); border-radius:3px;
  padding:1rem 1.15rem; margin:0 0 1rem; color:var(--ink-soft); font-size:.95rem;
}
.notice strong{color:var(--ink)}
.notice.warn{border-left-color:var(--flag-rule);background:var(--flag-soft)}
.scope{color:var(--ink-faint);font-size:.9rem;margin:1.5rem 0 2.5rem}
.docnav{
  display:flex;flex-wrap:wrap;gap:.4rem;
  padding:1rem 0 2rem;border-bottom:1px solid var(--rule);margin-bottom:2.5rem;
}
.docnav a{
  font:600 .8rem/1 ui-monospace,"SF Mono",Menlo,monospace;
  color:var(--accent); background:var(--accent-soft);
  padding:.45rem .6rem;border-radius:3px;text-decoration:none;
}
.docnav a:hover,.docnav a:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.doc{margin:0 0 3.5rem;scroll-margin-top:1rem}
.doc-head{border-bottom:2px solid var(--ink);padding-bottom:.6rem;margin-bottom:1.75rem}
.doc-meta{
  font:.8rem/1.5 ui-monospace,"SF Mono",Menlo,monospace;
  color:var(--ink-faint);font-variant-numeric:tabular-nums;
}
.cell{
  background:var(--surface);border:1px solid var(--rule);border-radius:4px;
  padding:1.25rem 1.4rem;margin:0 0 1rem;
}
.cell.flagged{border-color:var(--flag-rule)}
.cell-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:.5rem;margin-bottom:.75rem}
.chip{
  font:600 .68rem/1 ui-monospace,"SF Mono",Menlo,monospace;
  text-transform:uppercase;letter-spacing:.06em;
  padding:.35rem .5rem;border-radius:3px;white-space:nowrap;
}
.chip.label{color:var(--label);background:var(--label-soft)}
.chip.cand{color:var(--accent);background:var(--accent-soft)}
.chip.none{color:var(--ink-faint);background:var(--sunk)}
.chip.flag{color:var(--flag);background:var(--flag-soft)}
.cellid{
  font:.75rem/1 ui-monospace,"SF Mono",Menlo,monospace;
  color:var(--ink-faint);margin-left:auto;
}
.rule-line{font-size:.9rem;color:var(--ink-soft);margin:0 0 .9rem}
.rule-line em{color:var(--ink-faint);font-style:normal;
  text-transform:uppercase;font-size:.7rem;letter-spacing:.07em;margin-right:.4rem}
blockquote.span{
  margin:0 0 .75rem;padding:.7rem .9rem;background:var(--sunk);
  border-left:2px solid var(--accent);border-radius:0 3px 3px 0;
  font-family:Georgia,"Iowan Old Style",serif;font-size:.95rem;
  overflow-x:auto;
}
blockquote.span .cite{
  display:block;margin-top:.5rem;
  font:.72rem/1.4 ui-monospace,"SF Mono",Menlo,monospace;
  color:var(--ink-faint);font-variant-numeric:tabular-nums;font-style:normal;
}
.nospan{color:var(--ink-faint);font-style:italic;font-size:.9rem;margin:0 0 .75rem}
.why{font-size:.92rem;color:var(--ink-soft);margin:0 0 .9rem}
.verdict{
  border-top:1px solid var(--rule-soft);padding-top:.7rem;
  font-size:.85rem;color:var(--ink-faint);
}
code{
  font:.85em ui-monospace,"SF Mono",Menlo,monospace;
  background:var(--sunk);padding:.1em .35em;border-radius:2px;
}
a{color:var(--accent)}
"""


def code_spans(text: str) -> str:
    """Markdown `backticks` to <code>, escaping everything else."""
    parts = text.split("`")
    return "".join(
        html.escape(part) if index % 2 == 0 else f"<code>{html.escape(part)}</code>"
        for index, part in enumerate(parts)
    )


def render_html(by_doc, proposals, categories, documents, contestable, counts, flagged) -> str:
    """Same content as the Markdown, laid out for reading on screen."""
    e = html.escape
    chip = {
        "PRIMARY_LABEL": ("label", "proposed label"),
        "RECALL_CANDIDATE": ("cand", "candidate only"),
        "NO_CANDIDATE_FOUND": ("none", "nothing found"),
    }

    p: list[str] = []
    p.append("<title>IndiCUAD — proposal review</title>")
    p.append(f"<style>{CSS}</style>")
    p.append('<div class="wrap">')
    p.append("<h1>IndiCUAD — model-assisted proposal review</h1>")
    p.append(
        f'<p class="lede"><b>{sum(counts.values())}</b> cells across '
        f'<b>{len(by_doc)}</b> core documents — <b>{counts["PRIMARY_LABEL"]}</b> proposed '
        f'labels, <b>{counts["RECALL_CANDIDATE"]}</b> candidates, '
        f'<b>{counts["NO_CANDIDATE_FOUND"]}</b> nothing-found. '
        f'<b>{flagged}</b> flagged contestable.</p>'
    )
    p.append(
        '<p class="notice">These are <strong>proposals, not annotations</strong>. '
        "Nothing here is gold until you set a verdict in "
        "<code>annotations/review_worksheet.jsonl</code> and run "
        "<code>review_proposals.py --promote</code>.</p>"
    )
    p.append(
        '<p class="notice warn"><strong>ACCEPT on a “nothing found” row is refused '
        "by the tool.</strong> Confirming a cell empty is a judgement only you can make — "
        "those cells are what the benchmark measures hallucination against.</p>"
    )
    p.append(
        '<p class="notice"><strong>Settle this first.</strong> Three documents state '
        "liability as an uncapped indemnity or total exclusion rather than a monetary cap: "
        "<code>DOC-017×CAT-019</code>, <code>DOC-019×CAT-019</code>, "
        "<code>DOC-022×CAT-019</code>. Decide once whether CAT-019 covers exclusions, then "
        "apply it to all three — an inconsistent call makes that category's numbers "
        "meaningless.</p>"
    )
    p.append(
        '<p class="scope">Covers the 7 low-friction categories plus recall pointers. '
        "133 core cells and all 572 negatives cells are not here — those are annotated "
        "by hand.</p>"
    )

    p.append('<nav class="docnav">')
    for doc_id in sorted(by_doc):
        p.append(f'<a href="#{doc_id}">{doc_id} · {len(by_doc[doc_id])}</a>')
    p.append("</nav>")

    for doc_id in sorted(by_doc):
        document = documents[doc_id]
        p.append(f'<section class="doc" id="{doc_id}">')
        p.append('<div class="doc-head">')
        p.append(f"<h2>{e(doc_id)}</h2>")
        p.append(
            f'<div class="doc-meta">{e(document["filename"])} · tier '
            f'{e(document["tier"])} · {document["char_count"]:,} chars</div>'
        )
        p.append("</div>")

        for row in sorted(by_doc[doc_id], key=lambda r: r["category_id"]):
            category = categories[row["category_id"]]
            is_flagged = contestable(row)
            kind, text = chip[row["proposal_type"]]

            p.append(f'<article class="cell{" flagged" if is_flagged else ""}">')
            p.append('<div class="cell-head">')
            p.append(f'<h3>{e(row["category_id"])} — {e(row["category_name"])}</h3>')
            p.append(f'<span class="chip {kind}">{text}</span>')
            if is_flagged:
                p.append('<span class="chip flag">contestable</span>')
            p.append(f'<span class="cellid">{e(row["cell_id"])}</span>')
            p.append("</div>")
            p.append(
                f'<p class="rule-line"><em>Boundary</em>{e(category["boundary_rule"])}</p>'
            )

            spans = row.get("proposed_spans") or []
            if spans:
                for span, answer in zip(spans, proposals[row["cell_id"]]["answers"]):
                    p.append(
                        f'<blockquote class="span">{e(" ".join(span.split()))}'
                        f'<span class="cite">{e(answer["paragraph_id"])} · '
                        f'[{answer["answer_start"]}, {answer["answer_end"]})</span>'
                        "</blockquote>"
                    )
            else:
                p.append('<p class="nospan">No span proposed.</p>')

            p.append(f'<p class="why">{e(row["rationale"])}</p>')
            p.append(
                f'<p class="verdict">{code_spans(VERDICT_HELP[row["proposal_type"]])}</p>'
            )
            p.append("</article>")
        p.append("</section>")

    p.append("</div>")
    return "\n".join(p)


if __name__ == "__main__":
    sys.exit(main())
