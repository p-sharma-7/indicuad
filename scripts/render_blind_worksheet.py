"""Render the blind subsample as a single self-contained working document.

The sample draws cells from across the corpus, so answering it naively means
opening twenty-odd files and hunting for the same passage repeatedly. This
groups the cells by document and prints the document's full text once above
its questions, turning the job into "read each document once".

The model's verdicts are NOT in this file, and must not be — the whole
point of the exercise is that the reader reaches their own conclusion first.
The output deliberately contains no hint of what the model decided, not even
the count of positives, because knowing "3 of these 5 are positive" would
anchor the answers.

Record decisions in annotations/BLIND_SUBSAMPLE.jsonl, then score with
blind_subsample.py --score.

Output: annotations/BLIND_WORKSHEET.html
"""

from __future__ import annotations

import html
import sys
from collections import defaultdict

from indicuad_common import (
    ANNOTATIONS_DIR,
    NORMALIZED_DIR,
    Stage,
    read_jsonl,
)

CSS = """
:root{
  --paper:#FDFDFC;--surface:#FFF;--sunk:#F2F3F1;
  --ink:#17191E;--ink-soft:#4E545E;--ink-faint:#828892;
  --rule:#DEDFDC;--accent:#35506B;--accent-soft:#E8ECF1;
  --ask:#7E5F14;--ask-soft:#FAF4E4;--ask-rule:#D8BE79;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#131519;--surface:#191C21;--sunk:#1E2228;
  --ink:#E6E8EC;--ink-soft:#A0A7B2;--ink-faint:#767D89;
  --rule:#2A2E36;--accent:#8FAAC8;--accent-soft:#1B2531;
  --ask:#D9B871;--ask-soft:#2A2314;--ask-rule:#6B5722;
}}
:root[data-theme="dark"]{
  --paper:#131519;--surface:#191C21;--sunk:#1E2228;
  --ink:#E6E8EC;--ink-soft:#A0A7B2;--ink-faint:#767D89;
  --rule:#2A2E36;--accent:#8FAAC8;--accent-soft:#1B2531;
  --ask:#D9B871;--ask-soft:#2A2314;--ask-rule:#6B5722;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:52rem;margin:0 auto;padding:2.5rem 1.5rem 6rem}
h1{font:650 1.7rem/1.2 Georgia,serif;margin:0 0 .5rem}
h2{font:650 1.15rem/1.3 Georgia,serif;margin:0 0 .2rem}
.meta{font:.75rem/1.5 ui-monospace,Menlo,monospace;color:var(--ink-faint);
 text-transform:uppercase;letter-spacing:.06em}
.note{background:var(--surface);border:1px solid var(--rule);
 border-left:3px solid var(--accent);border-radius:2px;padding:.9rem 1.1rem;
 margin:1rem 0;font-size:.93rem;color:var(--ink-soft)}
.note strong{color:var(--ink)}
.doc{margin:0 0 3.5rem;border-top:2px solid var(--ink);padding-top:1rem}
.src{background:var(--sunk);border:1px solid var(--rule);border-radius:3px;
 padding:1rem 1.15rem;margin:.9rem 0 1.4rem;max-height:30rem;overflow:auto;
 font:.82rem/1.55 ui-monospace,"SF Mono",Menlo,monospace;white-space:pre-wrap}
.q{background:var(--ask-soft);border:1px solid var(--ask-rule);
 border-radius:3px;padding:.9rem 1.1rem;margin:.8rem 0}
.q .cid{font:.72rem/1 ui-monospace,Menlo,monospace;color:var(--ask);
 text-transform:uppercase;letter-spacing:.06em}
.q h3{font-size:1rem;margin:.35rem 0 .5rem}
.q .ask{font-size:.95rem;margin:.4rem 0}
.q dl{margin:.6rem 0 0;font-size:.86rem;color:var(--ink-soft)}
.q dt{font-weight:650;color:var(--ink);margin-top:.45rem;font-size:.74rem;
 text-transform:uppercase;letter-spacing:.05em}
.q dd{margin:.15rem 0 0}
.ansbox{margin-top:.7rem;padding-top:.6rem;border-top:1px dashed var(--ask-rule);
 font:.82rem/1.5 ui-monospace,Menlo,monospace;color:var(--ink-soft)}
code{font:.86em ui-monospace,Menlo,monospace;background:var(--sunk);
 padding:.1em .3em;border-radius:2px}
"""


def main() -> int:
    stage = Stage("A-5w", "Render blind worksheet")

    rows = read_jsonl(ANNOTATIONS_DIR / "BLIND_SUBSAMPLE.jsonl")
    if not rows:
        stage.fail("no BLIND_SUBSAMPLE.jsonl — run blind_subsample.py --emit first")
        return 1

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[row["doc_id"]].append(row)

    e = html.escape
    p: list[str] = []
    p.append("<title>IndiCUAD — blind validation worksheet</title>")
    p.append(f"<style>{CSS}</style>")
    p.append('<div class="wrap">')
    p.append("<h1>Blind validation worksheet</h1>")
    p.append(
        f'<p class="meta">{len(rows)} cells across {len(by_doc)} documents · '
        "model verdicts withheld</p>"
    )
    p.append(
        '<p class="note"><strong>Read the document, then answer its questions '
        "from what you see.</strong> Nothing here tells you what the model "
        "decided — not the answers, not how many are positive — because knowing "
        "that would anchor your reading and destroy the measurement.</p>"
    )
    p.append(
        '<p class="note">For each question record two things in '
        "<code>annotations/BLIND_SUBSAMPLE.jsonl</code>: "
        "<code>your_is_impossible</code> — <code>true</code> if the document has "
        "no such clause, <code>false</code> if it does — and, when false, the "
        "clause text in <code>your_spans</code>. Then run "
        "<code>blind_subsample.py --score</code>.</p>"
    )

    for doc_id in sorted(by_doc):
        cells = sorted(by_doc[doc_id], key=lambda r: r["category_id"])
        try:
            text = (NORMALIZED_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
        except OSError:
            text = "(source text unavailable)"

        p.append('<section class="doc">')
        p.append(f"<h2>{e(doc_id)}</h2>")
        p.append(
            f'<p class="meta">{len(text):,} characters · {len(cells)} '
            f'question{"s" if len(cells) != 1 else ""}</p>'
        )
        p.append(f'<div class="src">{e(text)}</div>')

        for cell in cells:
            p.append('<div class="q">')
            p.append(f'<div class="cid">{e(cell["cell_id"])} · {e(cell["attention_tier"])}</div>')
            p.append(f'<h3>{e(cell["category_id"])} — {e(cell["category_name"])}</h3>')
            p.append(f'<p class="ask">{e(cell["question"])}</p>')
            p.append("<dl>")
            p.append(f"<dt>Boundary rule</dt><dd>{e(cell['boundary_rule'])}</dd>")
            negatives = cell.get("negative_examples") or []
            if negatives:
                p.append(
                    "<dt>Does not count</dt><dd>"
                    + "; ".join(e(str(x)) for x in negatives)
                    + "</dd>"
                )
            p.append("</dl>")
            p.append(
                '<div class="ansbox">your_is_impossible: ____ &nbsp;·&nbsp; '
                "your_spans: [ ____ ]</div>"
            )
            p.append("</div>")
        p.append("</section>")

    p.append("</div>")

    path = ANNOTATIONS_DIR / "BLIND_WORKSHEET.html"
    path.write_text("\n".join(p), encoding="utf-8")
    stage.info(f"wrote {path.name} — {len(rows)} cells, {len(by_doc)} documents")
    stage.info("model verdicts are not present in this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
