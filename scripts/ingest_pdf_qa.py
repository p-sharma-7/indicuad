"""Turn the JSON a chat model returns from prompts/qa_from_pdf.md into qa.json rows.

The model gives us quoted text only. This resolves each quote to a paragraph_id
and character offsets by exact match against the frozen corpus, then expands the
canonical row into the three question registers for core-split documents.

A quote that doesn't match exactly is reported and dropped — never fuzzy-aligned.

    python scripts/ingest_pdf_qa.py --in model_output.json --doc DOC-044
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTERS = ("canonical", "semi_formal", "naive_layperson")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def locate(quote: str, paras: list[dict]) -> list[dict]:
    """Every exact occurrence of `quote` across the document's paragraphs.

    Protocol rule 8: duplicated text is marked at every occurrence.
    """
    hits = []
    for p in paras:
        start = 0
        while (i := p["text"].find(quote, start)) != -1:
            abs_start = p["char_start"] + i
            hits.append({
                "text": quote,
                "answer_start": abs_start,
                "answer_end": abs_start + len(quote),
                "paragraph_id": p["paragraph_id"],
            })
            start = i + 1
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="model's JSON array")
    ap.add_argument("--doc", required=True, help="doc_id, e.g. DOC-044")
    ap.add_argument("--out", default=None, help="default work/<doc>_rows.json")
    ap.add_argument("--annotator", default="claude-opus-5 (Anthropic) via Claude Code")
    args = ap.parse_args()

    raw = Path(args.infile).read_text(encoding="utf-8").strip()
    if raw.startswith("```"):  # strip a fenced block if they pasted one
        raw = raw.split("```", 2)[1].split("\n", 1)[1].rsplit("```", 1)[0]
    proposals = json.loads(raw)

    cats = {c["category_id"]: c for c in load_jsonl(ROOT / "corpus" / "categories.jsonl")}
    paras = [p for p in load_jsonl(ROOT / "corpus" / "paragraphs.jsonl")
             if p["doc_id"] == args.doc]
    paras.sort(key=lambda p: p["paragraph_index"])
    if not paras:
        print(f"No paragraphs for {args.doc}. Normalize the PDF into the corpus first.")
        return 1

    frozen_path = ROOT / "work" / "normalized" / f"{args.doc}.txt"
    frozen = frozen_path.read_text(encoding="utf-8") if frozen_path.exists() else None

    rows, dropped, seq = [], [], None
    for prop in proposals:
        cat = cats.get(prop["category_id"])
        if cat is None:
            dropped.append((prop.get("category_id"), "unknown category_id"))
            continue

        resolved = []
        for ans in prop.get("answers") or []:
            quote = ans["text"] if isinstance(ans, dict) else ans
            hits = locate(quote, paras)
            if not hits:
                dropped.append((prop["category_id"], f"no exact match: {quote[:60]!r}"))
                continue
            for h in hits:
                if frozen and frozen[h["answer_start"]:h["answer_end"]] != quote:
                    dropped.append((prop["category_id"], "offset disagrees with frozen text"))
                    continue
                resolved.append(h)

        if seq is None:
            seq = int(prop["qa_id"].rsplit("-", 1)[1])

        registers = REGISTERS if prop.get("split") == "core" else ("canonical",)
        for reg in registers:
            question = {
                "canonical": cat["canonical_question"],
                "semi_formal": cat["semi_formal_question"],
                "naive_layperson": cat["naive_layperson_question"],
            }[reg]
            rows.append({
                "qa_id": f"IND-Q-{seq:05d}",
                "doc_id": args.doc,
                "category_id": prop["category_id"],
                "question": question,
                "question_register": reg,
                "intent_group_id": f"IG-{args.doc.replace('-', '')}-{prop['category_id'].replace('-', '')}",
                "answers": resolved,
                "is_impossible": not resolved,
                "gold_paragraph_ids": sorted({h["paragraph_id"] for h in resolved}),
                "doc_origin": prop.get("doc_origin", "collected_real"),
                "tier": prop.get("tier", "A"),
                "split": prop.get("split", "core"),
                "annotator_ids": [args.annotator],
                "annotation_pass": "P1-model-assisted",
                "model_assisted": True,
                "adjudicated": False,
                "jaccard_agreement": None,
                "annotator_notes": (
                    f"Proposed from PDF by {args.annotator} via prompts/qa_from_pdf.md"
                    f" (confidence: {prop.get('confidence', 'unstated')})."
                    f" {prop.get('notes', '')} Awaiting human verification."
                ).strip(),
            })
            seq += 1

    out = Path(args.out) if args.out else ROOT / "work" / f"{args.doc}_rows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(json.dumps(rows, indent=2, ensure_ascii=False))

    answered = sum(1 for r in rows if r["answers"])
    print(f"{len(proposals)} proposals -> {len(rows)} rows "
          f"({answered} answered, {len(rows)-answered} empty)")
    if frozen is None:
        print(f"  note: no work/normalized/{args.doc}.txt, offsets unverified")
    if dropped:
        print(f"  {len(dropped)} quote(s) dropped:")
        for cid, why in dropped:
            print(f"    {cid}: {why}")
    print(f"wrote {out}")
    print("These are proposals. Review before merging into qa/qa.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
