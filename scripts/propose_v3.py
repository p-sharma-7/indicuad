"""Span proposal for IndiCUAD v2 — one document × one category per call.

Design notes that matter:

* The model NEVER emits character offsets. It emits `paragraph_id` + a verbatim
  `quote`, and `resolve_span()` computes offsets deterministically. LLMs cannot
  count characters reliably; paragraph anchoring removes the failure mode
  entirely instead of trying to prompt around it.
* A quote that is not an exact substring of its paragraph is DISCARDED. Never
  fuzzy-align a model span into gold (ANNOTATION_PROTOCOL, Appendix A).
* `temperature` / `top_p` / `top_k` return 400 on Claude Opus 5 — sampling
  parameters were removed. Self-consistency therefore comes from repeated
  sampling (generation is still stochastic) and from varying `effort`, not from
  a temperature sweep.
* Prompt caching: the instructions and the document body are a stable prefix
  reused across all 26 category calls for a document, so 25 of every 26 calls
  read the cache instead of re-paying for the contract.

Usage:
    python scripts/propose_v3.py --doc DOC-014
    python scripts/propose_v3.py --all --k 3 --out work/proposals_frozen.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-opus-5"
PROMPT_FILE = ROOT / "prompts" / "extract_spans.v3.md"


# --------------------------------------------------------------------------- #
# Output contract — structured outputs requires additionalProperties: false,
# which pydantic emits from extra="forbid".
# --------------------------------------------------------------------------- #
class ProposedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraph_id: str = Field(description="ID of the paragraph the quote comes from")
    quote: str = Field(description="Exact substring of that paragraph's text")
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(description="One sentence.")


class CategoryExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_id: str
    is_impossible: bool
    impossible_reason: Literal[
        "not_impossible",
        "category_absent",
        "mentioned_without_operative_provision",
        "category_inapplicable_to_document_type",
    ]
    spans: list[ProposedSpan]
    notes: str


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_prompt(md: str) -> tuple[str, str, str]:
    """Pull the three prompt blocks out of the frozen markdown."""
    system = md.split("## SYSTEM", 1)[1].split("## USER", 1)[0]
    system = system.split("\n", 1)[1].strip().lstrip("(").strip()
    user_a = md.split("## USER  (block A", 1)[1].split("## USER  (block B", 1)[0]
    user_b = md.split("## USER  (block B", 1)[1]
    return system, user_a.split("\n", 1)[1].strip(), user_b.split("\n", 1)[1].strip()


def paragraph_block(paras: list[dict]) -> str:
    """Numbered paragraphs. Boilerplate is kept but marked, so the model can see
    document structure without being invited to quote a letterhead."""
    out = []
    for p in paras:
        flag = " [boilerplate]" if p.get("is_boilerplate") else ""
        flag += " [superseded]" if p.get("is_superseded") else ""
        out.append(f"<<{p['paragraph_id']}{flag}>>\n{p['text']}")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# The mechanical gate: quote -> offsets, or discard
# --------------------------------------------------------------------------- #
def resolve_span(span: ProposedSpan, para: dict, frozen_text: str | None) -> dict | None:
    """Return a gold-shaped answer dict, or None if the quote does not match.

    Never fuzzy-aligns. A near-miss is a discard, not a repair.
    """
    idx = para["text"].find(span.quote)
    if idx == -1:
        return None

    start = para["char_start"] + idx
    end = start + len(span.quote)

    # Independent check against the frozen corpus text — catches paragraph
    # slicing drift, not just model error.
    if frozen_text is not None and frozen_text[start:end] != span.quote:
        return None

    return {
        "text": span.quote,
        "answer_start": start,
        "answer_end": end,
        "paragraph_id": span.paragraph_id,
        "confidence": span.confidence,
        "rationale": span.rationale,
        "occurrence_ambiguous": para["text"].count(span.quote) > 1,
    }


# --------------------------------------------------------------------------- #
# One call
# --------------------------------------------------------------------------- #
def propose(
    client: anthropic.Anthropic,
    system: str,
    doc_prefix_blocks: list[dict],
    user_b: str,
    cat: dict,
    effort: str,
) -> tuple[CategoryExtraction, dict]:
    filled = user_b.format(
        category_id=cat["category_id"],
        category_name=cat["category_name"],
        description=cat["description"],
        boundary_rule=cat["boundary_rule"],
        negative_examples="; ".join(cat.get("negative_examples") or ["(none given)"]),
    )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {
                "type": "json_schema",
                "schema": CategoryExtraction.model_json_schema(),
            },
        },
        system=[
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": doc_prefix_blocks + [{"type": "text", "text": filled}]}],
    )

    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused: {resp.stop_details}")

    text = next(b.text for b in resp.content if b.type == "text")
    usage = {
        "input": resp.usage.input_tokens,
        "cache_read": resp.usage.cache_read_input_tokens,
        "cache_write": resp.usage.cache_creation_input_tokens,
        "output": resp.usage.output_tokens,
    }
    return CategoryExtraction.model_validate_json(text), usage


# --------------------------------------------------------------------------- #
def run_document(client, doc, cats, paras_by_doc, prompt_md, k, effort_ladder, out_fh):
    doc_id = doc["doc_id"]
    paras = paras_by_doc.get(doc_id, [])
    if not paras:
        print(f"  {doc_id}: no paragraphs, skipped")
        return

    by_id = {p["paragraph_id"]: p for p in paras}
    frozen_path = ROOT / "work" / "normalized" / f"{doc_id}.txt"
    frozen = frozen_path.read_text(encoding="utf-8") if frozen_path.exists() else None
    if frozen is None:
        print(f"  {doc_id}: WARNING no frozen text, offsets unverified")

    system, user_a, user_b = split_prompt(prompt_md)

    # Stable per-document prefix: PDF (optional) + numbered paragraphs.
    prefix: list[dict] = []
    pdf = ROOT / "pdfs" / f"{doc_id}.pdf"
    if pdf.exists() and pdf.stat().st_size < 30 * 1024 * 1024:
        prefix.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(pdf.read_bytes()).decode(),
            },
        })
    prefix.append({
        "type": "text",
        "text": user_a.format(paragraph_block=paragraph_block(paras)),
        "cache_control": {"type": "ephemeral"},
    })

    for cat in cats:
        for sample in range(k):
            effort = effort_ladder[sample % len(effort_ladder)]
            try:
                result, usage = propose(client, system, prefix, user_b, cat, effort)
            except Exception as exc:  # noqa: BLE001 — log and continue the sweep
                print(f"  {doc_id}/{cat['category_id']} s{sample}: FAILED {exc}")
                continue

            resolved, discarded = [], []
            for span in result.spans:
                para = by_id.get(span.paragraph_id)
                if para is None:
                    discarded.append({"reason": "unknown_paragraph_id", **span.model_dump()})
                    continue
                got = resolve_span(span, para, frozen)
                (resolved.append(got) if got else
                 discarded.append({"reason": "quote_not_verbatim", **span.model_dump()}))

            out_fh.write(json.dumps({
                "doc_id": doc_id,
                "category_id": cat["category_id"],
                "sample": sample,
                "effort": effort,
                "model": MODEL,
                "prompt_sha256": hashlib.sha256(prompt_md.encode()).hexdigest()[:16],
                "is_impossible": result.is_impossible,
                "impossible_reason": result.impossible_reason,
                "answers": resolved,
                "discarded": discarded,
                "notes": result.notes,
                "usage": usage,
            }, ensure_ascii=False) + "\n")
            out_fh.flush()

            mark = "empty" if result.is_impossible else f"{len(resolved)} span(s)"
            drop = f" [{len(discarded)} discarded]" if discarded else ""
            print(f"  {doc_id}/{cat['category_id']} s{sample} {effort}: {mark}{drop}"
                  f"  cache_read={usage['cache_read']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="single doc_id, e.g. DOC-014")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, default=1, help="samples per cell (self-consistency)")
    ap.add_argument("--effort", default="high,high,medium",
                    help="effort ladder cycled across samples")
    ap.add_argument("--out", default="work/proposals_frozen.jsonl")
    args = ap.parse_args()

    if not args.doc and not args.all:
        ap.error("pass --doc DOC-014 or --all")

    cats = load_jsonl(ROOT / "corpus" / "categories.jsonl")
    docs = load_jsonl(ROOT / "corpus" / "documents.jsonl")
    paras = load_jsonl(ROOT / "corpus" / "paragraphs.jsonl")
    prompt_md = PROMPT_FILE.read_text(encoding="utf-8")

    paras_by_doc: dict[str, list[dict]] = {}
    for p in paras:
        paras_by_doc.setdefault(p["doc_id"], []).append(p)
    for v in paras_by_doc.values():
        v.sort(key=lambda p: p["paragraph_index"])

    targets = [d for d in docs if d["doc_id"] == args.doc] if args.doc else docs
    if not targets:
        print(f"no such document: {args.doc}")
        return 1

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY, or an `ant auth login` profile
    ladder = [e.strip() for e in args.effort.split(",")]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(targets)} document(s) × {len(cats)} categories × k={args.k}"
          f" = {len(targets) * len(cats) * args.k} calls")
    with open(out_path, "a", encoding="utf-8") as fh:
        for doc in targets:
            print(f"{doc['doc_id']}:")
            run_document(client, doc, cats, paras_by_doc, prompt_md, args.k, ladder, fh)

    print(f"\nwrote {out_path}")
    print("Hash it into SNAPSHOT.txt before review begins — the audit compares "
          "human verdicts against this frozen file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
