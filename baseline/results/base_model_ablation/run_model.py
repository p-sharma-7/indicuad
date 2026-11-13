"""Score one base model on IndiCUAD across both conditions and both settings.

Usage:
    python run_model.py --model mistral --conditions full_contract,oracle_local \
        --settings direct_extraction,structured_extraction [--limit N] [--batch-size B]

Writes outputs/{model}_{condition}_{setting}.jsonl (raw per-row records) and
prints/saves a summary block to outputs/{model}_summary.json (merged across runs).
"""
import argparse
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as c

MODEL_SPECS = {
    "mistral": dict(
        hf_id="mistralai/Mistral-7B-v0.1",
        revision="27d67f1b5f57dc0953326b2601d68371d40ea8da",
        context_window=8192,
        wrap=c.wrap_mistral,
    ),
    "llama2": dict(
        hf_id="meta-llama/Llama-2-7b-hf",
        revision="01c7f73d771dfac7d292323805ebc428287df4f9",
        context_window=4096,
        wrap=c.wrap_llama2_inlegal,
    ),
}

MAX_NEW_TOKENS = 512
SAFETY_MARGIN = 32
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def build_rows_for_condition(condition, rows, by_id, by_doc_idx, doc_max_idx, dist_index, doc_text_cache):
    """Return list of dicts: {row, document_text} for the given condition."""
    out = []
    for r in rows:
        if condition == "full_contract":
            doc_id = r["doc_id"]
            if doc_id not in doc_text_cache:
                doc_text_cache[doc_id] = c.get_full_contract_text(doc_id)
            document = doc_text_cache[doc_id]
        else:
            document = c.build_oracle_document(r, by_id, by_doc_idx, doc_max_idx, dist_index)
        out.append({"row": r, "document": document})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--conditions", default="full_contract,oracle_local")
    ap.add_argument("--settings", default="direct_extraction,structured_extraction")
    ap.add_argument("--limit", type=int, default=0, help="debug: only score first N rows")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    conditions = args.conditions.split(",")
    settings = args.settings.split(",")
    spec = MODEL_SPECS[args.model]

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{args.model}] loading tokenizer/model: {spec['hf_id']}@{spec['revision']}", flush=True)
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"], token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=spec["revision"], token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()

    rows = c.load_rows()
    if args.limit:
        rows = rows[: args.limit]
    by_id, by_doc_idx, doc_max_idx = c.load_paragraphs()
    dist_index = c.load_distractor_index()
    doc_text_cache = {}

    budget = spec["context_window"] - MAX_NEW_TOKENS - SAFETY_MARGIN

    all_summaries = {}
    summary_path = os.path.join(OUT_DIR, f"{args.model}_summary.json")
    if os.path.exists(summary_path):
        all_summaries = json.load(open(summary_path))

    for condition in conditions:
        docs = build_rows_for_condition(condition, rows, by_id, by_doc_idx, doc_max_idx, dist_index, doc_text_cache)
        for setting in settings:
            key = f"{condition}/{setting}"
            out_path = os.path.join(OUT_DIR, f"{args.model}_{condition}_{setting}.jsonl")
            partial_path = out_path + ".partial"

            if os.path.exists(out_path):
                print(f"[{args.model}] === {key} === already complete, skipping", flush=True)
                continue
            print(f"[{args.model}] === {key} ===", flush=True)

            # Build prompts, tokenize, filter by budget.
            items = []  # dicts with row, input_ids length info, wrapped text
            excluded = 0
            for d in docs:
                r = d["row"]
                prompt_text = c.build_prompt(condition, setting, r["question"], d["document"])
                wrapped = spec["wrap"](prompt_text)
                n_tokens = len(tok(wrapped, add_special_tokens=True)["input_ids"])
                if n_tokens > budget:
                    excluded += 1
                    continue
                items.append({"row": r, "wrapped": wrapped, "n_tokens": n_tokens})

            # Sort by token length to minimize padding waste within batches, but keep
            # a stable record of original row order for output notes (not required).
            items.sort(key=lambda x: x["n_tokens"])

            # Resume support: skip rows already generated in a prior (crashed) attempt.
            done_qa_ids = set()
            if os.path.exists(partial_path):
                with open(partial_path, encoding="utf-8") as f:
                    for line in f:
                        done_qa_ids.add(json.loads(line)["qa_id"])
                print(f"  resuming: {len(done_qa_ids)} rows already done in {partial_path}", flush=True)
            items = [it for it in items if it["row"]["qa_id"] not in done_qa_ids]

            def generate_chunk(chunk, single_retries=5):
                """Generate for a list of items; on CUDA OOM, halve and retry (recursively).
                At chunk size 1, retry a few times with a short wait first: this GPU is shared,
                so a transient spike from another process's job is possible, not just our own
                batch being too large."""
                texts = [b["wrapped"] for b in chunk]
                try:
                    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
                    with torch.no_grad():
                        gen = model.generate(
                            **enc,
                            do_sample=False,
                            max_new_tokens=MAX_NEW_TOKENS,
                            pad_token_id=tok.pad_token_id,
                        )
                    input_len = enc["input_ids"].shape[1]
                    raw_outputs = tok.batch_decode(gen[:, input_len:], skip_special_tokens=True)
                    return list(zip(chunk, raw_outputs))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(chunk) == 1:
                        if single_retries > 0:
                            print(f"  OOM at chunk size 1, waiting 20s and retrying ({single_retries} left)", flush=True)
                            time.sleep(20)
                            return generate_chunk(chunk, single_retries=single_retries - 1)
                        raise
                    mid = len(chunk) // 2
                    print(f"  OOM at chunk size {len(chunk)}, retrying as {mid}+{len(chunk)-mid}", flush=True)
                    return generate_chunk(chunk[:mid]) + generate_chunk(chunk[mid:])

            t0 = time.time()
            bs = args.batch_size
            with open(partial_path, "a", encoding="utf-8") as fout:
                for i in range(0, len(items), bs):
                    batch = items[i : i + bs]
                    for b, raw in generate_chunk(batch):
                        r = b["row"]
                        gold_texts = [a["text"] for a in r["answers"]]
                        predicted = c.extract_answer(setting, raw)
                        scored = c.score_row(predicted, gold_texts)
                        rec = {
                            "qa_id": r["qa_id"],
                            "condition": condition,
                            "setting": setting,
                            "raw_output": raw,
                            "predicted_answer": predicted,
                            "gold_text": scored["gold_text"],
                            "precision": scored["precision"],
                            "recall": scored["recall"],
                            "f1": scored["f1"],
                            "f2": scored["f2"],
                            "jaccard": scored["jaccard"],
                            "em": scored["em"],
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                    done = min(i + bs, len(items))
                    if done % (bs * 5) == 0 or done == len(items):
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        print(f"  {done}/{len(items)} rows, {elapsed:.0f}s elapsed, {rate:.2f} rows/s", flush=True)

            # Combo fully complete: finalize by renaming partial -> final, then score from it.
            os.rename(partial_path, out_path)
            f1_sum = em_sum = 0.0
            n_scored = 0
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    f1_sum += rec["f1"]
                    em_sum += rec["em"]
                    n_scored += 1

            coverage = n_scored / len(rows)
            summary = {
                "model": args.model,
                "condition": condition,
                "setting": setting,
                "n_rows_total": len(rows),
                "n_scored": n_scored,
                "n_excluded_context": excluded,
                "coverage": coverage,
                "mean_f1": f1_sum / n_scored if n_scored else 0.0,
                "mean_em": em_sum / n_scored if n_scored else 0.0,
                "elapsed_seconds": time.time() - t0,
            }
            print(f"[{args.model}] {key} summary: {summary}", flush=True)
            all_summaries[key] = summary
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
