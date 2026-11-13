"""Shared data loading, oracle_local construction, and scoring for the IndiCUAD base-model eval.

Dataset root is read-only; never write inside it.
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # baseline/, for the shared JSON parser
from benchmark.normalize import first_balanced_object  # noqa: E402

# Dataset = repository root (this file is baseline/results/base_model_ablation/common.py).
DATASET = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows():
    rows = json.load(open(f"{DATASET}/qa/qa.json", encoding="utf-8"))
    filt = [r for r in rows if r.get("is_impossible") is False and r.get("split") != "negatives"]
    assert len(filt) == 1521, f"expected 1521 filtered rows, got {len(filt)}"
    return filt


def load_paragraphs():
    paras = [json.loads(l) for l in open(f"{DATASET}/corpus/paragraphs.jsonl", encoding="utf-8")]
    by_id = {p["paragraph_id"]: p for p in paras}
    by_doc_idx = {}
    doc_max_idx = defaultdict(int)
    for p in paras:
        by_doc_idx[(p["doc_id"], p["paragraph_index"])] = p
        doc_max_idx[p["doc_id"]] = max(doc_max_idx[p["doc_id"]], p["paragraph_index"])
    return by_id, by_doc_idx, doc_max_idx


def load_distractor_index():
    dist = json.load(open(f"{DATASET}/qa/distractors.json", encoding="utf-8"))
    index = {}
    for rec in dist:
        for qid in rec["qa_ids"]:
            index[(qid, rec["answer_index"])] = rec
    return index


def get_full_contract_text(doc_id):
    with open(f"{DATASET}/work/normalized/{doc_id}.txt", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# oracle_local excerpt construction
# ---------------------------------------------------------------------------

def _find_neighbor(by_doc_idx, doc_id, start_idx, direction, max_idx):
    i = start_idx
    while 0 <= i <= max_idx:
        p = by_doc_idx.get((doc_id, i))
        if p is not None and not p["is_boilerplate"]:
            return p
        i += direction
    return None


def build_context_paragraphs(row, by_id, by_doc_idx, doc_max_idx):
    doc_id = row["doc_id"]
    max_idx = doc_max_idx[doc_id]
    collected = {}
    for pid in row["gold_paragraph_ids"]:
        gold = by_id[pid]
        collected[gold["paragraph_id"]] = gold
        back = _find_neighbor(by_doc_idx, doc_id, gold["paragraph_index"] - 1, -1, max_idx)
        if back is not None:
            collected[back["paragraph_id"]] = back
        fwd = _find_neighbor(by_doc_idx, doc_id, gold["paragraph_index"] + 1, +1, max_idx)
        if fwd is not None:
            collected[fwd["paragraph_id"]] = fwd
    return sorted(collected.values(), key=lambda p: p["paragraph_index"])


def build_distractor_segments(row, dist_index):
    segments = []
    n = len(row["answers"])
    for i in range(n):
        rec = dist_index[(row["qa_id"], i)]
        for d in rec["distractors"]:
            if d["source"] == "mined":
                header = f'[DISTRACTOR-SOURCE: {d["from_doc_id"]} / {d["from_paragraph_id"]}]'
            else:
                header = "[SYNTHETIC-DISTRACTOR]"
            segments.append((header, d["text"]))
    return segments


def build_oracle_document(row, by_id, by_doc_idx, doc_max_idx, dist_index):
    parts = []
    for p in build_context_paragraphs(row, by_id, by_doc_idx, doc_max_idx):
        parts.append(f'[GOLD-ADJACENT: {p["doc_id"]} / {p["paragraph_id"]}]\n{p["text"]}')
    for header, text in build_distractor_segments(row, dist_index):
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt templates (verbatim)
# ---------------------------------------------------------------------------

FULL_CONTRACT_DIRECT = """You are a contract clause extraction system. Given the question and the contract, return only the minimal verbatim text from the contract that directly answers the question. Copy the text exactly from the contract. Do not paraphrase. Do not explain. Do not return headings unless they are necessary to answer the question. The answer exists in the contract.

Question: {question}

Contract:
{document}

Answer:"""

FULL_CONTRACT_STRUCTURED = """You are a contract clause extraction system. Given the question and the contract, return only the minimal verbatim text from the contract that directly answers the question. Copy the text exactly from the contract. Do not paraphrase. Do not explain. Do not return headings unless they are necessary to answer the question. The answer exists in the contract.

Respond with strict JSON only, matching exactly this schema and nothing else:
{{"answer": "<minimal verbatim span copied from contract>"}}

Question: {question}

Contract:
{document}

JSON:"""

ORACLE_LOCAL_DIRECT = """You are a contract clause extraction system. Given the question and the contract excerpt below, return only the minimal verbatim text from the excerpt that directly answers the question. Copy the text exactly from the excerpt. Do not paraphrase. Do not explain. Do not return headings unless they are necessary to answer the question. The answer exists in the excerpt.

Question: {question}

Contract excerpt:
{document}

Answer:"""

ORACLE_LOCAL_STRUCTURED = """You are a contract clause extraction system. Given the question and the contract excerpt below, return only the minimal verbatim text from the excerpt that directly answers the question. Copy the text exactly from the excerpt. Do not paraphrase. Do not explain. Do not return headings unless they are necessary to answer the question. The answer exists in the excerpt.

Respond with strict JSON only, matching exactly this schema and nothing else:
{{"answer": "<minimal verbatim span copied from excerpt>"}}

Question: {question}

Contract excerpt:
{document}

JSON:"""

TEMPLATES = {
    ("full_contract", "direct_extraction"): FULL_CONTRACT_DIRECT,
    ("full_contract", "structured_extraction"): FULL_CONTRACT_STRUCTURED,
    ("oracle_local", "direct_extraction"): ORACLE_LOCAL_DIRECT,
    ("oracle_local", "structured_extraction"): ORACLE_LOCAL_STRUCTURED,
}


def build_prompt(condition, setting, question, document):
    return TEMPLATES[(condition, setting)].format(question=question, document=document)


# ---------------------------------------------------------------------------
# Chat wrappers (base-model tokenizers, no chat_template of their own)
# ---------------------------------------------------------------------------

def wrap_mistral(prompt_text):
    # Aalap's own tokenizer ships no chat_template either; this is its fallback.
    return f"[INST] {prompt_text} [/INST]"


def wrap_llama2_inlegal(prompt_text):
    # Verified real chat_template on sudipto-ducs/InLegalLLaMA's tokenizer_config.json:
    # single user turn -> "Human: {content}\nAssistant: " (no system message used here).
    return f"Human: {prompt_text}\nAssistant: "


# ---------------------------------------------------------------------------
# Answer post-processing
# ---------------------------------------------------------------------------

_STOP_MARKERS = [
    "\nQuestion:", "\nContract:", "\nContract excerpt:", "\nJSON:",
    "\n[INST]", "[INST]", "\nHuman:", "\nAssistant:", "<s>",
]


def extract_direct_answer(raw_output):
    text = raw_output
    cut = len(text)
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def extract_structured_answer(raw_output):
    blob = first_balanced_object(raw_output)
    if blob is None:
        return ""
    try:
        obj = json.loads(blob)
    except Exception:
        return ""
    if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
        return obj["answer"]
    return ""


def extract_answer(setting, raw_output):
    if setting == "direct_extraction":
        return extract_direct_answer(raw_output)
    elif setting == "structured_extraction":
        return extract_structured_answer(raw_output)
    raise ValueError(setting)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return TOKEN_RE.findall(text)


def normalize_string(text):
    text = unicodedata.normalize("NFKC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def _prf1(pred_tokens, gold_tokens):
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    pc, gc = Counter(pred_tokens), Counter(gold_tokens)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    p = overlap / len(pred_tokens)
    r = overlap / len(gold_tokens)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


def _fbeta(p, r, beta):
    b2 = beta * beta
    denom = b2 * p + r
    if denom == 0:
        return 0.0
    return (1 + b2) * p * r / denom


def _jaccard(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_row(pred_text, gold_texts):
    """Score pred_text against every gold candidate; keep whichever gives highest F1."""
    pred_tokens = normalize_tokenize(pred_text)
    best = None
    for g in gold_texts:
        g_tokens = normalize_tokenize(g)
        p, r, f1 = _prf1(pred_tokens, g_tokens)
        if best is None or f1 > best[0]:
            best = (f1, p, r, g_tokens, g)
    f1, p, r, g_tokens, g_text = best
    f2 = _fbeta(p, r, beta=2)
    jac = _jaccard(pred_tokens, g_tokens)
    em = 1.0 if normalize_string(pred_text) == normalize_string(g_text) else 0.0
    return {
        "precision": p, "recall": r, "f1": f1, "f2": f2,
        "jaccard": jac, "em": em, "gold_text": g_text,
    }
