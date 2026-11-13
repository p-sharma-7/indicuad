"""Zero-shot embedding span-ranking adapter (spec §9 explicitly names this
as an alternative to a trained extractive-QA head: "extractive QA/span-
ranking adapter").

law-ai/InLegalBERT ships as a fill-mask MLM checkpoint with no pretrained
extractive-QA head. Attaching AutoModelForQuestionAnswering would bolt on a
*randomly initialized* head whose predictions carry no information from
the checkpoint's actual training — scoring that and calling it "InLegalBERT
performance" would misrepresent the model, and this harness does not
fine-tune models (no training budget in scope).

Instead: encode the question and every candidate paragraph of the document
with InLegalBERT's own pretrained encoder (mean-pooled last-hidden-state —
a standard, untrained similarity representation), rank paragraphs by cosine
similarity, and return the top-ranked paragraph's full text as the
predicted span. This is genuine zero-shot use of the checkpoint's learned
representations, with no fabricated training.

Architecturally distinct from a real extractive-QA model
(cuad_legalbert_or_roberta) in two ways the report must never blur:
  1. it returns whole-paragraph spans, not token-level minimal spans;
  2. it has no supervised signal for "is this the answer" at all, only
     unsupervised similarity.
Every output record is tagged `adapter_type: "embedding_span_ranking"` so
scoring/reporting can label it accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from benchmark.models.base import BaseRunner
from benchmark.models.hf_causal import ModelLoadError, _is_oom


@dataclass
class EmbeddingSpanRanker(BaseRunner):
    model_entry: dict[str, Any]

    def __post_init__(self):
        self.model = None
        self.tokenizer = None
        self.load_time_seconds: float | None = None

    def load(self) -> None:
        entry = self.model_entry
        hf_id = entry.get("hf_id")
        revision = entry.get("revision")
        if not hf_id:
            raise ModelLoadError("checkpoint_unresolved", f"{entry['model_key']}: no hf_id registered")
        t0 = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
            self.model = AutoModel.from_pretrained(hf_id, revision=revision)
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise ModelLoadError("cuda_oom", f"{hf_id}: OOM during load: {exc}") from exc
            raise ModelLoadError("model_load_error", f"{hf_id}@{revision}: {exc}") from exc
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.load_time_seconds = time.time() - t0

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embed(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # mean-pooled, [n_texts, hidden]

    def predict_span_over_candidates(self, question: str, candidates_in: list[dict[str, Any]]) -> dict[str, Any]:
        """candidates_in: [{paragraph_id, char_start, char_end, text}, ...]
        — the caller resolves these (full document's real paragraphs for
        full_contract; the oracle_local SerializedDocument's own segments
        for that condition) so this class stays agnostic to context
        condition."""
        entry = self.model_entry
        max_length = entry.get("max_seq_length", 256)
        paragraphs = candidates_in
        if not paragraphs:
            return {"raw_output": "", "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0,
                    "peak_gpu_memory_mib": None, "best_span": None, "n_candidates": 0, "candidates": []}

        t0 = time.time()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            q_emb = self._embed([question], max_length=max_length)  # [1, hidden]
            texts = [p["text"] for p in paragraphs]
            batch_size = 32
            all_embs = []
            for i in range(0, len(texts), batch_size):
                all_embs.append(self._embed(texts[i:i + batch_size], max_length=max_length))
            p_embs = torch.cat(all_embs, dim=0)  # [n_paragraphs, hidden]

            q_norm = torch.nn.functional.normalize(q_emb, dim=-1)
            p_norm = torch.nn.functional.normalize(p_embs, dim=-1)
            sims = (p_norm @ q_norm.T).squeeze(-1)  # [n_paragraphs]
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ModelLoadError("cuda_oom", f"OOM during embedding-ranking inference: {exc}") from exc
            raise ModelLoadError("generation_error", str(exc)) from exc

        latency = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        order = torch.argsort(sims, descending=True).tolist()
        candidates = [{
            "paragraph_id": paragraphs[i]["paragraph_id"], "char_start": paragraphs[i]["char_start"],
            "char_end": paragraphs[i]["char_end"], "text": paragraphs[i]["text"], "score": float(sims[i]),
        } for i in order[: min(20, len(order))]]
        best = candidates[0]

        return {
            "raw_output": best["text"],
            "input_tokens": None, "output_tokens": None, "latency_seconds": latency,
            "peak_gpu_memory_mib": (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None,
            "best_span": {"char_start": best["char_start"], "char_end": best["char_end"], "text": best["text"], "score": best["score"]},
            "n_candidates": len(paragraphs),
            "candidates": candidates,
        }

    def predict_span(self, question: str, dataset, doc_id: str) -> dict[str, Any]:
        """Full-contract convenience: rank over the document's own
        non-boilerplate paragraphs (falling back to all paragraphs if the
        document is entirely boilerplate-flagged, which never happens in
        the core corpus but is handled rather than assumed)."""
        paragraphs = [p for p in dataset.paragraphs_by_doc.get(doc_id, []) if not p.get("is_boilerplate")]
        if not paragraphs:
            paragraphs = dataset.paragraphs_by_doc.get(doc_id, [])
        return self.predict_span_over_candidates(question, paragraphs)
