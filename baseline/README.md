# IndiCUAD baselines

Evaluation harness, raw model outputs, per-row scores and analysis for the twelve
systems in the paper. The dataset is the repository root; the harness only reads it.

```
benchmark/          harness: prompting, serialization, runners, scoring, bootstrap, analysis
configs/            indicuad_v2.yaml, models.yaml (12 systems, pinned revisions), prompts/
scripts/            run_model.py, paper_numbers.py, document_bootstrap.py, CLI wrappers
tests/              unit tests (pytest)
results/indicuad_v2/
  raw/<model>/<condition>.jsonl      every generation
  scores/<model>/<condition>.jsonl   per-row metrics (input to analysis)
  analysis/                          leaderboards, pairwise tests, breakdowns
  manifests/                         per-run provenance (revision, package versions)
results/base_model_ablation/         Mistral-7B-v0.1 / Llama-2-7B base runs (own script)
```

## Setup

Python 3.10, then `pip install -r requirements.txt` (choose the torch wheel for
your CUDA driver). Gated checkpoints (Aalap, Llama-3.1, Llama-2) need `HF_TOKEN`.
Run everything from this directory.

## Reproduce

CPU only, from the shipped scores:

```bash
python -m benchmark.cli analyze     # results/indicuad_v2/analysis/
python scripts/paper_numbers.py     # §5.1–5.3 numbers not written by analyze
pytest tests -q
```

| Paper | Source |
|---|---|
| Table 1 | `python -m benchmark.cli validate` (checks every count; runs `../scripts/reproduce.py --verify`) |
| Table 2 | `analysis/leaderboard_{full_contract,oracle_local}.csv`, `direct_extraction` rows |
| Table 3 | `analysis/pairwise_comparisons.json` (`metric: f1`) |
| Table 4 | `analysis/leaderboard_{full_contract,oracle_local}.csv` (DSR, DCR, margin) |
| §5.1 base models | `results/base_model_ablation/outputs/results_table.json`, `paper_numbers.py` |
| §5.2 JSON audit | `paper_numbers.py` |
| §5.3 | `analysis/doc_length_bins.json`, `paper_numbers.py` |
| §5.4 mined vs perturbed | `analysis/distractor_type_breakdown.json` |
| §5.5 | `analysis/register_robustness.json`, `analysis/per_category.json` |
| Appendix C | `python scripts/document_bootstrap.py` → `analysis/document_bootstrap.json` |

Re-score from raw outputs: `python -m benchmark.cli score` (skips rows already in
`scores/`, which holds the published scores).

Re-run a model (GPU):

```bash
python scripts/run_model.py --model qwen3_8b_fp16 --context full_contract
python scripts/run_model.py --model qwen3_8b_fp16 --context oracle_local
```

Base-model ablation: `cd results/base_model_ablation && python run_model.py --model mistral`
(or `llama2`), then `python aggregate.py`.
