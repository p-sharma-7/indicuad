#!/usr/bin/env python3
"""Run exactly one (model_key, context_condition) pair.

Handles preflight -> GPU selection -> CUDA_VISIBLE_DEVICES/LD_LIBRARY_PATH
setup -> model load -> dataset loop, all for a single model. Designed to be
invoked once per model by `benchmark.cli.cmd_run` (process isolation), but
is fully usable standalone:

  python scripts/run_model.py --model qwen3_8b_fp16 --context full_contract --gpu auto
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import Config, load_config, write_json  # noqa: E402


def parse_max_memory(s: str | None) -> dict[str, str]:
    """'1:28GiB,3:28GiB' (physical GPU ids) -> caller remaps to logical
    cuda:N keys once CUDA_VISIBLE_DEVICES order is known."""
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context", choices=["full_contract", "oracle_local"], required=True)
    p.add_argument("--parallelism", choices=["single", "auto_shard", "manual_shard"], default="single")
    p.add_argument("--gpu", default="auto")
    p.add_argument("--shard-gpus", default=None)
    p.add_argument("--max-memory", default=None)
    p.add_argument("--cpu-offload", choices=["none", "auto"], default="none")
    p.add_argument("--allow-disk-offload", action="store_true")
    p.add_argument("--settings", default="direct_extraction,structured_extraction")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--row-shard-index", type=int, default=None,
                    help="0-based shard index for data-parallel row splitting across concurrent processes "
                         "(each shard writes its own file; merge with scripts/merge_shards.py before scoring)")
    p.add_argument("--row-shard-count", type=int, default=None,
                    help="total number of row shards; row i goes to shard i %% row-shard-count")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config()
    entry = cfg.model_by_key(args.model)
    if entry is None:
        print(f"unknown model_key: {args.model}", file=sys.stderr)
        return 2

    required_env = entry.get("required_conda_env")
    if required_env:
        active_env = os.environ.get("CONDA_DEFAULT_ENV")
        if active_env != required_env:
            msg = (f"{args.model} requires conda env '{required_env}' (see configs/models.yaml notes for why), "
                   f"but the active environment is '{active_env}'. Refusing to attempt a load that would likely "
                   f"fail with a confusing 'unrecognized architecture' error instead. Run:\n"
                   f"  conda activate {required_env}\n"
                   f"  python scripts/run_model.py --model {args.model} ...")
            print(msg, file=sys.stderr)
            write_json(cfg.results_dir("manifests", f"{args.model}__{args.context}__not_run.json"),
                       {"model_key": args.model, "context_condition": args.context, "status": "NOT_RUN",
                        "reason": f"wrong_conda_env: required={required_env}, active={active_env}"})
            print(f"NOT_RUN model={args.model} reason=wrong_conda_env")
            return 0

    feasibility = entry.get("v100_feasibility")
    model_type = entry.get("model_type")

    if feasibility in ("CHECKPOINT_UNRESOLVED", "GATED_CREDENTIALS_REQUIRED", "UNSUPPORTED_ON_CURRENT_HARDWARE"):
        from benchmark.models.registry import gated_credentials_present
        if feasibility == "GATED_CREDENTIALS_REQUIRED":
            ok, reason = gated_credentials_present(entry)
            if ok:
                pass  # credentials now present; fall through and attempt the run
            else:
                write_json(cfg.results_dir("manifests", f"{args.model}__{args.context}__not_run.json"),
                           {"model_key": args.model, "context_condition": args.context, "status": "NOT_RUN",
                            "reason": f"{feasibility}: {reason}"})
                print(f"NOT_RUN model={args.model} reason={feasibility}: {reason}")
                return 0
        else:
            write_json(cfg.results_dir("manifests", f"{args.model}__{args.context}__not_run.json"),
                       {"model_key": args.model, "context_condition": args.context, "status": feasibility,
                        "reason": entry.get("notes")})
            print(f"{feasibility} model={args.model}: see configs/models.yaml notes")
            return 0

    if feasibility == "FALLBACK_REQUIRED":
        print(f"REFUSING model={args.model}: v100_feasibility=FALLBACK_REQUIRED — this entry's fp16 default does not "
              f"safely fit on a V100 32GB at the common context cap. Use its named *_4bit fallback entry explicitly "
              f"instead; this harness never silently downgrades precision (see configs/models.yaml notes).")
        write_json(cfg.results_dir("manifests", f"{args.model}__{args.context}__not_run.json"),
                   {"model_key": args.model, "context_condition": args.context, "status": "FALLBACK_REQUIRED",
                    "reason": entry.get("notes")})
        return 0

    # --- GPU preflight (no torch import yet) ---
    from benchmark.gpu_preflight import run_preflight

    min_vram = entry.get("preflight_min_free_vram_gib", cfg.gpu["min_free_vram_gib"])
    shard_gpus = [int(x) for x in args.shard_gpus.split(",")] if args.shard_gpus else None
    preflight = run_preflight(cfg, mode=args.parallelism, requested_gpu=args.gpu, shard_gpus=shard_gpus,
                               min_free_vram_gib=min_vram)
    write_json(cfg.results_dir("manifests", f"{args.model}__{args.context}__preflight.json"), preflight.to_dict())

    decision = preflight.decision
    status = decision.get("status")
    if status != "READY":
        print(status)  # WAITING_FOR_GPU / WAITING_FOR_MULTIGPU_RESOURCES / REJECTED, printed verbatim per spec
        write_json(cfg.results_dir("manifests", f"{args.model}__{args.context}__not_run.json"),
                   {"model_key": args.model, "context_condition": args.context, "status": status, "decision": decision})
        return 0

    if args.parallelism == "single":
        physical_gpus = [decision["chosen_physical_gpu"]]
    else:
        physical_gpus = decision["shard_physical_gpus"]

    cuda_visible_devices = ",".join(str(g) for g in physical_gpus)

    # --- fix LD_LIBRARY_PATH + pin CUDA_VISIBLE_DEVICES, then (only past
    # this point) is it safe to import torch ---
    from benchmark.gpu_env import ensure_cuda_lib_path
    ensure_cuda_lib_path(cuda_visible_devices=cuda_visible_devices)

    device_map: dict | str
    max_memory: dict[str, str] | None = None
    if args.parallelism == "single":
        device_map = {"": 0}
    else:
        physical_to_max = parse_max_memory(args.max_memory)
        max_memory = {}
        for logical_idx, physical_id in enumerate(physical_gpus):
            gib = physical_to_max.get(str(physical_id), f"{entry.get('shard_min_free_vram_gib_per_gpu', 26)}GiB")
            max_memory[logical_idx] = gib
        if args.cpu_offload == "auto":
            max_memory["cpu"] = "96GiB"
        device_map = "auto"

    from benchmark.dataset import load_dataset
    from benchmark.run_orchestrator import run_causal_model, run_extractqa_model

    dataset = load_dataset(cfg)
    if model_type == "causal_lm":
        run_status = run_causal_model(cfg, dataset, entry, args.context, args, device_map, max_memory,
                                       physical_gpus=physical_gpus)
    elif model_type == "extractive_qa":
        run_status = run_extractqa_model(cfg, dataset, entry, args.context, args, physical_gpus=physical_gpus)
    else:
        print(f"unsupported model_type={model_type} for model_key={args.model}", file=sys.stderr)
        return 2

    if args.cpu_offload == "auto" or (max_memory and "cpu" in max_memory):
        print("NOTE: this run used CPU offload — label as CPU_OFFLOAD_DIAGNOSTIC, exclude from primary latency table.")
    if args.allow_disk_offload:
        print("NOTE: disk offload permitted for this run — label as DISK_OFFLOAD_DIAGNOSTIC, exclude from primary results.")

    print(f"STATUS={run_status}")
    return 0 if run_status in ("COMPLETE", "SKIPPED_NOT_FEASIBLE") else 1


if __name__ == "__main__":
    sys.exit(main())
