"""GPU preflight — live eligibility checks before a model is loaded.

Every decision is based on a live `nvidia-smi` query at call time. Which
physical GPUs may be used comes from `gpu.allowed_physical_gpus` /
`gpu.excluded_physical_gpus` in configs/indicuad_v2.yaml (empty allowed list =
every GPU nvidia-smi reports); a GPU is then eligible only if it clears the
configured free-VRAM and utilization thresholds.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any

from benchmark.config import Config, utc_now


def gpu_policy(cfg: Config, visible: list[int]) -> tuple[frozenset[int], frozenset[int]]:
    """(allowed, excluded) physical GPU indices for this run."""
    excluded = frozenset(cfg.gpu.get("excluded_physical_gpus") or [])
    allowed = frozenset(cfg.gpu.get("allowed_physical_gpus") or visible) - excluded
    return allowed, excluded


@dataclass
class GpuState:
    index: int
    name: str
    memory_total_mib: float
    memory_free_mib: float
    memory_used_mib: float
    utilization_pct: float
    processes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def free_gib(self) -> float:
        return self.memory_free_mib / 1024.0


@dataclass
class GpuEligibility:
    index: int
    eligible: bool
    reasons: list[str]
    state: GpuState | None


def query_nvidia_smi() -> dict[int, GpuState]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed (rc={out.returncode}): {out.stderr}")

    states: dict[int, GpuState] = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        idx, name, total, free, used, util = parts
        states[int(idx)] = GpuState(
            index=int(idx), name=name,
            memory_total_mib=float(total), memory_free_mib=float(free),
            memory_used_mib=float(used), utilization_pct=float(util),
        )

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        uuid_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        uuid_to_index = {}
        for line in uuid_out.stdout.strip().splitlines():
            idx_s, uuid_s = [p.strip() for p in line.split(",")]
            uuid_to_index[uuid_s] = int(idx_s)
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines():
                if not line.strip():
                    continue
                gpu_uuid, pid, used_mem, pname = [p.strip() for p in line.split(",", 3)]
                idx = uuid_to_index.get(gpu_uuid)
                if idx in states:
                    states[idx].processes.append({"pid": pid, "used_memory_mib": used_mem, "process_name": pname})
    except Exception:  # noqa: BLE001 - process listing is best-effort context, never fatal
        pass

    return states


def evaluate_gpu(
    index: int,
    states: dict[int, GpuState],
    min_free_vram_gib: float,
    max_utilization_pct: float,
    allowed: frozenset[int],
    excluded: frozenset[int] = frozenset(),
    own_pids: set[str] | None = None,
) -> GpuEligibility:
    reasons: list[str] = []
    if index in excluded:
        return GpuEligibility(index=index, eligible=False,
                               reasons=[f"excluded by gpu.excluded_physical_gpus {sorted(excluded)}"],
                               state=states.get(index))
    if index not in allowed:
        return GpuEligibility(index=index, eligible=False,
                               reasons=[f"index {index} not in allowed set {sorted(allowed)}"],
                               state=states.get(index))

    state = states.get(index)
    if state is None:
        return GpuEligibility(index=index, eligible=False, reasons=["not reported by nvidia-smi"], state=None)

    own_pids = own_pids or set()
    other_processes = [p for p in state.processes if p["pid"] not in own_pids]
    if other_processes:
        reasons.append(f"{len(other_processes)} process(es) from other jobs present (logged, not killed): "
                        + "; ".join(f"pid={p['pid']} mem={p['used_memory_mib']}MiB name={p['process_name']}" for p in other_processes))

    ok = True
    if state.free_gib < min_free_vram_gib:
        ok = False
        reasons.append(f"free VRAM {state.free_gib:.1f} GiB < required {min_free_vram_gib:.1f} GiB")
    if state.utilization_pct > max_utilization_pct:
        ok = False
        reasons.append(f"utilization {state.utilization_pct:.0f}% > threshold {max_utilization_pct:.0f}%")

    if ok and not reasons:
        reasons.append(f"free={state.free_gib:.1f} GiB, util={state.utilization_pct:.0f}% — clears thresholds, no other processes")
    elif ok:
        reasons.append(f"free={state.free_gib:.1f} GiB, util={state.utilization_pct:.0f}% — clears thresholds "
                        f"despite other processes present (small footprint tolerated; never interfered with)")

    return GpuEligibility(index=index, eligible=ok, reasons=reasons, state=state)


@dataclass
class PreflightReport:
    timestamp: str
    requested_mode: str  # single | auto_shard | manual_shard
    min_free_vram_gib: float
    max_utilization_pct: float
    all_states: dict[int, GpuState]
    eligibility: dict[int, GpuEligibility]
    decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "requested_mode": self.requested_mode,
            "min_free_vram_gib": self.min_free_vram_gib,
            "max_utilization_pct": self.max_utilization_pct,
            "gpu_states": {
                i: {"name": s.name, "memory_total_mib": s.memory_total_mib, "memory_free_mib": s.memory_free_mib,
                    "memory_used_mib": s.memory_used_mib, "utilization_pct": s.utilization_pct,
                    "processes": s.processes}
                for i, s in self.all_states.items()
            },
            "eligibility": {i: {"eligible": e.eligible, "reasons": e.reasons} for i, e in self.eligibility.items()},
            "decision": self.decision,
        }


def run_preflight(
    cfg: Config,
    mode: str = "single",
    requested_gpu: str = "auto",
    shard_gpus: list[int] | None = None,
    min_free_vram_gib: float | None = None,
    max_utilization_pct: float | None = None,
) -> PreflightReport:
    gpu_cfg = cfg.gpu
    min_free_vram_gib = min_free_vram_gib if min_free_vram_gib is not None else gpu_cfg["min_free_vram_gib"]
    max_utilization_pct = max_utilization_pct if max_utilization_pct is not None else gpu_cfg["max_utilization_pct"]

    states = query_nvidia_smi()
    allowed, excluded = gpu_policy(cfg, sorted(states.keys()))
    eligibility = {
        i: evaluate_gpu(i, states, min_free_vram_gib, max_utilization_pct, allowed, excluded)
        for i in sorted(states.keys())
    }

    decision: dict[str, Any] = {"mode": mode}

    if mode == "single":
        # "auto" tries every allowed GPU in ascending physical index order.
        candidates = sorted(allowed) if requested_gpu == "auto" else [int(requested_gpu)]
        chosen = next((c for c in candidates if eligibility.get(c) and eligibility[c].eligible), None)
        if chosen is not None:
            decision.update({"status": "READY", "chosen_physical_gpu": chosen})
        else:
            decision.update({"status": "WAITING_FOR_GPU", "chosen_physical_gpu": None,
                              "candidates_checked": candidates})

    elif mode in ("auto_shard", "manual_shard"):
        candidates = shard_gpus or sorted(allowed)[:2]
        bad = [g for g in candidates if g not in allowed]
        if bad:
            decision.update({"status": "REJECTED", "reason": f"requested shard GPUs {bad} are not in the allowed set {sorted(allowed)}"})
        else:
            sharded_min = gpu_cfg.get("sharded_min_free_vram_gib_per_gpu", min_free_vram_gib)
            per_gpu_eligible = {
                g: evaluate_gpu(g, states, sharded_min, max_utilization_pct, allowed, excluded) for g in candidates
            }
            all_ok = all(e.eligible for e in per_gpu_eligible.values())
            if all_ok:
                decision.update({"status": "READY", "shard_physical_gpus": candidates,
                                  "per_gpu": {g: per_gpu_eligible[g].reasons for g in candidates}})
            else:
                decision.update({"status": "WAITING_FOR_MULTIGPU_RESOURCES",
                                  "shard_physical_gpus_requested": candidates,
                                  "per_gpu": {g: per_gpu_eligible[g].reasons for g in candidates}})
    else:
        decision.update({"status": "REJECTED", "reason": f"unknown parallelism mode {mode!r}"})

    return PreflightReport(
        timestamp=utc_now(), requested_mode=mode,
        min_free_vram_gib=min_free_vram_gib, max_utilization_pct=max_utilization_pct,
        all_states=states, eligibility=eligibility, decision=decision,
    )
