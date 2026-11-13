"""Configuration loading and path resolution for the IndiCUAD harness.

`REPO_ROOT` is the `baseline/` directory (the parent of this `benchmark/`
package). The dataset root comes from `dataset.root` in
configs/indicuad_v2.yaml (the repository root, `..`) and is only ever read.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_ROOT = REPO_ROOT / "results" / "indicuad_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class DatasetPaths:
    root: Path
    qa_release: Path
    qa_unanswerable: Path
    distractors: Path
    documents_jsonl: Path
    paragraphs_jsonl: Path
    categories_jsonl: Path
    normalized_dir: Path


@dataclass
class Config:
    raw: dict[str, Any]
    models_raw: dict[str, Any]
    category_partitions_raw: dict[str, Any]
    dataset: DatasetPaths

    @property
    def expected(self) -> dict[str, Any]:
        return self.raw["dataset"]["expected"]

    @property
    def context(self) -> dict[str, Any]:
        return self.raw["context"]

    @property
    def prompts(self) -> dict[str, Any]:
        return self.raw["prompts"]

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw["scoring"]

    @property
    def bootstrap(self) -> dict[str, Any]:
        return self.raw["bootstrap"]

    @property
    def gpu(self) -> dict[str, Any]:
        return self.raw["gpu"]

    @property
    def models(self) -> list[dict[str, Any]]:
        return self.models_raw["models"]

    def model_by_key(self, model_key: str) -> dict[str, Any] | None:
        for m in self.models:
            if m["model_key"] == model_key:
                return m
        return None

    def results_dir(self, *parts: str) -> Path:
        path = REPO_ROOT / self.raw["paths"]["results_dir"]
        for p in parts:
            path = path / p
        return path


def load_config(
    config_path: Path | None = None,
    models_path: Path | None = None,
    partitions_path: Path | None = None,
) -> Config:
    config_path = config_path or (CONFIGS_DIR / "indicuad_v2.yaml")
    models_path = models_path or (CONFIGS_DIR / "models.yaml")
    partitions_path = partitions_path or (CONFIGS_DIR / "category_partitions.yaml")

    raw = read_yaml(config_path)
    models_raw = read_yaml(models_path)
    partitions_raw = read_yaml(partitions_path)

    ds = raw["dataset"]
    dataset_root = (REPO_ROOT / ds["root"]).resolve()
    dataset = DatasetPaths(
        root=dataset_root,
        qa_release=dataset_root / ds["qa_release"],
        qa_unanswerable=dataset_root / ds["qa_unanswerable"],
        distractors=dataset_root / ds["distractors"],
        documents_jsonl=dataset_root / ds["documents_jsonl"],
        paragraphs_jsonl=dataset_root / ds["paragraphs_jsonl"],
        categories_jsonl=dataset_root / ds["categories_jsonl"],
        normalized_dir=dataset_root / ds["normalized_dir"],
    )

    return Config(
        raw=raw,
        models_raw=models_raw,
        category_partitions_raw=partitions_raw,
        dataset=dataset,
    )


# --------------------------------------------------------------------------
# Environment / provenance capture (§15, §7 manifest requirements)
# --------------------------------------------------------------------------


def git_commit() -> str:
    """Best-effort git commit for REPO_ROOT. REPO_ROOT is not a git
    repository at implementation time, so this reports that explicitly
    rather than raising."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
        return "not_a_git_repository"
    except Exception as exc:  # noqa: BLE001 - provenance capture must never crash a run
        return f"git_unavailable:{exc}"


def hashed_hostname() -> str:
    name = platform.node() or "unknown-host"
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in ("torch", "transformers", "accelerate", "safetensors", "tokenizers",
                "sentencepiece", "bitsandbytes", "numpy", "pandas", "scipy",
                "sklearn", "yaml", "gpustat"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[pkg] = None
    return versions


def cuda_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"torch_cuda_available": None, "cuda_version_torch": None}
    try:
        import torch  # noqa: PLC0415
        summary["torch_cuda_available"] = torch.cuda.is_available()
        summary["cuda_version_torch"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        summary["error"] = str(exc)
    return summary


def base_run_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fields required on every run manifest per spec §15/§7: git commit,
    environment, GPU, dtype/params (caller adds those), prompt hash (caller
    adds), timestamps."""
    manifest = {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "hostname_hashed": hashed_hostname(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "cuda": cuda_summary(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
