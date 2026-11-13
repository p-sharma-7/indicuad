"""Prompt construction (spec §8: P0 direct_extraction, P1 structured_extraction).

Templates live in configs/prompts/*.txt so prompt wording is versioned and
auditable outside the code. No model-specific prompt optimization happens
here — any such adaptation must live in a separately labelled best-effort
configuration, not this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from benchmark.config import Config, sha256_text

_SETTING_BLOCK_RE = re.compile(r"\[\[SETTING:(\w+)\]\]\n(.*?)\n\[\[/SETTING\]\]", re.DOTALL)


@dataclass
class PromptTemplateSet:
    path: Path
    version: str
    settings: dict[str, str]  # setting name -> template text (with {question}/{document} placeholders)


def load_template_set(path: Path) -> PromptTemplateSet:
    raw = path.read_text(encoding="utf-8")
    settings = {m.group(1): m.group(2) for m in _SETTING_BLOCK_RE.finditer(raw)}
    if not settings:
        raise ValueError(f"{path}: no [[SETTING:...]] blocks found")
    return PromptTemplateSet(path=path, version=sha256_text(raw)[:16], settings=settings)


@dataclass
class BuiltPrompt:
    setting: str
    template_path: str
    template_version: str
    prompt_text: str
    prompt_template_sha256: str  # hash of the template text itself (wording only)
    model_input_sha256: str      # hash of the fully assembled prompt actually sent


def build_prompt(template_set: PromptTemplateSet, setting: str, question: str, document_text: str) -> BuiltPrompt:
    if setting not in template_set.settings:
        raise KeyError(f"setting={setting!r} not found in {template_set.path} (have {list(template_set.settings)})")
    template = template_set.settings[setting]
    prompt_text = template.format(question=question, document=document_text)
    return BuiltPrompt(
        setting=setting,
        template_path=str(template_set.path),
        template_version=template_set.version,
        prompt_text=prompt_text,
        prompt_template_sha256=sha256_text(template),
        model_input_sha256=sha256_text(prompt_text),
    )


def load_full_contract_templates(cfg: Config) -> PromptTemplateSet:
    from benchmark.config import REPO_ROOT
    return load_template_set(REPO_ROOT / cfg.prompts["direct_extraction_template"])


def load_oracle_local_templates(cfg: Config) -> PromptTemplateSet:
    from benchmark.config import REPO_ROOT
    return load_template_set(REPO_ROOT / cfg.prompts["oracle_local_template"])
