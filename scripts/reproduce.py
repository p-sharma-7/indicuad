"""IndiCUAD reproduction harness — verify, rebuild, or re-annotate.

Three tiers, because the pipeline is not uniformly reproducible and pretending
otherwise is the failure this repository exists to avoid:

    python scripts/reproduce.py --verify    (default) read-only, offline, ~10s
    python scripts/reproduce.py --rebuild   re-derives files from the freeze
    python scripts/reproduce.py --annotate  re-runs the prompted stages (costs money)

VERIFY re-derives every published figure and invariant from the frozen artefacts
and compares against what the release claims. It writes nothing, needs no
network, no API key and no third-party package. This is the tier an artifact
evaluator runs, and the only one that can fail in a way that means the dataset
is wrong.

REBUILD re-runs the deterministic derivation stages — grid, scoping, row build,
context windows, drops, distractor mining. Given the same corpus and the same
annotations these produce byte-identical output. It rewrites files in place,
leaving .bak copies, so run it on a clean tree.

ANNOTATE re-runs the model-prompted stages. It is NOT reproducible in the strict
sense and never will be: no temperature or seed is available on the model, the
effort ladder varies deliberately. Re-running produces a *re-annotation* to compare against, not a
reproduction. Requires ANTHROPIC_API_KEY, `anthropic`, and `pydantic`.

Exit code is 0 only if every check in the selected tier passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
QA = ROOT / "qa"
FROZEN = ROOT / "work" / "normalized"
BLIND = ROOT / "annotations" / "blind"

# Windows consoles default to cp1252 and this report prints kappa and section
# signs. Fail soft rather than dying halfway through a check run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 — older streams, redirected output
    pass


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
class Report:
    """Collects check results. A check is PASS, FAIL, or INFO (no assertion)."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.failed = 0

    def check(self, group: str, name: str, expect, actual) -> bool:
        ok = expect == actual
        self.rows.append((group, name, "PASS" if ok else "FAIL",
                          f"{actual}" if ok else f"expected {expect}, got {actual}"))
        if not ok:
            self.failed += 1
        return ok

    def info(self, group: str, name: str, value) -> None:
        self.rows.append((group, name, "INFO", f"{value}"))

    def note(self, group: str, name: str, text: str) -> None:
        self.rows.append((group, name, "NOTE", text))

    def emit(self) -> int:
        width = max(len(n) for _, n, _, _ in self.rows) + 2
        current = None
        for group, name, status, detail in self.rows:
            if group != current:
                print(f"\n  {group}")
                print(f"  {'-' * (width + 46)}")
                current = group
            mark = {"PASS": "ok  ", "FAIL": "FAIL", "INFO": "    ", "NOTE": " !  "}[status]
            print(f"  {mark} {name:<{width}} {detail}")
        checks = sum(1 for _, _, s, _ in self.rows if s in ("PASS", "FAIL"))
        print()
        print("  " + "=" * (width + 46))
        if self.failed:
            print(f"  {self.failed} of {checks} checks FAILED")
        else:
            print(f"  all {checks} checks passed")
        return 1 if self.failed else 0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- #
# VERIFY
# --------------------------------------------------------------------------- #
def verify() -> int:
    r = Report()

    missing = [p for p in (CORPUS / "documents.jsonl", CORPUS / "paragraphs.jsonl",
                           CORPUS / "categories.jsonl", QA / "qa.json") if not p.is_file()]
    if missing:
        for p in missing:
            print(f"  FATAL: {p.relative_to(ROOT)} not found")
        return 1

    docs = jsonl(CORPUS / "documents.jsonl")
    paras = jsonl(CORPUS / "paragraphs.jsonl")
    cats = jsonl(CORPUS / "categories.jsonl")
    qa = load_json(QA / "qa.json")

    core_ids = {d["doc_id"] for d in docs if d["split"] == "core"}
    frozen: dict[str, str] = {}
    for doc_id in {d["doc_id"] for d in docs}:
        path = FROZEN / f"{doc_id}.txt"
        if path.is_file():
            frozen[doc_id] = path.read_text(encoding="utf-8")

    # ---- corpus -----------------------------------------------------------
    g = "corpus"
    r.check(g, "core documents", 43, len(core_ids))
    r.check(g, "frozen text present for every core doc", 0,
            len(core_ids - set(frozen)))
    r.check(g, "core corpus characters", 4_432_753,
            sum(len(frozen[d]) for d in sorted(core_ids) if d in frozen))

    core_paras = [p for p in paras if p["doc_id"] in core_ids]
    r.check(g, "core paragraphs", 9_174, len(core_paras))
    r.check(g, "core paragraphs flagged furniture", 1_719,
            sum(1 for p in core_paras if p.get("is_boilerplate")))
    r.check(g, "core paragraphs indexed", 7_455,
            sum(1 for p in core_paras
                if not p.get("is_boilerplate") and not p.get("is_superseded")))

    # The invariant the whole offset scheme rests on.
    drift = [p["paragraph_id"] for p in core_paras
             if frozen.get(p["doc_id"], "")[p["char_start"]:p["char_end"]] != p["text"]]
    r.check(g, "every paragraph slice-exact vs frozen text", 0, len(drift))
    if drift:
        r.note(g, "first drifted paragraphs", ", ".join(drift[:5]))

    # ---- schema -----------------------------------------------------------
    g = "schema"
    r.check(g, "categories", 26, len(cats))
    tiers = Counter(c["attention_tier"] for c in cats)
    r.check(g, "low_friction / moderate / high_contest", (7, 11, 8),
            (tiers["low_friction"], tiers["moderate"], tiers["high_contest"]))
    r.check(g, "every category carries three question registers", 26,
            sum(1 for c in cats if c.get("canonical_question")
                and c.get("semi_formal_question") and c.get("naive_layperson_question")))
    r.check(g, "every category carries a boundary rule", 26,
            sum(1 for c in cats if (c.get("boundary_rule") or "").strip()))
    statuses = Counter(c.get("review_status") for c in cats)
    r.info(g, "recorded legal-review status", dict(statuses))
    if statuses.get("WAIVED_NOT_REVIEWED"):
        r.note(g, "protocol §7 legal review",
               "recorded as WAIVED_NOT_REVIEWED — DISPUTED, see REPRODUCE.md §Open")

    # ---- grid arithmetic --------------------------------------------------
    g = "grid"
    r.check(g, "core grid cells (43 × 26)", 1_118, len(core_ids) * len(cats))
    scope = ROOT / "validation" / "category_scope_report.json"
    if scope.is_file():
        s = load_json(scope)
        r.check(g, "core cells scoped out", 173, s["core_cells_scoped_out"])
        r.check(g, "core cells askable", 945, s["core_cells_applicable"])
        r.check(g, "grid closes: askable + scoped = 43 × 26", 1_118,
                s["core_cells_applicable"] + s["core_cells_scoped_out"])

    # ---- release ----------------------------------------------------------
    g = "release"
    cells = {(row["doc_id"], row["category_id"]) for row in qa}
    groups = defaultdict(list)
    for row in qa:
        groups[row["intent_group_id"]].append(row)

    r.check(g, "query rows", 1_521, len(qa))
    r.check(g, "distinct cells", 507, len(cells))
    r.check(g, "intent groups", 507, len(groups))
    r.check(g, "rows = groups × 3 registers", len(qa), len(groups) * 3)
    r.check(g, "documents represented", 43, len({row["doc_id"] for row in qa}))
    r.check(g, "categories represented", 26, len({row["category_id"] for row in qa}))
    r.check(g, "every row is split=core", 1_521,
            sum(1 for row in qa if row["split"] == "core"))
    r.check(g, "no negatives row survives", 0,
            sum(1 for row in qa if row["split"] != "core"))
    r.check(g, "is_impossible false on every row", 0,
            sum(1 for row in qa if row["is_impossible"]))
    r.check(g, "qa_id unique", len(qa), len({row["qa_id"] for row in qa}))

    registers = Counter(row["question_register"] for row in qa)
    r.check(g, "canonical / semi_formal / naive_layperson", (507, 507, 507),
            (registers["canonical"], registers["semi_formal"],
             registers["naive_layperson"]))
    bad_groups = [gid for gid, rows in groups.items()
                  if {row["question_register"] for row in rows} !=
                  {"canonical", "semi_formal", "naive_layperson"}]
    r.check(g, "every group holds one row per register", 0, len(bad_groups))

    # one answer set per cell, shared by the three register rows
    per_cell = {}
    for row in qa:
        per_cell.setdefault((row["doc_id"], row["category_id"]), row)
    r.check(g, "gold spans", 651,
            sum(len(row["answers"]) for row in per_cell.values()))
    r.check(g, "every row carries at least one answer", 0,
            sum(1 for row in qa if not row["answers"]))

    # ---- gold span integrity — the load-bearing check ---------------------
    g = "gold spans"
    para_by_id = {p["paragraph_id"]: p for p in paras}
    off, missing_para, outside, ambiguous = [], [], [], []
    for row in per_cell.values():
        text = frozen.get(row["doc_id"])
        if text is None:
            continue
        for ans in row["answers"]:
            if text[ans["answer_start"]:ans["answer_end"]] != ans["text"]:
                off.append(row["qa_id"])
                continue
            para = para_by_id.get(ans["paragraph_id"])
            if para is None:
                missing_para.append(ans["paragraph_id"])
                continue
            if not (para["char_start"] <= ans["answer_start"]
                    and ans["answer_end"] <= para["char_end"]):
                outside.append(row["qa_id"])
            if para["text"].count(ans["text"]) > 1:
                ambiguous.append(row["qa_id"])

    r.check(g, "span text == frozen[start:end]", 0, len(off))
    r.check(g, "span's paragraph_id exists", 0, len(missing_para))
    r.check(g, "span lies inside its paragraph", 0, len(outside))
    r.info(g, "spans occurring twice in their paragraph", len(ambiguous))
    if off:
        r.note(g, "first offset mismatches", ", ".join(off[:5]))

    # ---- context windows --------------------------------------------------
    g = "context"
    with_ctx = [row for row in qa if row.get("context_paragraph_ids")]
    r.check(g, "rows carrying a context window", 1_521, len(with_ctx))
    ctx_bad, gold_outside = [], []
    for row in with_ctx:
        text = frozen.get(row["doc_id"])
        if text is None:
            continue
        spans = row.get("context_char_spans") or []
        ids = set(row["context_paragraph_ids"])
        covered = set()
        for start, end in spans:
            for pid in ids:
                para = para_by_id.get(pid)
                if para and start <= para["char_start"] and para["char_end"] <= end:
                    if text[para["char_start"]:para["char_end"]] != para["text"]:
                        ctx_bad.append(row["qa_id"])
                    covered.add(pid)
        if ids - covered:
            ctx_bad.append(row["qa_id"])
        for ans in row["answers"]:
            if not any(s <= ans["answer_start"] and ans["answer_end"] <= e
                       for s, e in spans):
                gold_outside.append(row["qa_id"])
    r.check(g, "context spans reproduce their paragraphs verbatim", 0,
            len(set(ctx_bad)))
    r.check(g, "gold span falls inside its context window", 0, len(set(gold_outside)))

    # ---- provenance -------------------------------------------------------
    g = "provenance"
    passes = Counter(row["annotation_pass"] for row in qa)
    r.check(g, "every row P5-verified", 1_521, passes.get("P5-verified", 0))
    r.check(g, "every row names a human annotator", 1_521,
            sum(1 for row in qa if any("claude" not in a.lower()
                                       for a in row.get("annotator_ids") or [])))
    r.info(g, "cells with a model in annotator_ids",
           sum(1 for row in per_cell.values()
               if any("claude" in a.lower() for a in row.get("annotator_ids") or [])))
    r.info(g, "rows carrying an agreement figure",
           sum(1 for row in qa if row.get("jaccard_agreement") is not None))

    # ---- distractors ------------------------------------------------------
    g = "distractors"
    dpath = QA / "distractors.json"
    if not dpath.is_file():
        r.note(g, "qa/distractors.json", "absent — run --rebuild")
    else:
        recs = load_json(dpath)
        r.check(g, "one record per gold span", 651, len(recs))
        slots = Counter(d["source"] for rec in recs for d in rec["distractors"])
        r.check(g, "slots filled", 1_953, len(recs) * 3 - slots.get("pending", 0))
        r.info(g, "mined / perturbed", f"{slots.get('mined', 0)} / {slots.get('perturbed', 0)}")

        leaked, same_as_gold, dupes = [], [], []
        frozen_norm = {k: norm_ws(v) for k, v in frozen.items()}
        for rec in recs:
            seen = set()
            for d in rec["distractors"]:
                if not d.get("text"):
                    continue
                n = norm_ws(d["text"])
                if n in frozen_norm.get(rec["doc_id"], ""):
                    leaked.append(rec["distractor_id"])
                if n == norm_ws(rec["gold_text"]):
                    same_as_gold.append(rec["distractor_id"])
                if n in seen:
                    dupes.append(rec["distractor_id"])
                seen.add(n)
        r.check(g, "no distractor appears in its source document", 0, len(set(leaked)))
        r.check(g, "no distractor equals its gold span", 0, len(set(same_as_gold)))
        r.check(g, "no duplicate distractors within a record", 0, len(set(dupes)))

    # ---- blind study ------------------------------------------------------
    g = "blind study"
    gcsv, kcsv = BLIND / "BLIND_Akshay_Kumar.csv", BLIND / "BLIND_Kanishka.csv"
    if not (gcsv.is_file() and kcsv.is_file()):
        r.note(g, "annotation returns", "absent — cannot recompute agreement")
    else:
        pre = QA / "qa.json.pre-negdrop.bak"
        labels = {}
        source = load_json(pre) if pre.is_file() else qa
        for row in source:
            labels[(row["doc_id"], row["category_id"])] = (not row["is_impossible"],
                                                           row["split"])

        def read(path: Path) -> dict[str, bool]:
            out = {}
            with open(path, encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    v = row["verdict__present_or_absent"].strip().lower()
                    out[row["cell_id"]] = v.startswith("p")
            return out

        A, B = read(gcsv), read(kcsv)
        meta = {}
        with open(gcsv, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                meta[row["cell_id"]] = labels.get((row["doc_id"], row["category_id"]))

        r.check(g, "both annotators returned identical cell sets", True, set(A) == set(B))
        r.check(g, "cells judged", 60, len(A))

        def kappa(x: list[bool], y: list[bool]) -> tuple[float, float, float]:
            n = len(x)
            po = sum(1 for a, b in zip(x, y) if a == b) / n
            px, py = sum(x) / n, sum(y) / n
            pe = px * py + (1 - px) * (1 - py)
            return po, pe, (po - pe) / (1 - pe)

        for label, keep in (("all 60", lambda m: True),
                            ("core only", lambda m: m and m[1] == "core")):
            ids = [c for c in A if keep(meta.get(c))]
            if len(ids) < 2:
                continue
            a = [A[c] for c in ids]
            b = [B[c] for c in ids]
            d = [meta[c][0] for c in ids]
            _, _, kab = kappa(a, b)
            _, _, kga = kappa(d, a)
            _, _, kkb = kappa(d, b)
            r.info(g, f"[{label}] n", len(ids))
            r.info(g, f"[{label}] annotator vs annotator κ", f"{kab:.4f}")
            r.info(g, f"[{label}] dataset vs Akshay Kumar / Kanishka κ", f"{kga:.4f} / {kkb:.4f}")
            if label == "all 60":
                r.check(g, "reproduces published κ (annotators)", "0.8020", f"{kab:.4f}")
                r.check(g, "reproduces published κ (vs Akshay Kumar)", "0.6284", f"{kga:.4f}")
                r.check(g, "reproduces published κ (vs Kanishka)", "0.5066", f"{kkb:.4f}")

        key = BLIND / "_SEALED_KEY.do-not-open.json"
        if key.is_file():
            k = load_json(key)
            sealed = set(k.get("cells", {}) if isinstance(k.get("cells"), dict)
                         else {c.get("cell_id") for c in k.get("cells", [])})
            if sealed and not (sealed & set(A)):
                r.note(g, "sealed key",
                       f"holds {len(sealed)} cells, none matching the {len(A)} scored "
                       "— --score will KeyError, see REPRODUCE.md §Open")

    # ---- hashes -----------------------------------------------------------
    g = "hashes"
    for rel in ("qa/qa.json", "qa/qa.jsonl", "corpus/documents.jsonl",
                "corpus/paragraphs.jsonl", "corpus/categories.jsonl"):
        p = ROOT / rel
        if p.is_file():
            r.info(g, rel, sha256(p)[:16])
    snap = ROOT / "SNAPSHOT.txt"
    if snap.is_file():
        recorded = dict(
            (line.split()[1], line.split()[0])
            for line in snap.read_text(encoding="utf-8").splitlines()
            if len(line.split()) == 2 and len(line.split()[0]) == 64
        )
        stale = []
        for rel, want in recorded.items():
            p = ROOT / rel
            if p.is_file() and sha256(p) != want:
                stale.append(rel)
            elif not p.is_file():
                stale.append(f"{rel} (missing)")
        r.info(g, "SNAPSHOT.txt entries", len(recorded))
        if stale:
            r.note(g, "SNAPSHOT.txt drifted", f"{len(stale)} of {len(recorded)}: "
                   + ", ".join(stale[:4]) + (" …" if len(stale) > 4 else ""))
        if "qa/qa.json" not in recorded:
            r.note(g, "SNAPSHOT.txt coverage",
                   "does not hash qa/qa.json, the released artefact")

    return r.emit()


# --------------------------------------------------------------------------- #
# REBUILD / ANNOTATE
# --------------------------------------------------------------------------- #
REBUILD = [
    ("apply_category_scope.py", ["--apply"], "scope the grid by instrument type"),
    ("build_qa.py", [], "cells → query rows, three registers per core cell"),
    ("add_context_paragraphs.py", ["--apply"], "context window on every row"),
    ("drop_unanswerable.py", ["--apply"], "remove is_impossible rows"),
    ("mine_distractors.py", ["--apply"], "slot 1 — mined from the corpus"),
    ("merge_perturbations.py", ["--apply"], "slots 2–3 — merge perturbations"),
    ("run_validation.py", [], "the seven validators"),
]

# propose_v3.py resolves quotes to offsets itself, in resolve_span(); the
# standalone resolve_spans.py is an interactive one-span tool, not a stage.
ANNOTATE = [
    ("propose_v3.py", ["--all", "--k", "1"], "prompt Opus 5 for every cell"),
    ("split_straddling_spans.py", ["--apply"], "cut spans that cross a paragraph break"),
]

# These ignore argv entirely — no argparse, no --help, no --dry-run. Invoking
# one to "see its options" RUNS it, and against an incomplete checkout it
# rewrites its report with zeros rather than failing. Found the hard way.
NO_ARGPARSE = {"apply_category_scope.py", "build_qa.py", "run_validation.py",
               "build_annotation_workspace.py", "normalize_corpus.py",
               "segment_paragraphs.py", "build_categories.py"}


def run_stages(stages: list[tuple[str, list[str], str]], dry: bool) -> int:
    failed = 0
    for script, flags, why in stages:
        path = ROOT / "scripts" / script
        cmd = [sys.executable, str(path), *flags]
        print(f"\n  ── {script} {' '.join(flags)}")
        print(f"     {why}")
        if not path.is_file():
            print(f"     SKIPPED — {script} not present in scripts/")
            failed += 1
            continue
        if script in NO_ARGPARSE:
            print("     note: takes no flags and cannot be dry-run — it writes on invocation")
        if dry:
            print(f"     would run: {' '.join(cmd[1:])}")
            continue
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if code != 0:
            print(f"     FAILED (exit {code})")
            failed += 1
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="read-only checks against the frozen artefacts (default)")
    mode.add_argument("--rebuild", action="store_true",
                      help="re-derive files from the freeze; rewrites in place")
    mode.add_argument("--annotate", action="store_true",
                      help="re-run the prompted stages; costs money, not deterministic")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --rebuild/--annotate, print the stages without running")
    args = ap.parse_args()

    if args.rebuild:
        print("REBUILD — re-deriving from the freeze. Files are rewritten in place.")
        failed = run_stages(REBUILD, args.dry_run)
        print(f"\n  {len(REBUILD) - failed} of {len(REBUILD)} stages ok")
        print("  Now run --verify to confirm the rebuild reproduces the release.")
        return 1 if failed else 0

    if args.annotate:
        print("ANNOTATE — this calls a paid API and does NOT reproduce the release.")
        print("Treat the output as a re-annotation to compare against.")
        failed = run_stages(ANNOTATE, args.dry_run)
        return 1 if failed else 0

    print("IndiCUAD — verifying the release against the frozen artefacts")
    print(f"root: {ROOT}")
    return verify()


if __name__ == "__main__":
    sys.exit(main())
