"""Check every number in a rendered report against the result files (D38).

Checks, by mode (prerun / final / dryrun):
  1. every marked slot re-renders to exactly the text in the report (no stale numbers);
  2. independently of the renderer, every number shown by a scalar slot, and
     every numeric cell of a fixed-row table slot, agrees with its source
     file within the displayed rounding (sources read with default dtypes);
  3. no decimal number or percentage appears in the prose outside a slot
     unless it is a constant from config.yaml (methodology settings, not results);
  4. final / dryrun: no `{{slot}}` and no PENDING marker is left;
     prerun: only NFHS-6 slots may be PENDING;
  5. final / prerun (the real report): nothing refers to a DRYRUN file;
  6. every referenced figure exists.
Integers in the prose (years, section numbers, D-numbers) are not results and
are not checked; every count that is a result is a slot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import report_build as rb

NUM_RE = re.compile(r"(?<![\w.\-])-?\d+\.\d+%?|(?<![\w.\-])-?\d+%")
FIG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
#: any number shown inside a slot (integers too)
VAL_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def _config_constants(cfg_path: Path) -> set[str]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    vals = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            vals.add(float(x))
    walk(raw)
    out = set()
    for v in vals:
        out |= {f"{v:g}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v * 100:g}%", f"{(1 - v) * 100:g}%"}
    return out


def _numbers(text: str) -> list[float]:
    out = []
    for tok in VAL_RE.findall(text):
        out.append(float(tok[:-1]) / 100 if tok.endswith("%") else float(tok))
    return out


def _tol(tok: str) -> float:
    t = tok.rstrip("%")
    d = len(t.split(".")[1]) if "." in t else 0
    tol = 0.5 * 10 ** (-d) + 1e-9
    return tol / 100 if tok.endswith("%") else tol


def _source_value(spec: dict, ctx: "rb.Ctx", where_extra: dict | None = None, col: str | None = None):
    path = ctx.resolve(spec["src"])
    if path.suffix == ".json":
        v = json.loads(path.read_text(encoding="utf-8"))[spec["col"]]
        return len(v) if spec.get("agg") == "len" else v
    df = pd.read_csv(path)
    for k, v in {**spec.get("where", {}), **(where_extra or {})}.items():
        if k.startswith("@"):
            continue
        v = str(v).replace("{primary}", ctx.primary)
        df = df[df[k].astype(str).str.lower() == v.lower()] if df[k].dtype == bool else \
            df[np.isclose(pd.to_numeric(df[k], errors="coerce"), float(v))] if _is_num(v) and df[k].dtype.kind in "fi" \
            else df[df[k].astype(str) == v]
    if spec.get("agg") == "count":
        return len(df)
    return df[col or spec["col"]].iloc[0] if len(df) == 1 else None


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _check_scalar(name, spec, shown, ctx) -> list[str]:
    if spec["fmt"] in ("text", "method"):
        return []
    src = _source_value(spec, ctx)
    toks = VAL_RE.findall(shown)
    if src is None or not toks:
        return [f"{name}: could not verify '{shown}' independently"]
    val = float(src)
    got = _numbers(toks[0])[0]
    if abs(got - val) > _tol(toks[0]):
        return [f"{name}: report shows {shown}, source has {val}"]
    return []


def _check_table(name, spec, shown, ctx) -> list[str]:
    if spec["rows"] is None:
        return []
    keys = spec["row_col"] if isinstance(spec["row_col"], list) else [spec["row_col"]]
    lines = [l for l in shown.splitlines() if l.startswith("|")][2:]
    by_label = {rb.METHOD_LABELS.get(r, r): r for r in spec["rows"]}
    problems = []
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        key = by_label.get(cells[0])
        if key is None:
            problems.append(f"{name}: unknown row '{cells[0]}'")
            continue
        for c, cell in zip(spec["cols"], cells[1:]):
            col, fmt = c[1], c[2]
            extra = dict(c[3]) if len(c) > 3 else {}
            sub = dict(spec)
            if "@src" in extra:
                sub["src"] = extra.pop("@src")
            if fmt == "text" or cell == "n/a":
                continue
            cols = [col[3:] + "_lo", col[3:] + "_hi"] if col.startswith("ci:") else [col]
            toks = VAL_RE.findall(cell)
            for tok, cc in zip(toks, cols):
                v = _source_value(sub, ctx, {keys[0]: key, **extra}, cc)
                if v is None or abs(_numbers(tok)[0] - float(v)) > _tol(tok):
                    problems.append(f"{name}: row {key}, column {cc}: report {tok}, source {v}")
    return problems


def check_report(report: Path, cfg: dict, mode: str) -> list[str]:
    text = Path(report).read_text(encoding="utf-8")
    problems: list[str] = []
    ctx = rb.Ctx(cfg, mode, rb.report_paths(cfg, mode)[2] if mode != "dryrun" else Path(report).parent, Path(report))
    marks = rb.MARK_RE.findall(text)
    if not marks:
        problems.append("no slot markers found")
    for name, shown in marks:
        if name not in rb.SLOTS:
            problems.append(f"unknown slot {name}")
            continue
        spec = rb.SLOTS[name]
        if shown == rb.PENDING:
            if mode != "prerun" or not spec.get("src", "").startswith("run:"):
                problems.append(f"{name}: still PENDING in mode {mode}")
            continue
        try:
            fresh = rb.render_slot(name, ctx)
        except Exception as e:
            problems.append(f"{name}: cannot re-render ({type(e).__name__}: {e})")
            continue
        if fresh != shown:
            problems.append(f"{name}: report text differs from the current result files (stale?)")
            continue
        try:
            if spec["kind"] == "scalar":
                problems += _check_scalar(name, spec, shown, ctx)
            elif spec["kind"] == "table":
                problems += _check_table(name, spec, shown, ctx)
        except Exception as e:
            problems.append(f"{name}: independent check failed ({type(e).__name__}: {e})")
    if rb.SLOT_RE.search(text):
        problems.append(f"unfilled slot(s): {sorted(set(rb.SLOT_RE.findall(text)))[:10]}")
    # prose outside slots: strip markers, comments, code, heading numbers, references section
    prose = rb.MARK_RE.sub(" ", text)
    prose = re.sub(r"<!--.*?-->", " ", prose, flags=re.S)
    prose = re.sub(r"```.*?```", " ", prose, flags=re.S)
    prose = re.sub(r"`[^`]*`", " ", prose)
    prose = re.sub(r"(?m)^(#+\s*)[\d.]+", r"\1", prose)
    prose = prose.split("\n## References")[0]
    allowed = _config_constants(Path(cfg["abs_paths"]["outputs_dir"]).parent / "config.yaml")
    for tok in NUM_RE.findall(prose):
        if tok.lstrip("-") not in allowed:
            problems.append(f"number '{tok}' in the prose is neither a slot value nor a config constant")
    if mode in ("final", "prerun") and re.search(r"dryrun", text, re.I):
        problems.append("the real report refers to a DRYRUN file or text")
    for fig in FIG_RE.findall(text):
        if not (Path(report).parent / fig).exists():
            problems.append(f"figure not found: {fig}")
    return problems
