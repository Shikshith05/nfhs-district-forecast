"""Independent parser for the IIPS NFHS-6 district fact-sheet PDFs (Task A4; DECISIONS D33).

The pipeline uses the community compilation of NFHS-6 (Patra 2026,
data/raw/nfhs6/all_india_nfhs6_district.csv). This module re-reads the
official IIPS PDFs so that a random sample of district x indicator values can
be checked against them (scripts/spot_check_nfhs6.py).

Nothing here guesses a value:
  * text is taken from the PDF's text layer (pypdf); a page without one
    (a scan) is reported, not OCR-ed;
  * an indicator row is accepted only if its last `len(value_columns)` tokens
    are value tokens (number, (number), *, na) AND the token before them is
    not a number, so a row whose layout is ambiguous is flagged
    `ambiguous` and excluded from comparisons;
  * the district a PDF belongs to comes from a manifest written by a human
    (data/raw/pdfs/manifest.csv: file, state, district), never from the
    file's content.

Cell markers follow the fact sheets: '*' = suppressed (< 25 unweighted
cases), '(x)' = based on 25-49 unweighted cases, 'na' = not available.

Leakage contract: parsed NFHS-6 values are used only to audit the NFHS-6
source file. They never reach training, tuning or selection.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .clean import normalise_indicator_text, normalise_name, squash

VALUE_TOKEN = r"\(\s*\d+(?:\.\d+)?\s*\)|\d+(?:\.\d+)?|\*|na|NA|n\.a\.|N\.A\."
_VALUE_RE = re.compile(rf"^(?:{VALUE_TOKEN})$")
_ROW_START = re.compile(r"^\s*(?P<no>\d{1,3})\s*\.\s+(?P<text>\S.*)$")
_NUMBER = re.compile(r"^\(?\d+(?:\.\d+)?\)?$")
DEFAULT_VALUE_COLUMNS = ("nfhs6", "nfhs5")


def parse_value(token: str) -> tuple[float, bool, bool, bool]:
    """(value, suppressed, small_sample, not_available) for one fact-sheet cell."""
    t = token.strip()
    if t == "*":
        return np.nan, True, False, False
    if t.lower() in ("na", "n.a."):
        return np.nan, False, False, True
    small = t.startswith("(") and t.endswith(")")
    try:
        return float(t.strip("() ")), False, small, False
    except ValueError:
        return np.nan, False, False, False


def _tokens(line: str) -> list[str]:
    # keep "( 12.3 )" together as one token
    line = re.sub(r"\(\s*(\d+(?:\.\d+)?)\s*\)", r"(\1)", line)
    return line.split()


def parse_fact_sheet_text(text: str, value_columns=DEFAULT_VALUE_COLUMNS) -> pd.DataFrame:
    """Parse the indicator table of one fact sheet from its extracted text.

    A record starts at a line '<number>. <text>' and may wrap over following
    lines; it ends when the joined text ends in len(value_columns) value tokens.
    Returns one row per record: indicator_no, indicator, <col>_raw / <col>_value /
    <col>_suppressed / <col>_small_sample / <col>_na for each value column, and
    parse_status ('ok' or 'ambiguous' or 'incomplete').
    """
    k = len(value_columns)
    records, cur = [], None

    def close(rec, status=None):
        toks = _tokens(rec["text"])
        row = {"indicator_no": int(rec["no"]), "line_no": rec["line"]}
        if len(toks) > k and all(_VALUE_RE.match(t) for t in toks[-k:]):
            body = toks[:-k]
            row["indicator"] = " ".join(body)
            row["parse_status"] = "ambiguous" if _NUMBER.match(body[-1]) else "ok"
            for col, tok in zip(value_columns, toks[-k:]):
                v, sup, small, na = parse_value(tok)
                row.update({f"{col}_raw": tok, f"{col}_value": v, f"{col}_suppressed": sup,
                            f"{col}_small_sample": small, f"{col}_na": na})
        else:
            row["indicator"] = " ".join(toks)
            row["parse_status"] = status or "incomplete"
        records.append(row)

    def complete(rec) -> bool:
        toks = _tokens(rec["text"])
        return len(toks) > k and all(_VALUE_RE.match(t) for t in toks[-k:])

    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = _ROW_START.match(line)
        if m:                      # a new numbered row closes an unfinished one as 'incomplete'
            if cur is not None:
                close(cur)
            cur = {"no": m.group("no"), "text": m.group("text"), "line": i}
        elif cur is not None and not complete(cur):
            cur["text"] += " " + line
        if cur is not None and complete(cur):
            close(cur)
            cur = None
    if cur is not None:
        close(cur)
    out = pd.DataFrame(records)
    for col in value_columns:
        for suffix in ("_raw", "_value", "_suppressed", "_small_sample", "_na"):
            if f"{col}{suffix}" not in out.columns:
                out[f"{col}{suffix}"] = np.nan
    return out


def extract_pdf_text(path: str | Path) -> tuple[str, int]:
    """Text layer of a PDF (pypdf) and its page count. No OCR."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages), len(pages)


def parse_district_pdf(path: str | Path, value_columns=DEFAULT_VALUE_COLUMNS,
                       min_text_chars: int = 200) -> tuple[pd.DataFrame, dict]:
    """Parse one IIPS NFHS-6 district fact sheet. Returns (rows, info)."""
    text, n_pages = extract_pdf_text(path)
    info = {"file": Path(path).name, "pages": n_pages, "text_chars": len(text.strip())}
    if info["text_chars"] < min_text_chars:
        info["status"] = "no_text_layer"
        return parse_fact_sheet_text("", value_columns), info
    rows = parse_fact_sheet_text(text, value_columns)
    info["status"] = "parsed"
    info["rows"] = len(rows)
    info["rows_ok"] = int((rows.parse_status == "ok").sum()) if len(rows) else 0
    return rows, info


def read_manifest(pdf_dir: str | Path) -> pd.DataFrame:
    """data/raw/pdfs/manifest.csv: file, state, district (names as in the NFHS-6 CSV)."""
    path = Path(pdf_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: list each PDF with its state and district "
                                "(columns file,state,district; see docs/NFHS6_PDF_CHECK.md)")
    m = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    missing = {"file", "state", "district"} - set(m.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return m


def parse_pdf_folder(pdf_dir: str | Path, value_columns=DEFAULT_VALUE_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse every PDF listed in the manifest. Returns (rows, per-file info)."""
    pdf_dir = Path(pdf_dir)
    manifest = read_manifest(pdf_dir)
    rows, infos = [], []
    for r in manifest.itertuples():
        path = pdf_dir / r.file
        if not path.exists():
            infos.append({"file": r.file, "state": r.state, "district": r.district, "status": "file_missing"})
            continue
        try:
            df, info = parse_district_pdf(path, value_columns)
        except Exception as e:  # a broken PDF must not stop the check; it is reported
            infos.append({"file": r.file, "state": r.state, "district": r.district,
                          "status": f"error: {type(e).__name__}: {e}"})
            continue
        infos.append({**info, "state": r.state, "district": r.district})
        rows.append(df.assign(file=r.file, state=r.state, district=r.district))
    parsed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return parsed, pd.DataFrame(infos)


# ---------------------------------------------------------------------
# Matching parsed rows to the community CSV
# ---------------------------------------------------------------------
def indicator_match_key(text: str) -> str:
    return squash(normalise_indicator_text(text))


def match_indicator(csv_text: str, candidates: pd.DataFrame, min_prefix: int = 25) -> tuple[int | None, str]:
    """Row index in `candidates` (one district's parsed rows) for an indicator text.

    exact normalised text, else a unique prefix match of >= min_prefix chars
    (truncated texts). Returns (index or None, method)."""
    key = indicator_match_key(csv_text)
    keys = candidates.indicator.map(indicator_match_key)
    exact = candidates.index[keys == key]
    if len(exact) == 1:
        return int(exact[0]), "exact"
    if len(exact) > 1:
        return None, "ambiguous_text"
    pref = candidates.index[[(len(key) >= min_prefix and k.startswith(key)) or
                             (len(k) >= min_prefix and key.startswith(k)) for k in keys]]
    if len(pref) == 1:
        return int(pref[0]), "prefix"
    return None, "no_match" if len(pref) == 0 else "ambiguous_text"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def compare_pairs(sample: pd.DataFrame, parsed: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """Compare sampled community-CSV rows with the parsed PDF rows of the same district.

    sample: state, district, indicator_no, indicator, nfhs6_value, nfhs5_value (raw CSV strings ok).
    A pair agrees on a round if both values are equal within `tol`, or both are
    suppressed ('*'), or both not available. Pairs that cannot be matched
    unambiguously get status != 'compared' and no agreement value.
    """
    from .clean import clean_value
    out = []
    pk = parsed.assign(_s=parsed.state.map(normalise_name), _d=parsed.district.map(normalise_name))
    for r in sample.itertuples():
        rec = {"state": r.state, "district": r.district, "indicator_no": r.indicator_no,
               "indicator": r.indicator, "csv_nfhs6": r.nfhs6_value, "csv_nfhs5": r.nfhs5_value}
        cand = pk[(pk._s == normalise_name(r.state)) & (pk._d == normalise_name(r.district))]
        if cand.empty:
            out.append({**rec, "status": "district_not_parsed"})
            continue
        idx, method = match_indicator(r.indicator, cand)
        if idx is None:
            out.append({**rec, "status": f"indicator_{method}"})
            continue
        row = cand.loc[idx]
        rec.update(match_method=method, pdf_indicator=row.indicator, pdf_indicator_no=row.indicator_no,
                   pdf_nfhs6=row.get("nfhs6_raw"), pdf_nfhs5=row.get("nfhs5_raw"))
        if row.parse_status != "ok":
            out.append({**rec, "status": f"pdf_row_{row.parse_status}"})
            continue
        for col, csv_raw in (("nfhs6", r.nfhs6_value), ("nfhs5", r.nfhs5_value)):
            c_sup = "*" in str(csv_raw)
            c_val = clean_value(csv_raw)
            p_val, p_sup, p_na = row[f"{col}_value"], bool(row[f"{col}_suppressed"]), bool(row[f"{col}_na"])
            if c_sup or p_sup:
                agree = c_sup and p_sup
            elif np.isnan(c_val) or np.isnan(p_val):
                agree = bool(np.isnan(c_val) and np.isnan(p_val))
            else:
                agree = abs(c_val - p_val) <= tol
            rec[f"agree_{col}"] = bool(agree)
            rec[f"diff_{col}"] = (c_val - p_val) if not (np.isnan(c_val) or np.isnan(p_val)) else np.nan
            rec[f"pdf_small_sample_{col}"] = bool(row[f"{col}_small_sample"])
        rec["status"] = "compared"
        out.append(rec)
    return pd.DataFrame(out)
