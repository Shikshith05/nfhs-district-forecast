"""Spatial neighbours from Census-2011 district polygons (DECISIONS D32).

Two Census-2011 districts are neighbours if their polygons in
Districts.geojson share at least `min_shared` vertices (coordinates rounded to
`digits` decimals). Districts with no such neighbour (islands) use their k
nearest centroids. A forecasting unit is linked to a polygon by its Census
2011 code (India.csv `district_code`); units created after 2011 have no code
and get no spatial features.

Leakage contract: only geometry and round-t values are used. Neighbour means
are computed on the full round-t cross-section, never on round t+1.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def norm_code(x) -> str | None:
    """Census code as a plain integer string ('532', 532.0 -> '532'); None if missing."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return str(int(float(str(x).strip())))
    except ValueError:
        return None


def _rings(geom: dict):
    if geom["type"] == "Polygon":
        yield from geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield from poly


def adjacency_from_features(features: list[dict], digits: int = 5, min_shared: int = 2,
                            k_fallback: int = 3) -> dict[str, set[str]]:
    """{census code: set of neighbouring codes} from GeoJSON features."""
    vertex_codes: dict[tuple, set] = defaultdict(set)
    centroid: dict[str, np.ndarray] = {}
    for f in features:
        code = norm_code(f["properties"].get("censuscode"))
        if code is None or f.get("geometry") is None:
            continue
        pts = []
        for ring in _rings(f["geometry"]):
            for x, y in ring:
                vertex_codes[(round(x, digits), round(y, digits))].add(code)
                pts.append((x, y))
        if pts:
            centroid[code] = np.mean(pts, axis=0)
    shared = Counter()
    for codes in vertex_codes.values():
        if len(codes) > 1:
            cs = sorted(codes)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    shared[(cs[i], cs[j])] += 1
    adj: dict[str, set[str]] = {c: set() for c in centroid}
    for (a, b), n in shared.items():
        if n >= min_shared:
            adj[a].add(b)
            adj[b].add(a)
    codes = sorted(centroid)
    xy = np.array([centroid[c] for c in codes])
    for i, c in enumerate(codes):
        if not adj[c] and len(codes) > 1:
            d = np.hypot(*(xy - xy[i]).T)
            d[i] = np.inf
            for j in np.argsort(d)[:k_fallback]:
                adj[c].add(codes[j])
                adj[codes[j]].add(c)
    return adj


def load_adjacency(path: str | Path, **kwargs) -> dict[str, set[str]]:
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    return adjacency_from_features(gj["features"], **kwargs)


def neighbour_matrix(codes: pd.Series, adjacency: dict[str, set[str]] | None) -> np.ndarray:
    """Row-normalised 0/1 neighbour matrix between units (rows = codes.index order).

    Unit j is a neighbour of unit i if j's code is in adjacency[i's code] or
    both share the same code (j != i). Rows without neighbours are all zero.
    """
    n = len(codes)
    W = np.zeros((n, n))
    if not adjacency:
        return W
    c = [norm_code(x) for x in codes]
    by_code = defaultdict(list)
    for j, cj in enumerate(c):
        if cj is not None:
            by_code[cj].append(j)
    for i, ci in enumerate(c):
        if ci is None:
            continue
        for nc in adjacency.get(ci, set()) | {ci}:
            for j in by_code.get(nc, []):
                if j != i:
                    W[i, j] = 1.0
    return W


def neighbour_means(values: pd.DataFrame, codes: pd.Series, adjacency) -> pd.DataFrame:
    """Mean of each column over a unit's neighbours, ignoring missing values (NaN if none)."""
    W = neighbour_matrix(codes.reindex(values.index), adjacency)
    V = values.to_numpy(float)
    ok = ~np.isnan(V)
    num = W @ np.where(ok, V, 0.0)
    den = W @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(out, index=values.index, columns=values.columns)
