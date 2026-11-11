"""Check population-weighted aggregation against IIPS's own merged NFHS-5 value for DNH&DD (DECISIONS D41).

Uses NFHS-5 values only (India.csv districts; the NFHS-6 file's nfhs5_value column). The nfhs6_value
column is never read (HARD RULE 1). Usage: python scripts/check_aggregation_dnhdd.py [repo_root]
"""
import re, sys
from pathlib import Path
import pandas as pd
repo = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1])
d = pd.read_csv(f"{repo}/data/raw/nfhs4_5/India.csv", low_memory=False)
d = d[d.State.isin(["Dadra & Nagar Haveli", "Daman & Diu"])]
m = pd.read_csv(f"{repo}/data/raw/nfhs6/dadra_and_nagar_haveli_and_daman_and_diu_nfhs6_district.csv",
                usecols=["district", "indicator", "nfhs5_value"])   # deliberately not reading nfhs6_value
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
d["k"] = d.Indicator.map(norm); m["k"] = m.indicator.map(norm)
d["v"] = pd.to_numeric(d["NFHS 5"], errors="coerce"); m["v"] = pd.to_numeric(m.nfhs5_value, errors="coerce")
piv = d.pivot_table(index="k", columns="District", values="v", aggfunc="first")
merged = m.groupby("k").v.first()
pops = {"total": {"Dadra & Nagar Haveli": 343709, "Daman": 191173, "Diu": 52074},
        "female": {"Dadra & Nagar Haveli": 149949, "Daman": 66514, "Diu": 26432}}
common = piv.dropna().index.intersection(merged.dropna().index)
print("indicators matched with all 3 districts + merged value:", len(common))
out = {}
for basis, p in pops.items():
    w = pd.Series(p)[piv.columns]; w = w / w.sum()
    agg = (piv.loc[common] * w).sum(axis=1)
    err = agg - merged.loc[common]
    out[basis] = err
    print(f"{basis:6s} weights {dict(w.round(3))}: mean abs err {err.abs().mean():.2f} pp, "
          f"median {err.abs().median():.2f}, 90th pct {err.abs().quantile(.9):.2f}, max {err.abs().max():.2f}")
e = pd.DataFrame(out); e["female_better"] = e.female.abs() < e.total.abs()
print("share of indicators where female weights are closer:", round(e.female_better.mean(), 2))
women = [k for k in common if k.startswith("women") or "pregnan" in k or "mothers" in k]
print(f"women/maternal indicators (n={len(women)}): mean abs err total {e.loc[women,'total'].abs().mean():.2f}, "
      f"female {e.loc[women,'female'].abs().mean():.2f}")
print(e.reindex(e.total.abs().sort_values(ascending=False).index).head(8).round(2).to_string())
