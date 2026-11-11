"""Release tables (D40): the pre-Gate-B release never contains NFHS-6 values."""

import hashlib
import json

import pandas as pd
import pytest

from nfhs_forecast import release
from tests.conftest import REPO

REL = REPO / "outputs" / "release"


def test_committed_release_has_no_nfhs6():
    m = json.loads((REL / "release_manifest.json").read_text(encoding="utf-8"))
    p = pd.read_csv(REL / "harmonised_panel.csv", usecols=["round"])
    if not m["includes_nfhs6"]:
        assert set(p["round"]) == {4, 5}
    for f, h in m["sha256"].items():
        assert hashlib.sha256((REL / f).read_bytes()).hexdigest() == h, f


def test_release_consistency():
    p = pd.read_csv(REL / "harmonised_panel.csv")
    u = pd.read_csv(REL / "units.csv")
    cb = pd.read_csv(REL / "codebook.csv")
    assert set(p.unit_uid) == set(u.unit_uid)
    assert set(p.indicator_id) == set(cb.indicator_id)
    assert not p.duplicated(["unit_uid", "indicator_id", "round"]).any()
    ok = p.dropna(subset=["se", "se_sampling"])
    assert ((ok.se ** 2 - ok.se_sampling ** 2 - ok.extra_var).abs() < 1e-3 * (1 + ok.se ** 2)).all()


@pytest.mark.skipif(not (REPO / "data" / "processed" / "panel_model.parquet").exists(),
                    reason="needs data/processed from run_pipeline.py")
def test_make_release_rounds(cfg, tmp_path):
    out = release.make_release(cfg, include_nfhs6=False, out_dir=tmp_path)
    assert set(pd.read_csv(out / "harmonised_panel.csv", usecols=["round"])["round"]) == {4, 5}
    assert "Rounds 4 and 5 only" in (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
