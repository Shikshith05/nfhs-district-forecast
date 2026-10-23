"""End-to-end DRYRUN of the post-run pipeline (D39).

Runs scripts/post_run.py --dryrun on the committed NFHS-4 -> NFHS-5 LOSO predictions
(stand-in in the NFHS-5 -> NFHS-6 format) into a temporary folder, with fewer
bootstrap replicates for speed. No NFHS-6 value is involved.
"""

import importlib.util
import json
import re

import pandas as pd
import pytest
import yaml

from nfhs_forecast import report_build as rb
from nfhs_forecast.report_check import check_report
from tests.conftest import REPO


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dry")
    raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {k: str((REPO / v).resolve()) for k, v in raw["paths"].items()}
    raw["post_run"].update({"bootstrap_reps": 60, "break_bootstrap_reps": 60})
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp / "dryrun"
    code = _script("post_run").main(["--config", str(cfg_path), "--dryrun", "--out-dir", str(out)])
    return code, out, cfg_path


def test_dryrun_completes_and_report_is_filled(dry, cfg):
    code, out, _ = dry
    assert code == 0
    report = out / "DRYRUN_report.md"
    text = report.read_text(encoding="utf-8")
    assert "mode=dryrun" in text[:300] and "DRYRUN — NOT NFHS-6 RESULTS" in text
    assert not rb.SLOT_RE.search(text) and rb.PENDING not in text           # no unfilled placeholders
    values = json.loads((out / "DRYRUN_report_values.json").read_text(encoding="utf-8"))
    assert all(v["text"] != rb.PENDING for v in values["slots"].values())
    from nfhs_forecast.config import load_config
    assert check_report(report, load_config(dry[2]), "dryrun") == []


def test_every_dryrun_file_is_marked(dry):
    _, out, _ = dry
    files = list(out.iterdir())
    assert len(files) >= 30
    assert all(f.name.startswith("DRYRUN_") for f in files)
    for f in files:
        if f.suffix == ".csv":
            assert pd.read_csv(f).run_mode.eq("DRYRUN").all(), f.name
        elif f.suffix == ".json" and "values" not in f.name:
            assert json.loads(f.read_text(encoding="utf-8"))["run_mode"] == "DRYRUN"
        elif f.suffix == ".md":
            assert "DRYRUN" in f.read_text(encoding="utf-8")[:600]
    assert pd.read_parquet(out / "DRYRUN_predictions_nfhs5to6.parquet").run_mode.eq("DRYRUN").all()


def test_real_mode_refuses_dryrun_predictions(dry):
    _, out, cfg_path = dry
    code = _script("post_run").main(["--config", str(cfg_path), "--predictions",
                                     str(out / "DRYRUN_predictions_nfhs5to6.parquet"), "--out-dir", str(out / "x")])
    assert code == 3 and not (out / "x").exists()


def test_real_report_cannot_be_built_from_dryrun_folder(dry, cfg):
    _, out, _ = dry
    with pytest.raises(ValueError, match="DRYRUN"):
        rb.build(cfg, "final", out_dir=out)


def test_real_report_never_references_dryrun_files():
    for path in (REPO / "docs" / "report.md", REPO / "docs" / "report_values.json",
                 REPO / "docs" / "report_template.md"):
        assert not re.search(r"dryrun", path.read_text(encoding="utf-8"), re.I), path.name
