"""Report rendering and the number checker (D38). Uses the committed NFHS-4 -> NFHS-5 outputs only."""

import json
import re

import pytest

from nfhs_forecast import report_build as rb
from nfhs_forecast.report_check import check_report
from tests.conftest import REPO


@pytest.fixture(scope="module")
def prerun(cfg, tmp_path_factory):
    """A fresh prerun render written next to docs/ so relative figure paths resolve."""
    report, values = rb.build(cfg, "prerun")
    return report, json.loads(values.read_text(encoding="utf-8"))


def _variant(report, tmp_name, text):
    p = report.parent / tmp_name
    p.write_text(text, encoding="utf-8")
    return p


def test_template_slots_all_defined():
    template = (REPO / "docs" / "report_template.md").read_text(encoding="utf-8")
    used = set(rb.SLOT_RE.findall(template))
    assert used and used <= set(rb.SLOTS), sorted(used - set(rb.SLOTS))


def test_prerun_report_checks_clean(cfg, prerun):
    report, values = prerun
    assert check_report(report, cfg, "prerun") == []
    text = report.read_text(encoding="utf-8")
    pending = {n for n, v in values["slots"].items() if v["text"] == rb.PENDING}
    assert pending and all(rb.SLOTS[n].get("src", "").startswith("run:") for n in pending)
    assert not rb.SLOT_RE.search(text)


@pytest.mark.parametrize("edit,expect", [
    (lambda t: re.sub(r"(<!--slot:s45_rtm-->)[^<]+", r"\g<1>0.700", t, count=1), "differs"),
    (lambda t: t.replace("## 1. Motivation", "## 1. Motivation\n\nRTM skill was 0.613 in our test."), "prose"),
    (lambda t: t + "\nSee outputs/dryrun/DRYRUN_report.md\n", "DRYRUN"),
    (lambda t: t + "\n{{s56_rtm}}\n", "unfilled"),
])
def test_checker_catches_tampering(cfg, prerun, edit, expect):
    report, _ = prerun
    p = _variant(report, "_tmp_tampered_report.md", edit(report.read_text(encoding="utf-8")))
    try:
        problems = check_report(p, cfg, "prerun")
    finally:
        p.unlink()
    assert any(expect.lower() in x.lower() for x in problems), problems


def test_pending_not_allowed_in_final_mode(cfg, prerun):
    report, _ = prerun
    p = _variant(report, "_tmp_final_report.md", report.read_text(encoding="utf-8"))
    try:
        problems = check_report(p, cfg, "final")
    finally:
        p.unlink()
    assert any("PENDING" in x for x in problems)


def test_committed_report_never_references_dryrun():
    """The real report and its value manifest must never point at a stand-in (DRYRUN) file."""
    for name in ("report.md", "report_values.json"):
        path = REPO / "docs" / name
        if path.exists():
            assert "dryrun" not in path.read_text(encoding="utf-8").lower(), name
