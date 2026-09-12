"""Tests for the evaluation runner in code/evaluation/run_all.py."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "evaluation"))

import run_all  # noqa: E402

DATASET = os.path.join(ROOT, "dataset")


# ---------------------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("summary,expected", [
    ("77 passed, 4 xfailed in 1.10s", {"passed": 77, "xfailed": 4}),
    ("1 failed, 76 passed in 0.50s", {"failed": 1, "passed": 76}),
    ("2 passed, 1 skipped in 0.10s", {"passed": 2, "skipped": 1}),
    ("no tests ran in 0.01s", {}),
])
def test_parse_pytest_summary(summary, expected):
    assert run_all._parse_pytest(summary) == expected


def test_verdict_flags_each_failing_stage():
    report = {"pytest": {"ok": False}, "samples": {"ok": True}, "adversarial": {"ok": True},
              "output_validation": {"ok": True}}
    ok, failed = run_all._verdict(report, run_all.ALL_STAGES, with_support=False)
    assert not ok and failed == ["pytest"]


def test_verdict_flags_support_failures():
    report = {"output_validation": {"ok": True},
              "support": {"log": {"ok": False}, "package": {"ok": True}}}
    ok, failed = run_all._verdict(report, ("output",), with_support=True)
    assert not ok and failed == ["support:log"]


def test_verdict_passes_when_every_stage_is_green():
    report = {"pytest": {"ok": True}, "samples": {"ok": True}, "adversarial": {"ok": True},
              "output_validation": {"ok": True}}
    ok, failed = run_all._verdict(report, run_all.ALL_STAGES, with_support=False)
    assert ok and failed == []


def test_unknown_stage_is_rejected():
    assert run_all.main(["--stages", "pytest,nonsense"]) == 2


# ---------------------------------------------------------------------------------------
# output preparation
# ---------------------------------------------------------------------------------------

def test_ensure_output_keeps_an_existing_file(tmp_path):
    out = tmp_path / "output.csv"
    out.write_text("request_id\n", encoding="utf-8")
    res = run_all.ensure_output(str(out), DATASET, "requests.csv", regenerate=False, allow_model=False)
    assert res == {"regenerated": False, "path": str(out), "reason": "already present"}


def test_ensure_output_regenerates_when_missing(tmp_path):
    out = tmp_path / "output.csv"
    res = run_all.ensure_output(str(out), DATASET, "requests.csv", regenerate=False, allow_model=False)
    assert res["regenerated"] and out.exists()
    assert res["rows"] == 250
    assert not res["model_calls"]


def test_ensure_output_honours_regenerate(tmp_path):
    out = tmp_path / "output.csv"
    out.write_text("request_id\nstale\n", encoding="utf-8")
    res = run_all.ensure_output(str(out), DATASET, "requests.csv", regenerate=True, allow_model=False)
    assert res["regenerated"]
    assert out.read_text(encoding="utf-8").count("\n") > 2


# ---------------------------------------------------------------------------------------
# the runner itself
# ---------------------------------------------------------------------------------------

def test_adversarial_stage_reports_every_case(tmp_path):
    report = run_all.run(stages=("adversarial",), dataset=DATASET,
                         output=str(tmp_path / "out.csv"), requests_file="requests.csv")
    adv = report["adversarial"]
    assert adv["total"] > 0
    assert adv["passed"] == adv["total"], adv.get("failures")
    assert adv["ok"]


def test_output_stage_validates_the_full_dataset(tmp_path):
    out = tmp_path / "out.csv"
    report = run_all.run(stages=("output",), dataset=DATASET, output=str(out),
                         requests_file="requests.csv")
    res = report["output_validation"]
    assert res["ok"], res["problems"]
    assert res["rows"] == res["expected_rows"] == 250
    assert report["ok"], report["failed_stages"]


def test_report_is_written_and_reloadable(tmp_path):
    report = run_all.run(stages=("adversarial",), dataset=DATASET,
                         output=str(tmp_path / "out.csv"), requests_file="requests.csv")
    run_all.write_reports(report)
    import json
    path = os.path.join(run_all.REPORTS, "run_all_report.json")
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["ok"] is True
    assert "adversarial" in loaded
    assert os.path.exists(os.path.join(run_all.REPORTS, "run_all_report.md"))


def test_pytest_stage_can_select_a_single_target(tmp_path):
    """Target a synthetic file: pointing the stage at code/tests would re-enter this suite."""
    leaf = tmp_path / "test_leaf.py"
    leaf.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    report = run_all.run(stages=("pytest",), dataset=DATASET, output=str(tmp_path / "out.csv"),
                         pytest_targets=(str(leaf),))
    assert report["pytest"]["ok"], report["pytest"].get("output_tail")
    assert report["pytest"]["counts"].get("passed") == 1


def test_pytest_stage_reports_a_failure(tmp_path):
    leaf = tmp_path / "test_leaf.py"
    leaf.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    report = run_all.run(stages=("pytest",), dataset=DATASET, output=str(tmp_path / "out.csv"),
                         pytest_targets=(str(leaf),))
    assert not report["pytest"]["ok"]
    assert not report["ok"] and report["failed_stages"] == ["pytest"]


def test_failed_stage_makes_the_run_fail(tmp_path):
    """A missing output with validation switched on must fail rather than pass silently."""
    res = run_all.stage_output_validation(str(tmp_path / "absent.csv"), DATASET, "requests.csv", False)
    assert not res["ok"]
    assert any("does not exist" in p for p in res["problems"])
