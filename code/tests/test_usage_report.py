"""Tests for the usage-report tooling in code/evaluation/write_usage_report.py."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "evaluation"))

import write_usage_report as wur  # noqa: E402

N = 4


def _usage(requests=N, provider="openai", model="deepseek-v4-flash",
           per_model=None, pricing_known=True):
    if per_model is None:
        per_model = {model: {"model": model, "calls": 8, "input_tokens": 100_000,
                             "output_tokens": 20_000, "cache_read_tokens": 5_000,
                             "cost_usd": 0.42, "pricing_known": pricing_known}}
    return {"provider": provider, "model": model, "requests": requests, "usage": per_model,
            "sources": {"message_01": "rules", "image_01": "golden"}, "rejected_evidence": []}


@pytest.fixture
def repo(tmp_path):
    """A miniature dataset + predictions pair the integrity checks can measure."""
    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "requests.csv").write_text("request_id,user_id\n" + "".join(f"r{i},u{i}\n" for i in range(N)),
                                     encoding="utf-8")
    out = tmp_path / "output.csv"
    out.write_text("request_id,amount_safe_to_pay\n" + "".join(f"r{i},{i}\n" for i in range(N)),
                   encoding="utf-8")
    usage_path = tmp_path / "usage_last_run.json"
    return {"dir": ds, "output": str(out), "usage": str(usage_path)}


def _ctx(repo, usage):
    return wur.gather_context(usage, str(repo["dir"]), repo["output"], repo["usage"])


# ---------------------------------------------------------------------------------------
# the mandated figures
# ---------------------------------------------------------------------------------------

REQUIRED_LABELS = ["Model provider(s)", "Model name(s)", "Model calls (total)",
                   "Input tokens (total)", "Output tokens (total)",
                   "Total tokens (input + output)", "Requests processed",
                   "Total tokens per request", "Estimated total cost (USD)",
                   "Estimated cost per request (USD)"]


def test_report_states_every_required_figure(repo):
    usage = _usage()
    text = wur.render(usage, _ctx(repo, usage))
    for label in REQUIRED_LABELS:
        assert label in text, f"report is missing the required row {label!r}"


def test_report_names_the_provider_and_model(repo):
    usage = _usage(provider="openai", model="deepseek-v4-flash-vision-exp")
    text = wur.render(usage, _ctx(repo, usage))
    assert "openai" in text
    assert "deepseek-v4-flash-vision-exp" in text


def test_totals_and_averages_are_computed_across_models(repo):
    usage = _usage(model="m-a", per_model={
        "m-a": {"model": "m-a", "calls": 3, "input_tokens": 1000, "output_tokens": 100,
                "cache_read_tokens": 0, "cost_usd": 1.0, "pricing_known": True},
        "m-b": {"model": "m-b", "calls": 1, "input_tokens": 4000, "output_tokens": 900,
                "cache_read_tokens": 0, "cost_usd": 2.0, "pricing_known": True},
    })
    text = wur.render(usage, _ctx(repo, usage))
    assert "**4**" in text                       # total calls
    assert "**5,000**" in text                   # total input tokens
    assert "**1,000**" in text                   # total output tokens
    assert "**6,000**" in text                   # total tokens
    assert f"{6000 / N:,.2f}" in text            # tokens per request
    assert "3.000000" in text                    # total cost


def test_zero_call_run_is_reported_cleanly(repo):
    usage = _usage(provider="none", model="none", per_model={})
    text = wur.render(usage, _ctx(repo, usage))
    assert "no model calls in this run" in text
    assert "0.000000" in text


def test_unpriced_model_is_flagged(repo):
    usage = _usage(model="mystery", per_model={
        "mystery": {"model": "mystery", "calls": 1, "input_tokens": 10, "output_tokens": 1,
                    "cache_read_tokens": 0, "cost_usd": 0.0, "pricing_known": False}})
    text = wur.render(usage, _ctx(repo, usage))
    assert "pricing was NOT configured for: mystery" in text
    assert any(not ok for ok, _ in wur.integrity(usage, _ctx(repo, usage)))


def test_report_never_contains_a_credential(repo):
    usage = _usage()
    text = wur.render(usage, _ctx(repo, usage))
    assert "API_KEY" not in text or "environment variables" in text
    for marker in ("sk-ant-", "sk-", "Bearer ey"):
        assert marker not in text


# ---------------------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------------------

def test_integrity_passes_on_a_full_dataset_run(repo):
    usage = _usage()
    assert all(ok for ok, _ in wur.integrity(usage, _ctx(repo, usage)))


def test_a_sample_run_is_rejected_as_not_the_full_dataset(repo):
    usage = _usage(requests=N - 1)
    bad = [msg for ok, msg in wur.integrity(usage, _ctx(repo, usage)) if not ok]
    assert any("not the final full-dataset run" in m for m in bad)


def test_a_missing_output_csv_fails_integrity(repo):
    usage = _usage()
    ctx = wur.gather_context(usage, str(repo["dir"]), str(repo["dir"] / "nope.csv"), repo["usage"])
    assert any(not ok for ok, _ in wur.integrity(usage, ctx))


def test_provider_configured_but_zero_calls_fails_integrity(repo):
    usage = _usage(provider="openai", model="m", per_model={})
    assert any(not ok for ok, _ in wur.integrity(usage, _ctx(repo, usage)))


def test_check_detects_a_stale_report(repo):
    usage = _usage()
    with open(repo["usage"], "w", encoding="utf-8") as fh:
        json.dump(usage, fh)
    report = os.path.join(os.path.dirname(repo["usage"]), "usage_report.md")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# Token usage and cost report\n\nrequests: 25\n")
    res = wur.check_report(repo["usage"], report, str(repo["dir"]), repo["output"])
    assert not res["ok"]
    assert any("out of date" in p for p in res["problems"])


def test_check_passes_after_a_fresh_render(repo):
    usage = _usage()
    with open(repo["usage"], "w", encoding="utf-8") as fh:
        json.dump(usage, fh)
    report = os.path.join(os.path.dirname(repo["usage"]), "usage_report.md")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(wur.render(usage, _ctx(repo, usage)))
    res = wur.check_report(repo["usage"], report, str(repo["dir"]), repo["output"])
    assert res["ok"], res["problems"]


def test_check_reports_a_missing_usage_json(repo):
    res = wur.check_report(repo["usage"], "x.md", str(repo["dir"]), repo["output"])
    assert not res["ok"]
    assert any("no usage JSON" in p for p in res["problems"])


def test_check_cli_exit_codes(repo):
    usage = _usage()
    with open(repo["usage"], "w", encoding="utf-8") as fh:
        json.dump(usage, fh)
    report = os.path.join(os.path.dirname(repo["usage"]), "usage_report.md")
    rc = wur.main(["--usage", repo["usage"], "--dataset", str(repo["dir"]),
                   "--predictions", repo["output"], "--out", report])
    assert rc == 0
    # Rendering wrote the file; a second --check must agree.
    assert wur.main(["--usage", repo["usage"], "--dataset", str(repo["dir"]),
                     "--predictions", repo["output"], "--out", report, "--check"]) == 0
    os.remove(report)
    assert wur.main(["--usage", repo["usage"], "--dataset", str(repo["dir"]),
                     "--predictions", repo["output"], "--out", report, "--check"]) == 1


# ---------------------------------------------------------------------------------------
# the report that actually ships
# ---------------------------------------------------------------------------------------

def test_shipped_usage_report_is_current():
    """code/evaluation/usage_report.md must describe the run that produced output.csv."""
    usage_path = os.path.join(ROOT, "code", "evaluation", "reports", "usage_last_run.json")
    report_path = os.path.join(ROOT, "code", "evaluation", "usage_report.md")
    if not os.path.exists(usage_path):
        pytest.skip("no usage JSON in this checkout")
    res = wur.check_report(usage_path, report_path, os.path.join(ROOT, "dataset"),
                           os.path.join(ROOT, "output.csv"))
    assert res["ok"], res["problems"]
