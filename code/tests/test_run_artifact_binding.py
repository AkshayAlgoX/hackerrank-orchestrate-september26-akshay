"""Run-artifact consistency: output.csv, proofs and usage_last_run.json are one atomic set.

The production entry point (code/main.py) is driven in-process with every artefact redirected
into tmp_path; interruptions are simulated by monkeypatching, never by signals.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(ROOT, "dataset")
BASELINE_SHA256 = "4b2f61af4e8306e9c3cff47ba7af75dcbb3f3a05ed9c9bfce0f8cc5284d718c6"   # output.csv at 045e7fb

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DATASET, "requests.csv")), reason="dataset not present")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def prod():
    """code/main.py loaded by path (evaluation/main.py would shadow a plain `import main`)."""
    spec = importlib.util.spec_from_file_location("production_main", os.path.join(ROOT, "code", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paths(tmp_path):
    return {"out": str(tmp_path / "output.csv"), "usage": str(tmp_path / "usage_last_run.json"),
            "proofs": str(tmp_path / "proofs.json")}


def _run(prod, p):
    return prod.main(["--no-model", "--out", p["out"], "--usage", p["usage"], "--proofs", p["proofs"], "--dataset", DATASET])


def _no_tmp(tmp_path):
    return [n for n in os.listdir(tmp_path) if n.endswith(".tmp")] == []


def _report(p, out_md):
    import write_usage_report as wur
    return wur.main(["--usage", p["usage"], "--predictions", p["out"], "--dataset", DATASET, "--out", out_md])


# ---------------------------------------------------------------------------------------
# E / H. a successful run binds and matches the baseline
# ---------------------------------------------------------------------------------------

def test_E_successful_run_records_the_actual_output_digest(prod, tmp_path):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    u = json.load(open(p["usage"]))
    assert u["output_sha256"] == _sha(p["out"]) and u["output_rows"] == 250 == u["requests"]
    assert u["status"] == "complete" and u["output_bytes"] == os.path.getsize(p["out"])
    assert u["proofs"] == os.path.abspath(p["proofs"]) and os.path.exists(p["proofs"])
    assert u["engine_fingerprint"]["combined"]
    assert not any(k.lower().endswith(("key", "token", "secret")) for k in u)
    assert _no_tmp(tmp_path)


def test_H_full_run_output_is_byte_identical_to_the_045e7fb_baseline(prod, tmp_path):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    assert _sha(p["out"]) == BASELINE_SHA256
    assert sum(1 for _ in open(p["out"], encoding="utf-8")) == 251


def test_F_rerun_after_an_interrupted_run_yields_a_consistent_set(prod, tmp_path, monkeypatch):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    # interrupt the second run right after output.csv is written: no proofs, no metadata update
    monkeypatch.setattr(prod, "write_proofs", lambda path, result: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _run(prod, p)
    monkeypatch.undo()
    assert _run(prod, p) == 0                                   # a clean rerun
    u = json.load(open(p["usage"]))
    assert u["output_sha256"] == _sha(p["out"]) and _no_tmp(tmp_path)
    assert _report(p, str(tmp_path / "usage_report.md")) == 0


# ---------------------------------------------------------------------------------------
# A / B / G. split-brain sets are refused
# ---------------------------------------------------------------------------------------

def test_A_process_dies_during_staging_publishes_nothing(prod, tmp_path, monkeypatch):
    """With the finalization protocol a death before PUBLISH leaves no artefact at all."""
    p = _paths(tmp_path)
    monkeypatch.setattr(prod, "write_proofs", lambda path, result: (_ for _ in ()).throw(RuntimeError("died")))
    with pytest.raises(RuntimeError):
        _run(prod, p)
    assert not os.path.exists(p["out"]) and not os.path.exists(p["usage"]) and not os.path.exists(p["proofs"])
    assert [n for n in os.listdir(tmp_path) if n.startswith(".run-")] == []      # staging dir removed
    assert _report(p, str(tmp_path / "usage_report.md")) == 2
    assert not os.path.exists(tmp_path / "usage_report.md")


def test_A2_incomplete_metadata_without_a_digest_is_refused(prod, tmp_path):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    u = json.load(open(p["usage"])); del u["output_sha256"]
    json.dump(u, open(p["usage"], "w"))
    import write_usage_report as wur
    assert any("incomplete" in m for m in wur.binding_problems(u, p["out"]))
    report = tmp_path / "usage_report.md"
    before = report.read_bytes()                       # written by the run itself
    assert _report(p, str(report)) == 2
    assert report.read_bytes() == before               # refusal leaves the signed report untouched


def test_B_new_output_with_old_metadata_fails_on_hash_mismatch(prod, tmp_path):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    old_usage = open(p["usage"], "rb").read()
    # a newer output.csv (one row edited) without a metadata update
    lines = open(p["out"], encoding="utf-8").read().split("\n")
    lines[1] = lines[1].replace(",", " ,", 1)                    # touch the first data row only
    open(p["out"], "w", encoding="utf-8").write("\n".join(lines))
    assert _sha(p["out"]) != json.loads(old_usage)["output_sha256"]
    import write_usage_report as wur
    problems = wur.binding_problems(json.loads(old_usage), p["out"])
    assert problems and "differs from the run metadata" in problems[0]
    report = tmp_path / "usage_report.md"
    before = report.read_bytes()
    assert _report(p, str(report)) == 2
    assert report.read_bytes() == before
    res = wur.check_report(p["usage"], str(report), DATASET, p["out"])
    assert not res["ok"] and any("differs from the run metadata" in m for m in res["problems"])
    assert res["manifest_state"] == "INCOMPLETE"


def test_G_stale_metadata_from_a_previous_run_cannot_accompany_a_new_output(prod, tmp_path):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    stale = json.load(open(p["usage"]))
    # a second run writes a different output (the samples) while the stale JSON is kept aside
    p2 = dict(p, out=str(tmp_path / "samples_output.csv"), usage=str(tmp_path / "u2.json"), proofs=str(tmp_path / "p2.json"))
    assert prod.main(["--no-model", "--samples", "--out", p2["out"], "--usage", p2["usage"], "--proofs", p2["proofs"], "--dataset", DATASET]) == 0
    import write_usage_report as wur
    assert wur.binding_problems(stale, p2["out"])                       # refused
    assert wur.binding_problems(json.load(open(p2["usage"])), p2["out"]) == []   # its own metadata binds
    assert "differs" in wur.binding_problems(stale, p2["out"])[0] or "rows" in wur.binding_problems(stale, p2["out"])[0]


def test_row_count_mismatch_is_reported_even_when_the_digest_is_faked(prod, tmp_path):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    u = json.load(open(p["usage"])); u["output_rows"] = 249
    import write_usage_report as wur
    assert any("249" in m for m in wur.binding_problems(u, p["out"]))


# ---------------------------------------------------------------------------------------
# C / D. metadata and proofs writes are atomic
# ---------------------------------------------------------------------------------------

def test_C_usage_metadata_serialisation_failure_keeps_the_previous_file(prod, tmp_path, monkeypatch):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    before = open(p["usage"], "rb").read()
    monkeypatch.setattr(prod, "engine_fingerprint", lambda: {"combined": object()})   # not JSON-serialisable
    with pytest.raises(TypeError):
        _run(prod, p)
    assert open(p["usage"], "rb").read() == before and json.load(open(p["usage"]))["status"] == "complete"
    assert _no_tmp(tmp_path)


def test_D_proofs_serialisation_failure_keeps_the_previous_proofs(prod, tmp_path, monkeypatch):
    from buyorwait import pipeline
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    before = open(p["proofs"], "rb").read()
    real_dump = pipeline.json.dump

    def broken(obj, fh, **kw):
        fh.write('{"half": ')
        raise TypeError("simulated serialization failure")
    monkeypatch.setattr(pipeline.json, "dump", broken)
    with pytest.raises(TypeError):
        pipeline.write_proofs(p["proofs"], pipeline.RunResult([], {}, {"r": {"x": 1}}))
    monkeypatch.setattr(pipeline.json, "dump", real_dump)
    assert open(p["proofs"], "rb").read() == before and json.load(open(p["proofs"]))
    assert _no_tmp(tmp_path)


def test_usage_json_and_proofs_are_written_atomically_via_the_helper(prod, tmp_path, monkeypatch):
    from buyorwait import atomic
    seen = []
    real = atomic.tempfile.mkstemp

    def spy(prefix="", suffix="", dir=None):
        seen.append(prefix)                                          # ".<target name>."
        return real(prefix=prefix, suffix=suffix, dir=dir)
    monkeypatch.setattr(atomic.tempfile, "mkstemp", spy)
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    # staged names carry the .tmp suffix; the manifest is written through the helper too
    assert {".output.csv.tmp.", ".proofs.json.tmp.", ".usage_last_run.json.tmp.", ".usage_report.md.tmp.", ".run_manifest.json."} <= set(seen)


# ---------------------------------------------------------------------------------------
# the report writer itself is atomic and refuses to write on mismatch (CLI)
# ---------------------------------------------------------------------------------------

def test_report_cli_refuses_and_leaves_an_existing_report_untouched(prod, tmp_path):
    p = _paths(tmp_path)
    assert _run(prod, p) == 0
    md = str(tmp_path / "usage_report.md")
    assert _report(p, md) == 0
    good = open(md, "rb").read()
    open(p["out"], "a", encoding="utf-8").write("request_999,0,not_affordable,not_recommended,none,,none,x\n")
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "code", "evaluation", "write_usage_report.py"),
                         "--usage", p["usage"], "--predictions", p["out"], "--dataset", DATASET, "--out", md],
                        capture_output=True, text=True)
    assert rc.returncode == 2 and "REFUSED" in rc.stderr and "differs from the run metadata" in rc.stderr
    assert open(md, "rb").read() == good and _no_tmp(tmp_path)
