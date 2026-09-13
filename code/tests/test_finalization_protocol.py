"""Finalization protocol (buyorwait/finalize.py): stage -> validate -> publish -> sign.

Crash injection at every transition, every stale/mixed artefact combination, partial and
corrupted manifests. The classifier must answer INCOMPLETE for everything except a fully
published, signed, matching set.
"""
import importlib.util
import json
import os

import pytest

from buyorwait import finalize
from buyorwait.finalize import build_manifest, classify, csv_rows, manifest_path_for, publish, sha256_of

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(ROOT, "dataset")
HAS_DATASET = os.path.exists(os.path.join(DATASET, "requests.csv"))


# ---------------------------------------------------------------------------------------
# a synthetic artefact set, no engine needed
# ---------------------------------------------------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _set(tmp_path, tag, rows=3):
    """Stage a complete artefact set tagged `tag` (different tags -> different bytes)."""
    stage = tmp_path / f"stage-{tag}"
    staged = {"output": str(stage / "output.csv.tmp"), "usage": str(stage / "usage.json.tmp"),
              "report": str(stage / "report.md.tmp"), "proofs": str(stage / "proofs.json.tmp")}
    _write(staged["output"], "request_id,amount_safe_to_pay\n" + "".join(f"r{i},{i}{tag}\n" for i in range(rows)))
    digest = sha256_of(staged["output"])
    usage = {"output_sha256": digest, "output_rows": rows, "status": "complete", "tag": tag}
    _write(staged["usage"], json.dumps(usage))
    _write(staged["report"], f"# report {tag}\nsha {digest}\n")
    _write(staged["proofs"], json.dumps({"r0": {"tag": tag}}))
    return staged, usage


def _final(tmp_path):
    return {"output": str(tmp_path / "output.csv"), "usage": str(tmp_path / "reports" / "usage_last_run.json"),
            "report": str(tmp_path / "usage_report.md"), "proofs": str(tmp_path / "reports" / "proofs.json")}


def _publish(tmp_path, tag, before_step=None, rows=3):
    staged, usage = _set(tmp_path, tag, rows)
    final = _final(tmp_path)
    assert finalize.validate_staged(staged, rows, usage) == []
    manifest = build_manifest({n: (staged[n], final[n]) for n in staged}, rows, "fp-" + tag)
    mp = manifest_path_for(final["usage"])
    return publish(staged, final, mp, manifest, before_step=before_step), mp, final


def _state(mp, final):
    return classify(mp, {k: final[k] for k in ("output", "usage", "report", "proofs")})["state"]


class Die(Exception):
    pass


# ---------------------------------------------------------------------------------------
# happy path and the ordering guarantee
# ---------------------------------------------------------------------------------------

def test_full_publication_is_complete_and_ordered(tmp_path):
    done, mp, final = _publish(tmp_path, "A")
    assert done == ["proofs", "usage", "report", "output", "manifest"]     # output last, manifest after it
    assert _state(mp, final) == "COMPLETE"
    m = json.load(open(mp))
    assert m["version"] == finalize.MANIFEST_VERSION and m["status"] == "complete" and m["output_rows"] == 3
    assert m["files"]["output"]["sha256"] == sha256_of(final["output"])
    assert m["files"]["usage"]["sha256"] == sha256_of(final["usage"])
    assert m["files"]["report"]["sha256"] == sha256_of(final["report"])


def test_no_run_at_all_is_incomplete(tmp_path):
    final = _final(tmp_path)
    v = classify(manifest_path_for(final["usage"]), {"output": final["output"], "usage": final["usage"]})
    assert v["state"] == "INCOMPLETE" and "no completion manifest" in v["problems"][0]


# ---------------------------------------------------------------------------------------
# process death at every transition
# ---------------------------------------------------------------------------------------

STEPS = ["proofs", "usage", "report", "output", "manifest"]


@pytest.mark.parametrize("die_at", STEPS)
def test_death_before_each_step_with_a_previous_complete_run(tmp_path, die_at):
    """Whatever survives must be classified: the OLD set intact -> COMPLETE(old); a mixed set -> INCOMPLETE."""
    _, mp, final = _publish(tmp_path, "A")
    old = {k: open(final[k], "rb").read() for k in final}
    old_manifest = open(mp, "rb").read()

    def hook(step):
        if step == die_at:
            raise Die(step)
    with pytest.raises(Die):
        _publish(tmp_path, "B", before_step=hook)
    v = classify(mp, {k: final[k] for k in final})
    published_before_death = STEPS[:STEPS.index(die_at)]
    if not published_before_death:
        # died before touching anything: the old set is untouched and still signed
        assert v["state"] == "COMPLETE"
        assert all(open(final[k], "rb").read() == old[k] for k in final) and open(mp, "rb").read() == old_manifest
    else:
        # some new files, old manifest: never classified complete
        assert v["state"] == "INCOMPLETE", v["problems"]
        assert any("not the file this run published" in p for p in v["problems"])
        assert open(mp, "rb").read() == old_manifest          # the manifest is only ever written last
        for k in published_before_death:
            assert open(final[k], "rb").read() != old[k]      # the new file is really in place
        for k in [s for s in STEPS if s not in published_before_death and s != "manifest"]:
            assert open(final[k], "rb").read() == old[k]      # untouched files are the old ones


@pytest.mark.parametrize("die_at", STEPS)
def test_death_before_each_step_on_a_first_ever_run(tmp_path, die_at):
    def hook(step):
        if step == die_at:
            raise Die(step)
    with pytest.raises(Die):
        _publish(tmp_path, "A", before_step=hook)
    final = _final(tmp_path)
    v = classify(manifest_path_for(final["usage"]), {k: final[k] for k in final})
    assert v["state"] == "INCOMPLETE"


def test_death_after_signing_leaves_a_complete_consistent_set(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    # a death right after the last write changes nothing on disk: the set is signed and matches
    assert _state(mp, final) == "COMPLETE"
    m = json.load(open(mp))
    for name, entry in m["files"].items():
        assert sha256_of(final[name]) == entry["sha256"]


def test_death_during_the_manifest_write_leaves_the_old_manifest(tmp_path):
    """atomic_write means a partially written manifest is never observed."""
    _, mp, final = _publish(tmp_path, "A")
    old_manifest = open(mp, "rb").read()
    staged, usage = _set(tmp_path, "B")
    manifest = build_manifest({n: (staged[n], final[n]) for n in staged}, 3, "fp-B")

    def broken_replace(src, dst):
        if dst == mp:
            raise Die("during manifest replace")
        os.replace(src, dst)
    from buyorwait import atomic
    import types
    real = atomic.atomic_write

    def flaky_atomic_write(path, write, **kw):
        return real(path, write, replace=broken_replace, **kw)
    finalize.atomic_write = flaky_atomic_write
    try:
        with pytest.raises(Die):
            publish(staged, final, mp, manifest)
    finally:
        finalize.atomic_write = real
    assert open(mp, "rb").read() == old_manifest
    assert _state(mp, final) == "INCOMPLETE"          # new files, old manifest
    assert [n for n in os.listdir(tmp_path / "reports") if n.endswith(".tmp")] == []


# ---------------------------------------------------------------------------------------
# stale / mixed combinations
# ---------------------------------------------------------------------------------------

def _swap_in(tmp_path, tag, names):
    """Publish set `tag` into a side directory and copy only `names` over the live set."""
    side = tmp_path / f"side-{tag}"
    side.mkdir()
    _, smp, sfinal = _publish(side, tag)
    final = _final(tmp_path)
    for n in names:
        os.replace(sfinal[n], final[n]) if n != "manifest" else os.replace(smp, manifest_path_for(final["usage"]))
    return final


def test_stale_old_output_new_metadata(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    _swap_in(tmp_path, "B", ["usage", "report", "proofs"])
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE" and any(p.startswith("usage:") for p in v["problems"])


def test_new_output_stale_metadata(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    _swap_in(tmp_path, "B", ["output"])
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE" and any(p.startswith("output:") for p in v["problems"])


def test_new_output_stale_report(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    _swap_in(tmp_path, "B", ["output", "usage", "proofs"])        # everything but the report
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE"


def test_all_old_artifacts_new_manifest(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    _swap_in(tmp_path, "B", ["manifest"])
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE"
    assert {p.split(":")[0] for p in v["problems"]} >= {"output", "usage", "report", "proofs"}


def test_new_report_only(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    _swap_in(tmp_path, "B", ["report"])
    assert _state(mp, final) == "INCOMPLETE"


def test_row_count_drift_is_caught_even_if_someone_forges_the_digest(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    m = json.load(open(mp))
    m["output_rows"] = 2                                            # digest still matches, rows do not
    json.dump(m, open(mp, "w"))
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE" and any("rows on disk" in p for p in v["problems"])


def test_missing_listed_file_is_incomplete(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    os.remove(final["report"])
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE" and any("is missing" in p for p in v["problems"])


def test_engine_fingerprint_drift_is_incomplete(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    assert classify(mp, final, engine_fingerprint="fp-A")["state"] == "COMPLETE"
    v = classify(mp, final, engine_fingerprint="fp-other")
    assert v["state"] == "INCOMPLETE" and any("engine files changed" in p for p in v["problems"])


# ---------------------------------------------------------------------------------------
# partial / corrupted manifests
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("content", [b"", b"{", b"[]", b"null", b"\xff\xfe", b'"complete"'])
def test_corrupted_manifest_is_incomplete(tmp_path, content):
    _, mp, final = _publish(tmp_path, "A")
    open(mp, "wb").write(content)
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE" and ("unreadable" in v["problems"][0] or "not an object" in v["problems"][0])


@pytest.mark.parametrize("drop", ["version", "status", "engine_fingerprint", "output_rows", "files"])
def test_partial_manifest_is_incomplete(tmp_path, drop):
    _, mp, final = _publish(tmp_path, "A")
    m = json.load(open(mp)); del m[drop]
    json.dump(m, open(mp, "w"))
    v = classify(mp, final)
    assert v["state"] == "INCOMPLETE" and f"missing {drop}" in v["problems"][0]


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(status="publishing"), lambda m: m.update(version=99),
    lambda m: m["files"].pop("output"), lambda m: m["files"].pop("usage"), lambda m: m.update(files=[]),
])
def test_wrong_status_version_or_entries_is_incomplete(tmp_path, mutate):
    _, mp, final = _publish(tmp_path, "A")
    m = json.load(open(mp)); mutate(m)
    json.dump(m, open(mp, "w"))
    assert _state(mp, final) == "INCOMPLETE"


def test_validate_staged_refuses_inconsistent_sets(tmp_path):
    staged, usage = _set(tmp_path, "A")
    assert finalize.validate_staged(staged, 3, usage) == []
    assert finalize.validate_staged(staged, 4, usage)                                  # wrong row count
    assert finalize.validate_staged(staged, 3, dict(usage, output_sha256="0" * 64))    # digest mismatch
    assert finalize.validate_staged(staged, 3, dict(usage, status="partial"))
    os.remove(staged["report"])
    assert any("report" in p for p in finalize.validate_staged(staged, 3, usage))


def test_resign_updates_only_the_report_entry(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    _write(final["report"], "# re-rendered\n")
    assert _state(mp, final) == "INCOMPLETE"
    finalize.resign(mp, "report", final["report"])
    assert _state(mp, final) == "COMPLETE"
    m = json.load(open(mp))
    assert m["files"]["output"]["sha256"] == sha256_of(final["output"]) and "resigned_at" in m


def test_resign_refuses_when_the_manifest_is_invalid(tmp_path):
    _, mp, final = _publish(tmp_path, "A")
    open(mp, "w").write("{")
    with pytest.raises(ValueError):
        finalize.resign(mp, "report", final["report"])


# ---------------------------------------------------------------------------------------
# the real entry point
# ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prod():
    spec = importlib.util.spec_from_file_location("production_main_fin", os.path.join(ROOT, "code", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not HAS_DATASET, reason="dataset not present")
def test_main_stages_validates_publishes_and_signs(prod, tmp_path):
    out, usage = str(tmp_path / "output.csv"), str(tmp_path / "r" / "usage_last_run.json")
    assert prod.main(["--no-model", "--out", out, "--usage", usage, "--proofs", str(tmp_path / "r" / "p.json"), "--dataset", DATASET]) == 0
    mp = manifest_path_for(usage)
    v = classify(mp, {"output": out, "usage": usage})
    assert v["state"] == "COMPLETE", v["problems"]
    m = json.load(open(mp))
    assert set(m["files"]) == {"output", "usage", "report", "proofs"} and m["output_rows"] == 250
    assert m["engine_fingerprint"] == json.load(open(usage))["engine_fingerprint"]["combined"]
    assert os.path.exists(tmp_path / "r" / "usage_report.md")
    assert [n for n in os.listdir(tmp_path / "r") if n.startswith(".run-")] == []


@pytest.mark.skipif(not HAS_DATASET, reason="dataset not present")
@pytest.mark.parametrize("die_at", STEPS)
def test_main_death_at_every_publish_step_is_never_complete(prod, tmp_path, monkeypatch, die_at):
    """A full run is signed; a *different* run (the 25 samples) dies at each publish step.

    Note: a crashed re-run of the *same* input publishes byte-identical files (deterministic
    engine), and the old manifest then still truthfully describes the disk - COMPLETE is the
    correct answer there. The dying run must therefore differ, which the sample file guarantees.
    """
    out, usage, proofs = str(tmp_path / "output.csv"), str(tmp_path / "r" / "usage_last_run.json"), str(tmp_path / "r" / "proofs.json")
    full = ["--no-model", "--out", out, "--usage", usage, "--proofs", proofs, "--dataset", DATASET]
    other = full + ["--requests", "sample_requests.csv"]
    assert prod.main(full) == 0
    final = {"output": out, "usage": usage, "proofs": proofs, "report": str(tmp_path / "r" / "usage_report.md")}
    old = {k: open(v, "rb").read() for k, v in final.items()}
    mp = manifest_path_for(usage)
    old_manifest = open(mp, "rb").read()
    real_publish = finalize.publish

    def dying_publish(staged, fin, manifest_path, manifest, **kw):
        def hook(step):
            if step == die_at:
                raise Die(step)
        return real_publish(staged, fin, manifest_path, manifest, before_step=hook)
    monkeypatch.setattr(prod.finalize, "publish", dying_publish)
    with pytest.raises(Die):
        prod.main(other)
    monkeypatch.undo()
    published = STEPS[:STEPS.index(die_at)]
    v = classify(mp, {"output": out, "usage": usage})
    assert open(mp, "rb").read() == old_manifest                    # manifest is only ever written last
    if not published:
        assert v["state"] == "COMPLETE" and all(open(final[k], "rb").read() == old[k] for k in final)
    else:
        assert v["state"] == "INCOMPLETE", v["problems"]
        for k in published:
            assert open(final[k], "rb").read() != old[k]            # the 25-sample file is really in place
        for k in [s for s in STEPS if s not in published and s != "manifest"]:
            assert open(final[k], "rb").read() == old[k]
    assert [n for n in os.listdir(tmp_path / "r") if n.startswith(".run-")] == []
    # recovery: a clean full run re-establishes a complete, self-consistent set
    assert prod.main(full) == 0 and classify(mp, {"output": out, "usage": usage})["state"] == "COMPLETE"
    assert open(out, "rb").read() == old["output"]


@pytest.mark.skipif(not HAS_DATASET, reason="dataset not present")
def test_main_refuses_to_publish_an_inconsistent_staged_set(prod, tmp_path, monkeypatch):
    out, usage = str(tmp_path / "output.csv"), str(tmp_path / "r" / "usage_last_run.json")
    monkeypatch.setattr(prod.finalize, "validate_staged", lambda staged, n, u: ["injected inconsistency"])
    rc = prod.main(["--no-model", "--out", out, "--usage", usage, "--dataset", DATASET])
    assert rc == 3 and not os.path.exists(out) and not os.path.exists(usage)
    assert not os.path.exists(manifest_path_for(usage))


@pytest.mark.skipif(not HAS_DATASET, reason="dataset not present")
def test_samples_run_never_touches_the_canonical_set(prod, tmp_path):
    """A --samples run defaults to its own metadata file, so it cannot desync the full-run set."""
    out = str(tmp_path / "samples_output.csv")
    canonical = os.path.join(ROOT, "code", "evaluation", "reports", "usage_last_run.json")
    before = open(canonical, "rb").read() if os.path.exists(canonical) else None
    assert prod.main(["--no-model", "--samples", "--out", out, "--usage", str(tmp_path / "s" / "u.json"), "--dataset", DATASET]) == 0
    after = open(canonical, "rb").read() if os.path.exists(canonical) else None
    assert before == after


# ---------------------------------------------------------------------------------------
# the packaging gate refuses to ship a split-brain set
# ---------------------------------------------------------------------------------------

def _mini_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "code" / "evaluation" / "reports").mkdir(parents=True)
    (root / "code" / "main.py").write_text("print('hi')\n")
    (root / "code" / "requirements.txt").write_text("# none\n")
    (root / "README.md").write_text("# r\n")
    _write(str(root / "output.csv"), "request_id,amount_safe_to_pay\nr0,1\n")
    _write(str(root / "code" / "evaluation" / "usage_report.md"), "# report A\n")
    usage = str(root / "code" / "evaluation" / "reports" / "usage_last_run.json")
    _write(usage, json.dumps({"output_sha256": sha256_of(str(root / "output.csv")), "output_rows": 1, "status": "complete"}))
    return root, usage


def test_package_check_passes_a_signed_set_and_refuses_a_broken_one(tmp_path):
    import check_package
    root, usage = _mini_repo(tmp_path)
    files = {"output": (str(root / "output.csv"),) * 2, "usage": (usage, usage),
             "report": (str(root / "code" / "evaluation" / "usage_report.md"),) * 2}
    m = build_manifest(files, 1, "fp")
    _write(manifest_path_for(usage), json.dumps(m))
    assert check_package.check(str(root))["ok"]
    # the report is regenerated by hand (or by a crashed run): the set is no longer signed
    _write(str(root / "code" / "evaluation" / "usage_report.md"), "# report B\n")
    res = check_package.check(str(root))
    assert not res["ok"] and any("not a completed set" in e for e in res["errors"])
    assert not check_package.build(str(root), str(root / "code.zip"))["built"]


def test_package_check_refuses_when_the_manifest_is_missing_after_a_run(tmp_path):
    import check_package
    root, usage = _mini_repo(tmp_path)
    res = check_package.check(str(root))
    assert not res["ok"] and any("no completion manifest" in e for e in res["errors"])


def test_package_check_only_warns_on_a_fresh_checkout(tmp_path):
    import check_package
    root, usage = _mini_repo(tmp_path)
    os.remove(usage)
    res = check_package.check(str(root))
    assert res["ok"] and any("no run recorded" in w for w in res["warnings"])
