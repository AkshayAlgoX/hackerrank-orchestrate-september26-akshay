"""Atomic artefact writes: output.csv and evidence_cache.json can never be left half-written."""
import hashlib
import json
import os
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_profile, mk_request, monthly
from buyorwait import atomic
from buyorwait.atomic import atomic_write, is_temp_artifact
from buyorwait.extraction import gather as G
from buyorwait.extraction import llm
from buyorwait.extraction.gather import _load_cache, _save_cache, gather_evidence
from buyorwait.models import Message
from buyorwait.output import COLUMNS, OutputRow, write_csv
from buyorwait.pipeline import run

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RD = date(2026, 6, 2)


def _rows(n=3):
    return [OutputRow(f"r{i}", "1", "affordable_now", "full_payment", f"2026-06-02:{i}", "2026-06-02", "none", f"row {i}") for i in range(n)]


def _legacy_bytes(rows):
    import csv, io
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow(r.as_list())
    return buf.getvalue().encode("utf-8")


def _tmp_siblings(directory):
    return [n for n in os.listdir(directory) if is_temp_artifact(n)]


# ---------------------------------------------------------------------------------------
# A. output.csv
# ---------------------------------------------------------------------------------------

def test_successful_write_is_byte_identical_to_the_previous_writer(tmp_path):
    out = tmp_path / "output.csv"
    write_csv(str(out), _rows())
    assert out.read_bytes() == _legacy_bytes(_rows())
    assert _tmp_siblings(tmp_path) == []


def test_failure_before_replacement_leaves_the_previous_output_intact(tmp_path, monkeypatch):
    out = tmp_path / "output.csv"
    write_csv(str(out), _rows(2))
    before = out.read_bytes()

    class Poison(OutputRow):
        def as_list(self):
            raise RuntimeError("serialization failed mid-way")
    rows = _rows(1) + [Poison(*_rows(1)[0].as_list())]
    with pytest.raises(RuntimeError):
        write_csv(str(out), rows)
    assert out.read_bytes() == before
    assert _tmp_siblings(tmp_path) == []            # the partial temp file is gone


def test_failure_of_the_rename_itself_keeps_the_old_file_and_cleans_up(tmp_path, monkeypatch):
    out = tmp_path / "output.csv"
    write_csv(str(out), _rows(2))
    before = out.read_bytes()

    def broken_replace(src, dst):
        raise OSError("simulated crash during rename")
    with pytest.raises(OSError):
        atomic_write(str(out), lambda fh: fh.write("partial"), replace=broken_replace)
    assert out.read_bytes() == before and _tmp_siblings(tmp_path) == []


def test_interrupted_process_leaves_only_a_temp_sibling_never_a_truncated_output(tmp_path, monkeypatch):
    """Simulate the process dying after the temp file exists but before the rename."""
    out = tmp_path / "output.csv"
    write_csv(str(out), _rows(2))
    before = out.read_bytes()

    def die(src, dst):
        raise KeyboardInterrupt            # BaseException, like a signal-driven exit
    with pytest.raises(KeyboardInterrupt):
        atomic_write(str(out), lambda fh: fh.write("half"), replace=die)
    assert out.read_bytes() == before
    # and even if cleanup could not run, the temp name is recognisably not an output
    assert is_temp_artifact(".output.csv.abc123.tmp") and not is_temp_artifact("output.csv")


def test_temp_file_lives_in_the_same_directory_and_is_unique(tmp_path):
    seen = []

    def spy(src, dst):
        seen.append(src)
        assert os.path.exists(src)          # the temp file is complete and closed at rename time
        os.replace(src, dst)
    out = tmp_path / "sub" / "output.csv"
    atomic_write(str(out), lambda fh: fh.write("a\n"), replace=spy)
    atomic_write(str(out), lambda fh: fh.write("b\n"), replace=spy)
    assert out.read_text() == "b\n"
    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(os.path.dirname(s) == str(tmp_path / "sub") for s in seen)
    assert all(is_temp_artifact(os.path.basename(s)) for s in seen)


def test_packaging_checker_never_ships_a_temp_artifact(tmp_path):
    import check_package
    root = tmp_path
    (root / "code" / "evaluation").mkdir(parents=True)
    (root / "code" / "main.py").write_text("print()\n")
    (root / "code" / "requirements.txt").write_text("\n")
    (root / "code" / "evaluation" / "usage_report.md").write_text("# r\n")
    (root / "README.md").write_text("# r\n")
    (root / "code" / ".output.csv.x.tmp").write_text("half\n")
    res = check_package.check(str(root))
    assert "code/.output.csv.x.tmp" in res["excluded"]


# ---------------------------------------------------------------------------------------
# B. evidence_cache.json
# ---------------------------------------------------------------------------------------

def test_successful_cache_write_is_valid_json_and_deterministic(tmp_path):
    p = tmp_path / "evidence_cache.json"
    _save_cache(str(p), {"b": [1], "a": {"x": None}})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": {"x": None}, "b": [1]}
    first = p.read_bytes()
    _save_cache(str(p), {"a": {"x": None}, "b": [1]})
    assert p.read_bytes() == first and _tmp_siblings(tmp_path) == []


def test_cache_write_failure_leaves_the_previous_cache_intact(tmp_path):
    p = tmp_path / "evidence_cache.json"
    _save_cache(str(p), {"good": True})
    before = p.read_bytes()

    class Unserialisable:
        pass
    with pytest.raises(TypeError):
        _save_cache(str(p), {"bad": Unserialisable()})
    assert p.read_bytes() == before and json.load(open(p)) == {"good": True}
    assert _tmp_siblings(tmp_path) == []


@pytest.mark.parametrize("content", [b"", b"{", b'{"msg:abc": [', b"\xff\xfe\x00", b"[1, 2]", b"null", b'"str"'])
def test_corrupted_or_truncated_cache_loads_as_empty(tmp_path, content):
    p = tmp_path / "evidence_cache.json"
    p.write_bytes(content)
    notes = []
    assert _load_cache(str(p), notes) == {}
    assert notes and "empty cache" in notes[0]


def test_missing_cache_behaves_exactly_as_before(tmp_path):
    notes = []
    assert _load_cache(str(tmp_path / "absent.json"), notes) == {} and notes == []
    assert _load_cache("", notes) == {} and notes == []


def _ds_with_untemplated_message():
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    return mk_dataset(mk_profile(), ev, mk_request(100, rd=RD),
                      messages=[Message("m1", "u1", "r1", None, "2026-05-30T09:00:00Z", "unknown", "free text with no template")])


def test_run_with_a_corrupted_cache_completes_and_replaces_it(tmp_path):
    p = tmp_path / "evidence_cache.json"
    p.write_bytes(b'{"msg:abc": [{"kind": "inc')          # truncated by an interrupted writer
    res = run(_ds_with_untemplated_message(), use_model=False, cache_path=str(p))
    assert len(res.rows) == 1 and res.errors == {}
    assert res.bundle.cache_notes and "unreadable cache ignored" in res.bundle.cache_notes[0]
    assert json.load(open(p)) == {}                        # rewritten as a valid (empty) cache


class _Dead:
    provider, model = "openai", "dead"

    def __init__(self, usage=None):
        pass

    def extract_message(self, msg):
        raise llm.ProviderError("model endpoint returned HTTP 502", status=502, transient=True, attempts=4)

    def extract_image(self, img, event):
        raise llm.ProviderError("timeout", transient=True, attempts=4)


def test_failed_provider_response_never_poisons_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ModelExtractor", _Dead)
    p = tmp_path / "evidence_cache.json"
    _save_cache(str(p), {"msg:previous": [{"kind": "income_ended", "source_kind": "message", "source_id": "m0", "user_id": "u1"}]})
    b = gather_evidence(_ds_with_untemplated_message(), use_model=True, cache_path=str(p))
    assert b.sources["m1"] == "rules-after-provider-error" and len(b.provider_errors) == 1
    assert json.load(open(p)) == {"msg:previous": [{"kind": "income_ended", "source_kind": "message", "source_id": "m0", "user_id": "u1"}]}


# ---------------------------------------------------------------------------------------
# C. the shipped 250-row output
# ---------------------------------------------------------------------------------------

BASELINE_SHA256 = "4b2f61af4e8306e9c3cff47ba7af75dcbb3f3a05ed9c9bfce0f8cc5284d718c6"   # output.csv at 50f864c


def test_full_dataset_output_is_byte_identical_to_the_50f864c_baseline(tmp_path):
    dataset = os.path.join(ROOT, "dataset")
    if not os.path.exists(os.path.join(dataset, "requests.csv")):
        pytest.skip("dataset not present in this checkout")
    from buyorwait.loaders import load_dataset
    ds = load_dataset(dataset, "requests.csv")
    res = run(ds, use_model=False, cache_path=os.path.join(ROOT, "code", "evidence_cache.json"))
    out = tmp_path / "output.csv"
    write_csv(str(out), res.rows)
    assert len(res.rows) == 250 and res.errors == {}
    assert hashlib.sha256(out.read_bytes()).hexdigest() == BASELINE_SHA256
