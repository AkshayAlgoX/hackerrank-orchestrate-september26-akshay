"""Tests for the rigorous output.csv validator in code/evaluation/validate_output.py.

The clean fixture is produced by the real deterministic pipeline over the 25 solved samples,
so every mutation below differs from a valid submission in exactly one respect.
"""
import csv
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "evaluation"))

import validate_output as v  # noqa: E402
from buyorwait.loaders import load_dataset  # noqa: E402
from buyorwait.output import COLUMNS, write_csv  # noqa: E402
from buyorwait.pipeline import run  # noqa: E402

DATASET = os.path.join(ROOT, "dataset")
REQUESTS = "sample_requests.csv"


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """A valid sample output plus the dataset it was produced from."""
    ds = load_dataset(DATASET, REQUESTS)
    res = run(ds, use_model=False, cache_path=None)
    path = tmp_path_factory.mktemp("clean") / "output.csv"
    write_csv(str(path), res.rows)
    return ds, str(path)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rewrite(path, rows, newline="\r\n"):
    """Rewrite an output CSV with the given rows (mutating one field at a time)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator=newline)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in COLUMNS])


def _mutate(clean, tmp_path, field, value, which=0):
    """Return a path to a copy of the clean output with one cell replaced."""
    _, src = clean
    rows = _rows(src)
    rows[which][field] = value
    out = tmp_path / f"mutated_{field}.csv"
    _rewrite(str(out), rows)
    return str(out), rows[which]


def _problems(path, recompute=False, requests=REQUESTS):
    return v.validate(path, DATASET, requests, recompute=recompute)


# ---------------------------------------------------------------------------------------
# the clean case
# ---------------------------------------------------------------------------------------

def test_clean_output_passes(clean):
    _, path = clean
    res = v.validate(path, DATASET, REQUESTS)
    assert res["ok"], res["problems"]
    assert res["problems"] == []
    assert res["rows"] == res["expected_rows"] == 25


def test_clean_output_has_no_hard_serialization_defects(clean):
    _, path = clean
    res = v.validate(path, DATASET, REQUESTS, recompute=False)
    assert [p for p in res["problems"]] == []


def test_line_endings_are_reported(clean):
    _, path = clean
    assert v.line_endings(path) in {"crlf", "lf"}


def test_crlf_is_a_warning_not_an_error(clean, tmp_path):
    """A CRLF file parses identically, so it is reported without failing the submission.

    The CRLF copy is written by hand here, not by the engine's writer: ``write_csv`` emits LF
    to match the organizer files, so an externally produced CRLF file is the realistic input
    for this tolerance check.
    """
    _, src = clean
    out = tmp_path / "crlf.csv"
    _rewrite(str(out), _rows(src), newline="\r\n")
    raw = open(str(out), "rb").read()
    assert b"\r\n" in raw
    res = v.validate(str(out), DATASET, REQUESTS, recompute=False)
    assert res["ok"]
    assert any("CRLF" in w for w in res["warnings"])


def test_writer_emits_lf_like_every_organizer_csv(clean):
    """The engine's own output must use LF, matching dataset/*.csv and the golden file."""
    _, path = clean
    raw = open(path, "rb").read()
    assert b"\r" not in raw, "writer emitted a CR (CRLF) line ending"
    assert v.line_endings(path) == "lf"
    res = v.validate(path, DATASET, REQUESTS, recompute=False)
    assert not [w for w in res["warnings"] if "CRLF" in w]


# ---------------------------------------------------------------------------------------
# shape: ids and header
# ---------------------------------------------------------------------------------------

def test_duplicate_request_id_is_an_error(clean, tmp_path):
    _, src = clean
    rows = _rows(src)
    rows.append(dict(rows[0]))
    out = tmp_path / "dup.csv"
    _rewrite(str(out), rows)
    res = _problems(str(out))
    assert not res["ok"]
    assert any("duplicate row" in p for p in res["problems"])


def test_missing_request_id_is_an_error(clean, tmp_path):
    _, src = clean
    rows = _rows(src)[:-1]
    out = tmp_path / "missing.csv"
    _rewrite(str(out), rows)
    res = _problems(str(out))
    assert not res["ok"]
    assert any("missing" in p and "request ids" in p for p in res["problems"])


def test_unknown_request_id_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "request_id", "request_does_not_exist")
    res = _problems(out)
    assert not res["ok"]
    assert any("not in" in p for p in res["problems"])


def test_wrong_column_order_is_an_error(clean, tmp_path):
    _, src = clean
    rows = _rows(src)
    cols = list(COLUMNS)
    cols[1], cols[2] = cols[2], cols[1]
    out = tmp_path / "reordered.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    res = _problems(str(out))
    assert not res["ok"]
    assert any("header" in p for p in res["problems"])


def test_missing_file_is_reported_not_raised(tmp_path):
    res = v.validate(str(tmp_path / "nope.csv"), DATASET, REQUESTS, recompute=False)
    assert not res["ok"]
    assert any("does not exist" in p for p in res["problems"])


def test_empty_file_is_reported(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    res = v.validate(str(p), DATASET, REQUESTS, recompute=False)
    assert not res["ok"]


def test_short_row_is_reported(clean, tmp_path):
    _, src = clean
    rows = _rows(src)
    out = tmp_path / "short.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerow([rows[0][c] for c in COLUMNS][:5])
    res = _problems(str(out))
    assert not res["ok"]
    assert any("columns" in p for p in res["problems"])


# ---------------------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------------------

def test_bom_is_an_error(clean, tmp_path):
    _, src = clean
    out = tmp_path / "bom.csv"
    out.write_bytes(b"\xef\xbb\xbf" + open(src, "rb").read())
    res = _problems(str(out))
    assert not res["ok"]
    assert any("BOM" in p for p in res["problems"])


@pytest.mark.parametrize("bad", ["NaN", "nan", "inf", "-Infinity", "Infinity"])
def test_non_finite_amount_is_an_error(clean, tmp_path, bad):
    out, _ = _mutate(clean, tmp_path, "amount_safe_to_pay", bad)
    res = _problems(out)
    assert not res["ok"]


def test_scientific_notation_is_rejected(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "amount_safe_to_pay", "1e5")
    res = _problems(out)
    assert not res["ok"]
    assert any("plain non-negative amount" in p for p in res["problems"])


def test_thousands_separator_is_rejected(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "amount_safe_to_pay", "1,000")
    res = _problems(out)
    assert not res["ok"]


def test_negative_amount_is_rejected(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "amount_safe_to_pay", "-5")
    res = _problems(out)
    assert not res["ok"]


def test_three_decimal_amount_is_rejected(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "amount_safe_to_pay", "10.123")
    res = _problems(out)
    assert not res["ok"]


def test_stray_whitespace_is_reported(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "affordability_status", " affordable_now ")
    res = _problems(out)
    assert not res["ok"]
    assert any("whitespace" in p for p in res["problems"])


def test_literal_none_in_a_non_none_column_is_reported(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "affordability_status", "none")
    res = _problems(out)
    assert not res["ok"]
    assert any("literal 'none'" in p or "bad status" in p for p in res["problems"])


# ---------------------------------------------------------------------------------------
# enums, plans, changes, bounds
# ---------------------------------------------------------------------------------------

def test_bad_status_enum_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "affordability_status", "affordable_maybe")
    res = _problems(out)
    assert not res["ok"]
    assert any("bad status" in p for p in res["problems"])


def test_bad_method_enum_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "recommended_payment_method", "credit_card")
    res = _problems(out)
    assert not res["ok"]
    assert any("bad method" in p for p in res["problems"])


def test_bad_plan_syntax_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "payment_plan", "2026-09-07:300|not-a-date:5")
    res = _problems(out)
    assert not res["ok"]
    assert any("bad plan item" in p for p in res["problems"])


def test_non_chronological_plan_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "payment_plan", "2026-09-07:300|2026-08-07:300")
    res = _problems(out)
    assert not res["ok"]
    assert any("not chronological" in p for p in res["problems"])


@pytest.mark.parametrize("bad", ["2026-02-30:100", "2026-13-01:100", "2026-00-10:100"])
def test_impossible_calendar_date_does_not_crash_the_validator(clean, tmp_path, bad):
    """A forged impossible date must be reported, never raised out of the validator."""
    out, _ = _mutate(clean, tmp_path, "payment_plan", bad)
    res = _problems(out)
    assert not res["ok"]
    assert any("calendar date" in p or "bad plan item" in p for p in res["problems"])


def test_amount_above_requested_is_out_of_bounds(clean, tmp_path):
    out, row = _mutate(clean, tmp_path, "amount_safe_to_pay", "999999999")
    res = _problems(out)
    assert not res["ok"]
    assert any("out of [0, requested_amount]" in p for p in res["problems"])


def test_bad_spending_change_syntax_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "spending_changes_needed", "cancel:event_1")
    res = _problems(out)
    assert not res["ok"]
    assert any("bad change" in p for p in res["problems"])


def test_stop_and_reduce_on_the_same_event_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "spending_changes_needed", "stop:event_1|reduce_to:event_1:5")
    res = _problems(out)
    assert not res["ok"]
    assert any("same event" in p for p in res["problems"])


def test_more_than_three_spending_changes_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "spending_changes_needed", "stop:a|stop:b|stop:c|stop:d")
    res = _problems(out)
    assert not res["ok"]
    assert any("three spending changes" in p for p in res["problems"])


def test_unknown_spending_change_event_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "spending_changes_needed", "stop:event_does_not_exist")
    res = _problems(out)
    assert not res["ok"]
    assert any("unknown event" in p for p in res["problems"])


def test_earliest_date_before_request_date_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "earliest_date_for_full_payment", "1999-01-01")
    res = _problems(out)
    assert not res["ok"]
    assert any("before request_date" in p for p in res["problems"])


def test_partial_payment_that_misses_the_second_payment_is_an_error(clean, tmp_path):
    _, src = clean
    rows = _rows(src)
    i = next(n for n, r in enumerate(rows) if r["recommended_payment_method"] == "partial_payment")
    rows[i]["payment_plan"] = rows[i]["payment_plan"].split("|")[0]
    out = tmp_path / "short_plan.csv"
    _rewrite(str(out), rows)
    res = _problems(str(out))
    assert not res["ok"]
    assert any("exactly two payments" in p or "sum to" in p for p in res["problems"])


def test_empty_explanation_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "decision_explanation", "")
    res = _problems(out)
    assert not res["ok"]
    assert any("empty explanation" in p for p in res["problems"])


def test_explanation_without_the_home_currency_is_an_error(clean, tmp_path):
    out, _ = _mutate(clean, tmp_path, "decision_explanation", "Looks fine to me.")
    res = _problems(out)
    assert not res["ok"]
    assert any("home currency" in p for p in res["problems"])


# ---------------------------------------------------------------------------------------
# the deepest layer: the 90-day replay catches a forged safe plan
# ---------------------------------------------------------------------------------------

def test_forged_affordable_now_is_caught_by_the_plan_replay(clean, tmp_path):
    """A row that satisfies every syntactic rule but is financially unsafe must still fail."""
    ds, src = clean
    rows = _rows(src)
    req_by_id = {r.request_id: r for r in ds.requests}
    # Pick a request the engine judged not safely payable in full today.
    i = next(n for n, r in enumerate(rows) if r["affordability_status"] in
             ("not_affordable", "affordable_later"))
    req = req_by_id[rows[i]["request_id"]]
    cur = ds.profiles[req.user_id].home_currency
    rows[i].update({
        "amount_safe_to_pay": f"{req.requested_amount:.2f}".rstrip("0").rstrip("."),
        "affordability_status": "affordable_now",
        "recommended_payment_method": "full_payment",
        "payment_plan": f"{req.request_date.isoformat()}:{req.requested_amount:.2f}".rstrip("0").rstrip("."),
        "earliest_date_for_full_payment": req.request_date.isoformat(),
        "spending_changes_needed": "none",
        "decision_explanation": f"Pay {cur} {req.requested_amount:,.2f} today from the available balance.",
    })
    out = tmp_path / "forged.csv"
    _rewrite(str(out), rows)

    # Syntactically clean: without the replay the forgery is invisible.
    assert v.validate(str(out), DATASET, REQUESTS, recompute=False)["ok"]

    res = v.validate(str(out), DATASET, REQUESTS, recompute=True)
    assert not res["ok"], "the 90-day replay should reject a forged affordable_now row"
    assert any("90-day forecast" in p for p in res["problems"])


def test_recompute_can_be_disabled(clean):
    _, path = clean
    res = v.validate(path, DATASET, REQUESTS, recompute=False)
    assert res["ok"]
