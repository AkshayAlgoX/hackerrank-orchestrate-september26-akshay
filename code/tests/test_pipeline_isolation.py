"""pipeline.run must yield exactly one contract-valid row per request even when one request raises."""
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_event, mk_profile, monthly
from buyorwait import pipeline
from buyorwait.extraction.gather import EvidenceBundle
from buyorwait.models import Dataset, FxTable, Request
from buyorwait.output import COLUMNS, validate_row

RD = date(2026, 6, 2)


def _request(i, user="u1", amount=300):
    return Request(f"r{i}", user, RD, "purchase", D(str(amount)), RD + timedelta(days=40), True, "?")


def _dataset(requests):
    profile = mk_profile()
    events = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    return Dataset(profiles={profile.user_id: profile}, events=events,
                   events_by_user={profile.user_id: sorted(events, key=lambda e: (e.event_date, e.event_id))},
                   events_by_id={e.event_id: e for e in events}, requests=list(requests),
                   options_by_request={r.request_id: [] for r in requests}, messages=[], images=[], fx=FxTable())


def _rowdict(row):
    return dict(zip(COLUMNS, row.as_list()))


def test_a_request_for_an_unknown_user_does_not_abort_the_batch():
    """A natural poison: the profile lookup raises KeyError for a request whose user is absent."""
    reqs = [_request(1), _request(2, user="ghost"), _request(3)]
    res = pipeline.run(_dataset(reqs), bundle=EvidenceBundle([], []))
    assert [r.request_id for r in res.rows] == ["r1", "r2", "r3"]
    assert set(res.errors) == {"r2"} and res.errors["r2"].startswith("KeyError")
    assert res.violations == {}
    bad = _rowdict(res.rows[1])
    assert validate_row(bad, reqs[1]) == []
    assert (bad["amount_safe_to_pay"], bad["affordability_status"], bad["recommended_payment_method"],
            bad["payment_plan"], bad["earliest_date_for_full_payment"], bad["spending_changes_needed"]) == \
        ("0", "not_affordable", "not_recommended", "none", "", "none")
    assert "could not be evaluated" in bad["decision_explanation"]
    # the healthy neighbours are evaluated normally and identically to a clean run
    clean = pipeline.run(_dataset([_request(1), _request(3)]), bundle=EvidenceBundle([], []))
    assert [r.as_list() for r in clean.rows] == [res.rows[0].as_list(), res.rows[2].as_list()]
    assert set(res.decisions) == {"r1", "r3"}


def test_an_arbitrary_exception_inside_the_engine_is_isolated(monkeypatch):
    real = pipeline.decide

    def poisoned(req, L, options):
        if req.request_id == "r2":
            raise RuntimeError("synthetic engine failure")
        return real(req, L, options)

    monkeypatch.setattr(pipeline, "decide", poisoned)
    reqs = [_request(1), _request(2), _request(3)]
    res = pipeline.run(_dataset(reqs), bundle=EvidenceBundle([], []))
    assert len(res.rows) == 3 and [r.request_id for r in res.rows] == ["r1", "r2", "r3"]
    assert res.errors == {"r2": "RuntimeError: synthetic engine failure"}
    assert res.rows[0].affordability_status == "affordable_now" == res.rows[2].affordability_status
    assert res.rows[1].recommended_payment_method == "not_recommended"
    proof = res.proofs["r2"]
    assert proof["fallback"] is True and proof["error"]["type"] == "RuntimeError"
    assert "synthetic engine failure" in proof["error"]["traceback"]
    assert res.proofs["r1"]["status"] == "affordable_now"  # normal proofs untouched


def test_fallback_row_is_valid_for_any_request():
    for amount in ("0", "0.01", "12345.67"):
        req = _request(9, amount=amount)
        row = pipeline.fallback_row(req, ValueError("x"))
        assert validate_row(_rowdict(row), req) == []
        assert row.decision_explanation == pipeline.FALLBACK_EXPLANATION.format(error="ValueError")


def test_clean_run_records_no_errors_and_no_fallbacks():
    res = pipeline.run(_dataset([_request(1), _request(2)]), bundle=EvidenceBundle([], []))
    assert res.errors == {} and not any(p.get("fallback") for p in res.proofs.values())
    assert set(res.decisions) == {"r1", "r2"}
