"""Fail closed on unresolved cash evidence (Target #10).

Rule: a cash debit that will still leave the account inside the planning window (pending or
scheduled, reservation date max(cash_date, request_date) <= horizon) and has no usable
home-currency amount - blank and not resolved by evidence, or no exchange rate - can neither be
reserved nor assumed zero. The ledger refuses to be built (UnresolvedCashEvidence) and the
pipeline renders its conservative fallback row: 0 safe, not_affordable, not_recommended, no plan,
no earliest date, no changes. Nothing is fabricated.

Everything whose amount cannot change what leaves the account is still ignored with an audit
line: settled history (already inside current_available_balance), credits of any status
(never counted before settlement), non-cash rows, failed/cancelled/unrealized rows, and debits
beyond the window.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly
from buyorwait.evidence import Evidence
from buyorwait.extraction import gather as G
from buyorwait.extraction.gather import EvidenceBundle
from buyorwait.ledger import UnresolvedCashEvidence, build_ledger, forecast_horizon_end
from buyorwait.loaders import load_dataset
from buyorwait.models import PaymentOption
from buyorwait.pipeline import run

RD = date(2026, 6, 2)
H = forecast_horizon_end(RD)
FAIL_CLOSED = ["0", "not_affordable", "not_recommended", "none", "", "none"]
DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")


def blank(eid, status, on, cat="groceries", etype="expense", direction="debit", settle=None, desc=None, currency="EUR"):
    return mk_event(eid, etype, cat, direction, None, on, status=status, settle=settle, desc=desc, currency=currency)


def world(extra, balance="3000", minimum="500", amount="1000", options=(), deadline=None):
    ev = monthly("sal", "salary", 2000, 15, 5, etype="income", desc="Payroll credit") + list(extra)
    p = mk_profile(current_available_balance=D(balance), minimum_balance_to_keep=D(minimum))
    req = mk_request(amount, rd=RD, deadline=deadline or RD + timedelta(days=40), partial=True)
    return mk_dataset(p, ev, req, options), req


def run_clean(ds):
    return run(ds, bundle=EvidenceBundle([], []))


# ---- MUST FAIL CLOSED ---------------------------------------------------------------------

@pytest.mark.parametrize("name,extra", [
    ("pending debit settling before the payment", [blank("u1", "pending", RD - timedelta(days=1), settle=RD + timedelta(days=3))]),
    ("pending debit whose settlement date is already past (reserved on the request date)", [blank("u1", "pending", RD - timedelta(days=10), settle=RD - timedelta(days=2))]),
    ("scheduled debit inside the window", [blank("u1", "scheduled", RD + timedelta(days=20), cat="healthcare")]),
    ("scheduled debit on the last day of the window", [blank("u1", "scheduled", H, cat="healthcare")]),
    ("pending lifecycle one-off (authorization) is still cash", [blank("u1", "pending", RD - timedelta(days=1), settle=RD + timedelta(days=2), desc="Pending card authorization")]),
    ("category never exempts a cash debit", [blank("u1", "pending", RD, settle=RD + timedelta(days=1), cat="misc_fees")]),
    ("pending parent with a cancelled linked child is still pending", [blank("u1", "pending", RD, settle=RD + timedelta(days=1)),
                                                                     mk_event("c1", "expense", "groceries", "debit", 10, RD, status="cancelled", linked="u1")]),
    ("foreign pending debit with no supplied rate has no usable amount", [mk_event("u1", "expense", "utilities", "debit", 100, RD + timedelta(days=3), status="pending", currency="USD")]),
])
def test_unresolved_future_debit_fails_closed(name, extra):
    ds, req = world(extra)
    with pytest.raises(UnresolvedCashEvidence) as info:
        build_ledger(ds, "u1", RD, [])
    assert info.value.event_ids == ["u1"]
    res = run_clean(ds)
    row = res.rows[0].as_list()
    assert row[1:7] == FAIL_CLOSED, name
    assert res.errors[req.request_id].startswith("UnresolvedCashEvidence")
    assert "could not be established" in row[7] and "u1" in row[7]
    assert not res.violations                                   # the fallback row is contract-valid


def test_multiple_unresolved_debits_are_all_named_and_settled_history_is_not():
    ds, req = world([blank("u1", "pending", RD, settle=RD + timedelta(days=1)),
                     blank("u2", "scheduled", RD + timedelta(days=9)),
                     blank("u3", "settled", RD - timedelta(days=9))])
    with pytest.raises(UnresolvedCashEvidence) as info:
        build_ledger(ds, "u1", RD, [])
    assert info.value.event_ids == ["u1", "u2"]
    assert run_clean(ds).rows[0].as_list()[1:7] == FAIL_CLOSED


def test_missing_evidence_can_only_make_a_request_less_affordable():
    """The same request with the debit resolved is an ordinary decision; with it unresolved the
    row is the fallback. Missing evidence never raises amount_safe_to_pay."""
    ds, req = world([blank("u1", "pending", RD, settle=RD + timedelta(days=1))])
    resolved = Evidence("image", "image_x", "u1", "expense_amount_resolved", request_id="r1", related_event_id="u1",
                        amount=D("200"), currency="EUR", sent_at="2026-06-01T00:00:00Z")
    with_ev = run(ds, bundle=EvidenceBundle([resolved], [])).rows[0].as_list()
    without = run_clean(ds).rows[0].as_list()
    assert with_ev[2] == "affordable_now" and D(with_ev[1]) == D("1000")
    assert without[1:7] == FAIL_CLOSED and D(without[1]) <= D(with_ev[1])


# ---- SAFE TO IGNORE (audited, never fatal) ---------------------------------------------------

@pytest.mark.parametrize("name,extra", [
    ("scheduled debit one day after the window", [blank("u1", "scheduled", H + timedelta(days=1), cat="healthcare")]),
    ("settled debit (history: already in the opening balance)", [blank("u1", "settled", RD - timedelta(days=1))]),
    ("scheduled payroll credit (not counted, conservative)", [blank("u1", "scheduled", RD + timedelta(days=13), cat="salary", etype="income", direction="credit", desc="Next confirmed salary")]),
    ("pending refund credit (never counted before settlement)", [blank("u1", "pending", RD + timedelta(days=2), cat="refund", etype="refund", direction="credit")]),
    ("non-cash valuation", [blank("u1", "unrealized", RD - timedelta(days=1), cat="investment", etype="investment_valuation", direction="non_cash")]),
    ("cancelled debit", [blank("u1", "cancelled", RD + timedelta(days=2))]),
    ("failed debit", [blank("u1", "failed", RD - timedelta(days=2))]),
    ("settled lifecycle one-off (card retry)", [blank("u1", "settled", RD - timedelta(days=3), desc="Card retry after failed charge")]),
])
def test_unresolved_rows_that_cannot_change_outflows_are_ignored(name, extra):
    ds, req = world(extra)
    L = build_ledger(ds, "u1", RD, [])
    assert not any(f.source_event_id == "u1" for f in L.known_flows + L.salary_flows)
    res = run_clean(ds)
    row = res.rows[0].as_list()
    assert row[2] == "affordable_now" and D(row[1]) == D("1000"), name
    assert not res.errors


def test_unresolved_debit_beyond_the_window_only_blocks_plans_that_reach_it():
    """A scheduled debit with no amount after the nominal horizon does not fail the request; an
    installment plan whose last leg reaches past that debit cannot be verified and is rejected
    with the reason, while full payment today is decided normally."""
    ds, req = world([blank("u1", "scheduled", H + timedelta(days=40), cat="healthcare")],
                    options=[PaymentOption("payment_option_1", "r1", "installments", D("250"), 4, RD, 45, D("0"), D("1000"))],
                    deadline=RD + timedelta(days=150))
    ds.profiles["u1"] = mk_profile(current_available_balance=D("3000"), minimum_balance_to_keep=D("500"), max_installment_months=12)
    res = run_clean(ds)
    dec = res.decisions[req.request_id]
    assert not res.errors and dec.status == "affordable_now"
    assert any("payment_option_1" in r and "no usable amount" in r for r in dec.rejected)


# ---- the real dataset in a clean environment ------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No cache, no golden, no provider credentials."""
    monkeypatch.setattr(G, "IMAGE_GOLDEN", str(tmp_path / "no-golden.json"))
    for k in ("BUYORWAIT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    return str(tmp_path / "empty-cache.json")


def test_request_64_never_becomes_more_affordable_when_its_evidence_is_missing(clean_env):
    ds = load_dataset(DATASET, "requests.csv")
    ds.requests = [r for r in ds.requests if r.request_id in ("request_64", "request_73")]
    clean = run(ds, use_model=False, cache_path=clean_env)
    assert clean.bundle.evidence == [] or not any(e.kind == "expense_amount_resolved" for e in clean.bundle.evidence)
    rows = {r.request_id: r.as_list() for r in clean.rows}
    # request_64: pending grocery invoice (event_6033) with no readable amount -> fail closed
    assert rows["request_64"][1:7] == FAIL_CLOSED and "event_6033" in rows["request_64"][7]
    # request_73: scheduled hospital bill (event_6859) with no readable amount -> fail closed
    assert rows["request_73"][1:7] == FAIL_CLOSED and "event_6859" in rows["request_73"][7]
    assert set(clean.errors) == {"request_64", "request_73"}
    assert not clean.violations
    # the cached (resolved) run is the ordinary decision and is never less conservative than the fallback
    normal = run(ds, use_model=False, cache_path=None)
    nrows = {r.request_id: r.as_list() for r in normal.rows}
    assert not normal.errors
    assert D(nrows["request_64"][1]) >= D(rows["request_64"][1])


# ---- Target #11: unresolved SETTLED history (amount unknown, cash already in the balance) -------
#
# A settled debit with no usable amount never changes the opening balance (current_available_
# balance already contains it) and never has to be reserved. Its DATE still counts toward
# recurrence, and its missing AMOUNT degrades the per-occurrence estimate: the mean of the known
# rows would be optimistic by (x - mean_known)/n per occurrence whenever the missing sample x was
# above the known mean (the real request_35 invoice: +285 of room from one unread image). The
# degraded estimate is therefore the largest typical known occurrence - never below anything the
# history shows, never invented - and the audit says so. The residual, documented below, is the
# case where the missing sample was the history's sole outlier: no estimate built from the known
# rows can bound it, and failing closed on every unresolved historical row would zero nine of the
# eleven image-backed evaluation requests in a clean environment, so it is accepted with audit.

import dataclasses
from statistics import mean


def _cad(prefix, cat, amounts, last, gap, desc=None):
    n = len(amounts)
    return [mk_event(f"{prefix}{k}", "expense", cat, "debit", amounts[k], last - timedelta(days=gap * (n - 1 - k)),
                     desc=(desc[k] if desc else f"{cat} {k}")) for k in range(n)]


def _blank_out(events, ids):
    return [dataclasses.replace(e, amount=None) if e.event_id in ids else e for e in events]


def _binding_world(extra, balance="3000"):
    """full_payment only, request larger than the room, so amount_safe_to_pay reports the room."""
    ev = monthly("sal", "salary", 2000, 15, 5, etype="income", desc="Payroll credit") + list(extra)
    p = mk_profile(current_available_balance=D(balance), minimum_balance_to_keep=D("500"), payment_methods=("full_payment",))
    req = mk_request("5000", rd=RD, deadline=RD + timedelta(days=40), partial=False)
    return mk_dataset(p, ev, req), req


def _series(ds, cat):
    L = build_ledger(ds, "u1", RD, [])
    return L, next((s for s in L.series if s.category == cat), None)


LAST = RD - timedelta(days=3)


def test_unresolved_history_row_keeps_its_date_for_recurrence_and_never_moves_the_opening_balance():
    # exactly the minimum number of rows: dropping the row (instead of only its amount) would
    # dissolve the series and remove every projected occurrence from the forecast
    for events, cat in ((_cad("d", "dining", [100, 100, 100, 100], LAST, 10), "dining"),
                        (_cad("u", "utilities", [100, 100, 100], LAST, 30, desc=["Elec"] * 3), "utilities"),
                        (_cad("u", "utilities", [100, 100, 100], LAST, 30, desc=["Elec", "Power", "Utility"]), "utilities")):
        full, _ = _binding_world(events)
        missing, _ = _binding_world(_blank_out(events, {events[-1].event_id}))
        Lf, sf = _series(full, cat)
        Lm, sm = _series(missing, cat)
        assert sm is not None and sm.occurrences == sf.occurrences and sm.period_days == sf.period_days
        assert Lm.opening_balance == Lf.opening_balance
        deleted, _ = _binding_world([e for e in events if e.event_id != events[-1].event_id])
        assert _series(deleted, cat)[1] is None                  # HEAD semantics would have lost the series


@pytest.mark.parametrize("amounts,missing_idx,expected", [
    ([100, 100, 100, 100, 100, 300], 0, D("300")),               # a low row missing: largest known typical
    ([100, 120, 110, 130, 100, 300], 5, D("130")),               # the high row missing: largest of the rest
    ([100, 100, 100, 100, 100, 100], 2, D("100")),               # constant history: unchanged
    ([100, 100, 100, 100, 100, 900], 0, D("100")),               # 900 > 3 x median is unusual and stays out
])
def test_degraded_estimate_is_the_largest_typical_known_occurrence(amounts, missing_idx, expected):
    events = _cad("d", "dining", amounts, LAST, 10)
    ds, _ = _binding_world(_blank_out(events, {events[missing_idx].event_id}))
    L, s = _series(ds, "dining")
    assert s.amount == expected
    known = [D(str(a)) for i, a in enumerate(amounts) if i != missing_idx]
    assert s.amount >= D(str(round(mean(known), 2))) or expected == D("100")   # never below the known mean
    assert any("historical amount(s) unresolved" in a and events[missing_idx].event_id in a for a in L.audit)


def test_missing_history_amount_cannot_raise_the_room_when_the_history_shows_a_larger_occurrence():
    """safe_missing <= safe_full whenever the missing sample is not the sole outlier."""
    events = _cad("d", "dining", [100, 120, 110, 130, 100, 125], LAST, 10)
    full, _ = _binding_world(events)
    for eid in ("d0", "d1", "d3", "d5"):
        missing, _ = _binding_world(_blank_out(events, {eid}))
        sf = run(full, bundle=EvidenceBundle([], [])).decisions["r1"].amount_safe
        sm = run(missing, bundle=EvidenceBundle([], [])).decisions["r1"].amount_safe
        assert sm <= sf, (eid, sm, sf)


def test_documented_residual_sole_outlier_missing():
    """When the missing sample was the only high occurrence the known history cannot bound it:
    the degraded room exceeds the full-history room by exactly (mean_full - max_known) per
    projected occurrence before the trough. Accepted with audit - the alternative is failing
    closed on every unresolved historical row."""
    events = _cad("d", "dining", [100, 100, 100, 100, 100, 300], LAST, 10)      # mean 133.33, max known 100
    full, _ = _binding_world(events)
    missing, _ = _binding_world(_blank_out(events, {"d5"}))
    sf = run(full, bundle=EvidenceBundle([], [])).decisions["r1"].amount_safe
    sm = run(missing, bundle=EvidenceBundle([], [])).decisions["r1"].amount_safe
    assert sm - sf == D("33.33")                                   # one occurrence (06-09) before the 06-15 payday
    assert _series(missing, "dining")[1].amount == D("100")


def test_all_history_amounts_missing_fails_closed_but_partial_history_does_not():
    events = _cad("d", "dining", [100, 100, 100, 100], LAST, 10)
    with pytest.raises(UnresolvedCashEvidence):
        build_ledger(_binding_world(_blank_out(events, {"d0", "d1", "d2", "d3"}))[0], "u1", RD, [])
    L, s = _series(_binding_world(_blank_out(events, {"d0", "d1", "d2"}))[0], "dining")
    assert s is not None and s.amount == D("100")


def test_unusual_amount_audit_counts_only_known_rows_when_one_is_missing():
    events = _cad("d", "dining", [100, 100, 100, 100, 100, 100], LAST, 10)
    L, s = _series(_binding_world(_blank_out(events, {"d1"}))[0], "dining")
    assert not any("unusual amount" in a for a in L.audit)


def test_minimum_boundary_around_the_degraded_estimate():
    events = _cad("d", "dining", [100, 120, 110, 130, 100, 125], LAST, 10)       # degraded estimate 130
    ds, req = _binding_world(_blank_out(events, {"d5"}))
    L = build_ledger(ds, "u1", RD, [])
    from buyorwait.forecast import project_flows, simulate
    trough = simulate(L.opening_balance, project_flows(L)).minimum
    room = trough - L.minimum_balance
    for delta, expect in ((D("0"), room), (D("0.01"), room - D("0.01")), (D("-0.01"), room + D("0.01"))):
        ds.profiles["u1"] = dataclasses.replace(ds.profiles["u1"], minimum_balance_to_keep=D("500") + delta)
        assert run(ds, bundle=EvidenceBundle([], [])).decisions["r1"].amount_safe == expect


def test_request_35_clean_environment_is_not_more_affordable_than_the_resolved_run(clean_env):
    ds = load_dataset(DATASET, "requests.csv")
    ds.requests = [r for r in ds.requests if r.request_id in ("request_35", "request_33", "request_78", "request_113")]
    clean = run(ds, use_model=False, cache_path=clean_env)
    normal = run(ds, use_model=False, cache_path=None)
    assert not clean.errors and not normal.errors
    for rid in ("request_35", "request_33", "request_78", "request_113"):
        c, n = clean.decisions[rid], normal.decisions[rid]
        assert c.ledger.opening_balance == n.ledger.opening_balance
        assert c.amount_safe <= n.amount_safe, (rid, c.amount_safe, n.amount_safe)
        assert any("historical amount(s) unresolved" in a for a in c.ledger.audit)
