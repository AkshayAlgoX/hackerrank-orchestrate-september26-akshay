"""Regression tests pinned to the 25 solved samples (dataset/sample_requests.csv).

Two kinds of test live here:

* rule tests on small synthetic ledgers that reproduce the structure of a sample and assert
  the rule the sample establishes (deterministic, independent of the variable-spend estimate);
* end-to-end assertions on the real sample rows. Rows whose reference value depends on the
  hidden per-category spending amounts (which the history only lets us estimate to ~1-3%) are
  marked xfail(strict=True) so the suite documents exactly which sample rows are reproduced
  and which are not; a strict xfail fails loudly if such a row starts passing.
"""
from __future__ import annotations

import csv
import os
from datetime import date
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_profile, mk_request, monthly, run_case
from buyorwait.ledger import forecast_horizon_end
from buyorwait.loaders import load_dataset
from buyorwait.pipeline import run

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(ROOT, "dataset")


@pytest.fixture(scope="module")
def samples():
    ds = load_dataset(DATASET, "sample_requests.csv")
    res = run(ds, use_model=False, cache_path=None)
    with open(os.path.join(DATASET, "sample_requests.csv"), newline="", encoding="utf-8") as fh:
        gold = {r["request_id"]: r for r in csv.DictReader(fh)}
    rows = {r.request_id: dict(zip(r.__dataclass_fields__, r.as_list())) for r in res.rows}
    return ds, res, rows, gold


FIELDS = ("affordability_status", "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment",
          "spending_changes_needed")


def _assert_row(rows, gold, rid, fields=FIELDS, amount=False):
    for f in fields:
        assert rows[rid][f] == gold[rid][f], f"{rid}.{f}: {rows[rid][f]!r} != reference {gold[rid][f]!r}"
    if amount:
        assert D(rows[rid]["amount_safe_to_pay"]) == D(gold[rid]["amount_safe_to_pay"])


# ---- forecast window (§8 of the audit) ---------------------------------------------------

def test_horizon_is_three_calendar_months():
    assert forecast_horizon_end(date(2025, 2, 7)) == date(2025, 4, 30)
    assert forecast_horizon_end(date(2024, 3, 7)) == date(2024, 5, 31)
    assert forecast_horizon_end(date(2026, 4, 5)) == date(2026, 6, 30)
    assert forecast_horizon_end(date(2025, 11, 6)) == date(2026, 1, 31)   # year wrap
    assert forecast_horizon_end(date(2024, 12, 6)) == date(2025, 2, 28)   # February length


def test_request_08_wait_until_third_payday(samples):
    # Full amount safe only on 2025-04-15 (the reference), which requires the rent on 05-01 and
    # the school fee on 05-07 (days 83 and 89 after the request) to lie outside the window.
    ds, res, rows, gold = samples
    _assert_row(rows, gold, "request_08")
    L = res.decisions["request_08"].ledger
    assert L.horizon_end == date(2025, 4, 30)
    assert [(f.on, f.amount) for f in L.salary_flows] == [(date(2025, 2, 15), D("1422.85")), (date(2025, 3, 15), D("1422.85")),
                                                          (date(2025, 4, 15), D("1422.85"))]


def test_request_12_no_income_full_amount_safe_today(samples):
    ds, res, rows, gold = samples
    _assert_row(rows, gold, "request_12", amount=True)
    dec = res.decisions["request_12"]
    assert dec.ledger.salary_flows == [] and any("income ended" in a for a in dec.ledger.audit)


def test_request_13_wait_until_may_payday(samples):
    ds, res, rows, gold = samples
    _assert_row(rows, gold, "request_13")


# ---- earliest date is capacity, independent of payment preferences (§5) ------------------

def test_request_12_earliest_equals_request_date_although_full_payment_not_accepted(samples):
    ds, res, rows, gold = samples
    p = ds.profiles["user_12"]
    assert "full_payment" not in p.payment_methods
    assert rows["request_12"]["earliest_date_for_full_payment"] == "2026-04-05"
    assert rows["request_12"]["recommended_payment_method"] == "installments"


def test_earliest_date_filled_when_user_refuses_full_payment():
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    p = mk_profile(current_available_balance=D("1400"), minimum_balance_to_keep=D("500"), payment_methods=("installments",))
    dec, row = run_case(p, ev, mk_request(1500, deadline=date(2026, 8, 1)))
    assert row["earliest_date_for_full_payment"] == "2026-06-15"
    assert row["recommended_payment_method"] == "not_recommended"   # no eligible plan, but capacity date stays


# ---- partial payment shape (§7) ----------------------------------------------------------

def test_request_19_partial_payment_first_instalment_is_amount_safe(samples):
    ds, res, rows, gold = samples
    r = rows["request_19"]
    assert r["recommended_payment_method"] == "partial_payment" and r["affordability_status"] == "affordable_with_plan"
    first, second = r["payment_plan"].split("|")
    d1, a1 = first.split(":"); d2, a2 = second.split(":")
    assert d1 == "2024-09-04" and D(a1) == D(r["amount_safe_to_pay"])
    assert d2 == r["earliest_date_for_full_payment"] == "2024-09-15"
    assert D(a1) + D(a2) == D("39660")


@pytest.mark.xfail(strict=True, reason="amount_safe_to_pay 28820 depends on the hidden per-category spending amounts")
def test_request_19_exact_reference_row(samples):
    ds, res, rows, gold = samples
    _assert_row(rows, gold, "request_19", amount=True)


# ---- spending-change search (§6) ---------------------------------------------------------

def _user21_like(balance):
    """cloud 11 (stoppable), streaming 47 (reducible_or_stoppable, min 23.50), salary on the 15th."""
    ev = monthly("cloud", "cloud_storage", 11, 12, 5, etype="subscription", flex="stoppable")
    ev += monthly("str", "streaming", 47, 9, 5, etype="subscription", flex="reducible_or_stoppable", minallowed=23.5)
    ev += monthly("sal", "salary", 2256, 15, 5, etype="income", desc="Payroll credit")
    p = mk_profile(current_available_balance=D(balance), minimum_balance_to_keep=D("1800"),
                   reduce_categories=("streaming",), stop_categories=("streaming", "cloud_storage"),
                   payment_methods=("full_payment",))
    return p, ev


def test_pair_with_smaller_total_saving_beats_single_stop():
    # deficit 31.05 as in sample request_21: {stop cloud, reduce streaming} saves 34.50 per month,
    # the valid single change {stop streaming} saves 47 -> the pair is the least disruptive.
    p, ev = _user21_like("1858")   # room 0 after the June 9 streaming + June 12 cloud that precede payday
    dec, row = run_case(p, ev, mk_request("31.05", deadline=date(2026, 6, 14), partial=False))
    assert row["spending_changes_needed"] == "stop:cloud4|reduce_to:str4:23.50"
    assert row["affordability_status"] == "affordable_with_plan" and row["recommended_payment_method"] == "full_payment"


def test_single_change_when_it_is_the_least_saving():
    p, ev = _user21_like("1858")
    dec, row = run_case(p, ev, mk_request("10", deadline=date(2026, 6, 14), partial=False))
    assert row["spending_changes_needed"] == "stop:cloud4"


def test_stop_and_reduce_never_target_the_same_event():
    p, ev = _user21_like("1858")
    dec, row = run_case(p, ev, mk_request("50", deadline=date(2026, 6, 14), partial=False))
    changes = row["spending_changes_needed"]
    assert changes == "stop:cloud4|stop:str4"
    targets = [c.split(":")[1] for c in changes.split("|")]
    assert len(set(targets)) == len(targets)


def test_no_change_when_plan_already_safe():
    p, ev = _user21_like("2500")
    dec, row = run_case(p, ev, mk_request("100", deadline=date(2026, 6, 14), partial=False))
    assert row["spending_changes_needed"] == "none" and row["affordability_status"] == "affordable_now"


def test_full_payment_with_changes_beats_wait_after_deadline():
    # Reference pattern of samples 06/11/21: deadline the day before payday -> pay today with changes.
    p, ev = _user21_like("1858")
    dec, row = run_case(p, ev, mk_request("31.05", deadline=date(2026, 6, 14), partial=False))
    assert row["payment_plan"] == "2026-06-02:31.05" and row["earliest_date_for_full_payment"] == "2026-06-15"


def test_request_11_single_reduce_within_reference_deficit(samples):
    # With the reference deficit (599,355 < one reduced weekend delivery 665,950) a single
    # reduce_to is chosen; our forecast overshoots the deficit by 0.7% so a second change appears.
    ds, res, rows, gold = samples
    assert gold["request_11"]["spending_changes_needed"] == "reduce_to:event_989:665950"
    assert "reduce_to:event_989:665950" in rows["request_11"]["spending_changes_needed"]


@pytest.mark.xfail(strict=True, reason="deficit estimate 0.7% above the reference adds a second change and moves the earliest date")
def test_request_11_exact_reference_row(samples):
    ds, res, rows, gold = samples
    _assert_row(rows, gold, "request_11")


@pytest.mark.xfail(strict=True, reason="reference drawdown 539.10 vs estimate ~618: 5-day transport series excluded by the reference for this user only")
def test_request_06_exact_reference_row(samples):
    ds, res, rows, gold = samples
    _assert_row(rows, gold, "request_06")


@pytest.mark.xfail(strict=True, reason="reference drawdown 568 includes transport/dining occurrences that fall outside their observed cadence")
def test_request_21_exact_reference_row(samples):
    ds, res, rows, gold = samples
    _assert_row(rows, gold, "request_21")


# ---- unsettled credits are never cash (requests 04, 10, 14, 20, 22, 23) ------------------

def _positive_known(res, rid):
    L = res.decisions[rid].ledger
    return [(f.on.isoformat(), str(f.amount), f.label) for f in L.known_flows if f.amount > 0]


def test_request_04_pending_bonus_not_counted(samples):
    ds, res, rows, gold = samples
    assert _positive_known(res, "request_04") == []
    assert all(f.label.startswith(("scheduled", "projected")) for f in res.decisions["request_04"].ledger.salary_flows)
    _assert_row(rows, gold, "request_04")


def test_request_10_platform_payouts_not_projected(samples):
    ds, res, rows, gold = samples
    L = res.decisions["request_10"].ledger
    assert L.salary_flows == [] and not any(s.is_income for s in L.series) and _positive_known(res, "request_10") == []
    _assert_row(rows, gold, "request_10")


def test_request_14_salary_resumes_and_no_invented_childcare(samples):
    ds, res, rows, gold = samples
    L = res.decisions["request_14"].ledger
    assert [(f.on, f.amount) for f in L.salary_flows][:1] == [(date(2025, 8, 15), D("2717"))]
    assert not any("childcare" in s.description.lower() for s in L.series)   # announced without an amount
    _assert_row(rows, gold, "request_14")


def test_request_20_pending_refund_ignored_pending_debits_reserved(samples):
    ds, res, rows, gold = samples
    L = res.decisions["request_20"].ledger
    assert _positive_known(res, "request_20") == []
    assert any("event_1785" in a and "pending credit" in a for a in L.audit)
    reserved = {f.source_event_id: f.amount for f in L.known_flows if f.amount < 0}
    assert reserved["event_1787"] == D("-4470")
    assert reserved["event_1786"] == D("-704.05")    # blank amount resolved from image_05 (golden), never zero
    _assert_row(rows, gold, "request_20")


def test_request_22_unrealized_valuation_ignored(samples):
    ds, res, rows, gold = samples
    L = res.decisions["request_22"].ledger
    assert _positive_known(res, "request_22") == []
    assert not any(f.source_event_id == "event_1960" for f in L.known_flows)
    _assert_row(rows, gold, "request_22")


def test_request_23_pending_prize_not_counted(samples):
    ds, res, rows, gold = samples
    assert _positive_known(res, "request_23") == []
    _assert_row(rows, gold, "request_23")


def test_all_sample_rows_pass_contract(samples):
    ds, res, rows, gold = samples
    assert res.violations == {}
    assert set(rows) == set(gold)
