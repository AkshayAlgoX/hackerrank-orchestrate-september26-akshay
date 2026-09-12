"""desired_completion_date is a hard eligibility gate (planning.completes_by_deadline).

Statement: "A recommendation is safe only if the user can make every listed payment, complete
the full request by its deadline, ..." and "The plan must complete the request by
desired_completion_date". A late plan is never a candidate, so it can never win by default.
"""
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_profile, mk_request, monthly
from buyorwait.extraction.gather import EvidenceBundle
from buyorwait.models import PaymentOption
from buyorwait.pipeline import decide_request
from buyorwait.planning import completes_by_deadline

RD = date(2026, 6, 2)


def _events():
    return monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")


def _opt(oid, n, first, freq=30, amount="200"):
    return PaymentOption(f"payment_option_{oid}", "r1", "installments", D(amount), n, first, freq, D("0"), D(amount) * n)


def _decide(req, options=(), **profile):
    base = dict(current_available_balance=D("5000"), max_installment_months=12, payment_methods=("installments",))
    base.update(profile)
    ds = mk_dataset(mk_profile(**base), _events(), req, options=list(options))
    return decide_request(ds, req, EvidenceBundle([], []))


# ---------------------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("offset,ok", [(-1, True), (0, True), (1, False)])
def test_final_payment_relative_to_the_deadline(offset, ok):
    deadline = date(2026, 7, 10)
    pays = [(RD, D("100")), (deadline + timedelta(days=offset), D("100"))]
    assert completes_by_deadline(pays, deadline)[0] is ok


def test_empty_or_non_positive_schedules_never_pass():
    assert completes_by_deadline([], date(2026, 7, 10)) == (False, "empty payment schedule")
    assert not completes_by_deadline([(RD, D("0"))], date(2026, 7, 10))[0]
    assert not completes_by_deadline([(RD, D("-5"))], date(2026, 7, 10))[0]


def test_unsorted_schedule_is_judged_by_its_latest_payment():
    deadline = date(2026, 7, 10)
    pays = [(date(2026, 7, 11), D("100")), (RD, D("100"))]  # late leg is not the last element
    assert not completes_by_deadline(pays, deadline)[0]


# ---------------------------------------------------------------------------------------
# decide(): installments
# ---------------------------------------------------------------------------------------

def test_installments_ending_one_day_before_the_deadline_are_eligible():
    opt = _opt(1, 3, RD)                               # 06-02, 07-02, 08-01
    dec = _decide(mk_request(600, partial=False, deadline=date(2026, 8, 2)), [opt])
    assert dec.method == "installments" and dec.plan.option is opt


def test_installments_ending_exactly_on_the_deadline_are_eligible():
    opt = _opt(1, 3, RD)
    dec = _decide(mk_request(600, partial=False, deadline=date(2026, 8, 1)), [opt])
    assert dec.method == "installments" and dec.plan.option is opt


def test_installments_ending_one_day_after_the_deadline_are_ineligible():
    opt = _opt(1, 3, RD)
    dec = _decide(mk_request(600, partial=False, deadline=date(2026, 7, 31)), [opt])
    assert dec.method == "not_recommended" and dec.status == "not_affordable" and dec.plan is None
    assert any("after desired_completion_date" in r for r in dec.rejected)


def test_only_candidate_being_late_yields_no_plan_not_a_default_win():
    """The pre-fix bug: a sole late candidate sorted first and was recommended."""
    late = _opt(1, 4, RD)                              # ends 09-30
    dec = _decide(mk_request(800, partial=False, deadline=date(2026, 7, 15)), [late])
    assert dec.candidates == [] and dec.plan is None
    assert (dec.status, dec.method) == ("not_affordable", "not_recommended")


def test_late_candidate_is_rejected_and_the_valid_one_selected():
    late = _opt(1, 3, RD, amount="200")                # 06-02, 07-02, 08-01: lowest option id, ends late
    ok = _opt(2, 2, RD, amount="300")                  # 06-02, 07-02
    dec = _decide(mk_request(600, partial=False, deadline=date(2026, 7, 20)), [late, ok])
    assert dec.method == "installments" and dec.plan.option is ok
    assert all(c.option is not late for c in dec.candidates)
    assert any(r.startswith("payment_option_1:") and "after desired_completion_date" in r for r in dec.rejected)


def test_late_cheaper_option_never_outranks_a_valid_dearer_one():
    late_cheap = _opt(1, 3, date(2026, 7, 1), amount="200")   # ends 08-30, total 600
    valid_dear = _opt(2, 3, RD, amount="210")                  # ends 08-01, total 630
    dec = _decide(mk_request(600, partial=False, deadline=date(2026, 8, 1)), [late_cheap, valid_dear])
    assert dec.plan.option is valid_dear


# ---------------------------------------------------------------------------------------
# decide(): wait / partial / full
# ---------------------------------------------------------------------------------------

def test_wait_after_the_deadline_is_not_affordable_but_keeps_the_capacity_date():
    # balance 1400 - rent 800 leaves 600 - min 500 -> 100 safe today; full 1500 is safe on the 06-15 payday
    req = mk_request(1500, partial=False, deadline=date(2026, 6, 14))
    dec = _decide(req, payment_methods=("full_payment",), current_available_balance=D("1400"), max_installment_months=None)
    assert dec.earliest_full == date(2026, 6, 15)
    assert (dec.status, dec.method, dec.plan) == ("not_affordable", "not_recommended", None)
    assert any(r.startswith("wait:") for r in dec.rejected)
    on_time = _decide(mk_request(1500, partial=False, deadline=date(2026, 6, 15)),
                      payment_methods=("full_payment",), current_available_balance=D("1400"), max_installment_months=None)
    assert (on_time.status, on_time.method) == ("affordable_later", "wait")


def test_partial_second_leg_respects_the_deadline():
    req_ok = mk_request(1500, partial=True, deadline=date(2026, 6, 15))
    dec = _decide(req_ok, payment_methods=("full_payment", "partial_payment"),
                  current_available_balance=D("1400"), max_installment_months=None)
    assert dec.method == "partial_payment" and dec.plan.last_date == date(2026, 6, 15)
    req_late = mk_request(1500, partial=True, deadline=date(2026, 6, 14))
    dec = _decide(req_late, payment_methods=("full_payment", "partial_payment"),
                  current_available_balance=D("1400"), max_installment_months=None)
    assert dec.method == "not_recommended"


def test_full_payment_today_is_always_by_deadline():
    dec = _decide(mk_request(100, partial=False, deadline=RD), payment_methods=("full_payment",),
                  current_available_balance=D("1400"), max_installment_months=None)
    assert (dec.status, dec.method) == ("affordable_now", "full_payment")
