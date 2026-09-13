"""max_installment_months bounds an installment plan's ELAPSED calendar duration (H4).

The final payment (first_payment_date + payment_frequency_days * (number_of_payments - 1)) must
fall within max_installment_months calendar months of the first payment; the payment count by
itself is not the limit. A multi-payment option with no positive interval has no final date: it
stays an undefined schedule, rejected downstream without inventing a cadence.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_profile, mk_request, monthly, run_case
from buyorwait.models import PaymentOption
from buyorwait.planning import installment_eligible

RD = date(2026, 1, 5)


def opt(n, freq, first=RD, amount="100", fee="0"):
    return PaymentOption("payment_option_1", "r1", "installments", D(amount), n, first, freq, D(fee), D(amount) * n + D(fee))


def prof(maxm, balance="100000"):
    return mk_profile(max_installment_months=maxm, payment_methods=("installments",),
                      current_available_balance=D(balance), minimum_balance_to_keep=D("500"))


@pytest.mark.parametrize("n,freq,maxm", [
    (52, 7, 12),    # weekly for a year: 357 days
    (26, 14, 12),   # biweekly: 350 days
    (13, 28, 12),   # four-weekly: 336 days
    (12, 30, 12), (12, 31, 12),
    (6, 30, 6),
    (4, 7, 1),
    (1, None, 1),   # a single payment needs no interval
    (4, 28, 3),     # the old count rule rejected n=4 > 3; 84 days is inside 3 months
])
def test_plans_inside_the_permitted_duration_are_eligible(n, freq, maxm):
    ok, why = installment_eligible(opt(n, freq), prof(maxm))
    assert ok, why


@pytest.mark.parametrize("n,freq,maxm,last", [
    (3, 30, 2, "2026-03-06"),    # one day beyond 2 months from 01-05
    (2, 60, 2, "2026-03-06"),
    (3, 60, 3, "2026-05-05"),    # the old count rule accepted n=3 <= 3; 120 days is beyond 3 months
])
def test_plans_beyond_the_permitted_duration_are_rejected(n, freq, maxm, last):
    ok, why = installment_eligible(opt(n, freq), prof(maxm))
    assert not ok and f"final payment {last}" in why and "beyond max_installment_months" in why


def test_exact_duration_boundary_is_inclusive():
    assert installment_eligible(opt(2, 59), prof(2))[0]          # last 03-05 == 01-05 + 2 months
    assert installment_eligible(opt(3, 29), prof(2))[0]          # last 03-04
    assert not installment_eligible(opt(2, 60), prof(2))[0]      # last 03-06


@pytest.mark.parametrize("freq", [None, 0])
def test_multi_payment_option_without_an_interval_stays_undefined(freq):
    # eligibility does not invent a cadence; the pipeline rejects the undefined schedule
    assert installment_eligible(opt(3, freq), prof(12))[0]
    ev = monthly("sal", "salary", 2000, 15, 5, etype="income", desc="Payroll credit")
    dec, row = run_case(prof(12), ev, mk_request("300", rd=RD, deadline=RD + timedelta(days=100), partial=False), options=[opt(3, freq)])
    assert row["affordability_status"] == "not_affordable" and any("undefined" in r for r in dec.rejected)


def _pipe(o, maxm, deadline, balance="100000"):
    ev = monthly("sal", "salary", 2000, 15, 5, etype="income", desc="Payroll credit")
    return run_case(prof(maxm, balance), ev, mk_request("300", rd=RD, deadline=deadline, partial=False), options=[o])


def test_deadline_and_duration_boundaries_together():
    dec, row = _pipe(opt(3, 30), 3, date(2026, 3, 6))                    # last leg 03-06 == deadline, 60 days inside 3 months
    assert row["affordability_status"] == "affordable_with_plan" and row["payment_plan"].endswith("2026-03-06:100")
    dec, row = _pipe(opt(3, 30), 3, date(2026, 3, 5))                    # deadline one day earlier -> deadline gate
    assert row["affordability_status"] == "not_affordable" and any("after desired_completion_date" in r for r in dec.rejected)
    dec, row = _pipe(opt(2, 59, amount="150"), 2, date(2026, 3, 5))      # duration exact and deadline exact at once
    assert row["affordability_status"] == "affordable_with_plan"


def test_insufficient_funds_with_a_duration_valid_plan_is_not_affordable():
    dec, row = _pipe(opt(3, 30), 3, RD + timedelta(days=100), balance="700")
    assert row["affordability_status"] == "not_affordable" and any("unsafe through its last payment" in r for r in dec.rejected)


def test_duration_invalid_but_otherwise_affordable_plan_is_rejected_for_duration():
    dec, row = _pipe(opt(3, 60), 3, RD + timedelta(days=200))
    assert row["affordability_status"] == "not_affordable" and any("beyond max_installment_months" in r for r in dec.rejected)
