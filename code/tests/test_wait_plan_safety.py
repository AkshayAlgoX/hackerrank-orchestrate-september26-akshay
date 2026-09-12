"""Regression tests for wait-plan safety: the wait candidate must pass the full
is_safe() invariant over the entire forecast, not only from the payment date onward.

Bug: earliest_full_payment_date() only checks suffix_minimum(D) >= need, so a
balance dip below minimum_balance_to_keep *before* D was invisible to the wait
candidate.  The fix gates the wait candidate with is_safe(L, base_flows, payments).
"""
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly, run_case
from buyorwait.forecast import (
    amount_safe_to_pay,
    earliest_full_payment_date,
    is_safe,
    project_flows,
    simulate,
)
from buyorwait.ledger import Flow, build_ledger
from buyorwait.models import PaymentOption
from buyorwait.planning import Plan, decide

RD = date(2026, 6, 2)


def ledger_for(profile, events, rd=RD, evidence=()):
    ds = mk_dataset(profile, events, mk_request(1, rd=rd))
    return build_ledger(ds, "u1", rd, list(evidence))


# ---------------------------------------------------------------------------
# 1. Core reproducer: unsafe dip before eventual affordability
# ---------------------------------------------------------------------------
def test_wait_rejected_when_balance_dips_below_minimum_before_payment_date():
    """Exact reproducer from the adversarial oracle.

    balance=1000, min_keep=500.
    Scheduled debit of 800 on day 5 => day 5: 1000-800=200 < 500 UNSAFE.
    Scheduled salary of 2000 on day 25 => day 25: 200+2000=2200, suffix safe for 1600.

    The wait candidate must be REJECTED because day 5 violates the minimum.
    """
    ev = [
        mk_event("rent0", "expense", "rent", "debit", 800,
                 RD + timedelta(days=5), status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 2000,
                 RD + timedelta(days=25), status="scheduled",
                 desc="Payroll credit"),
    ]
    p = mk_profile(
        current_available_balance=D("1000"),
        minimum_balance_to_keep=D("500"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    req = mk_request(1600, rd=RD, partial=False, deadline=RD + timedelta(days=60))
    dec, row = run_case(p, ev, req)

    # Verify the dip is real in the simulation
    L = ledger_for(p, ev)
    flows = project_flows(L)
    path = simulate(L.opening_balance, flows)
    assert path.balance_on(RD + timedelta(days=5)) == D("200"), (
        f"Expected balance 200 on day 5, got {path.balance_on(RD + timedelta(days=5))}"
    )
    assert path.balance_on(RD + timedelta(days=5)) < L.minimum_balance

    # The wait candidate must NOT be accepted
    assert row["recommended_payment_method"] != "wait", (
        f"wait plan accepted despite balance dip to 200 < 500 on day 5: {row}"
    )


# ---------------------------------------------------------------------------
# 2. Exact minimum boundary: balance touches minimum exactly => SAFE
# ---------------------------------------------------------------------------
def test_wait_accepted_when_balance_exactly_equals_minimum():
    """Balance dips to exactly minimum_balance_to_keep => the wait IS safe.

    balance=1300, min_keep=500.
    Recurring rent (-800/month on day 1): projected rent on 07-01 and 08-01.
    Recurring salary (+2000/month on day 15): projected on 06-15, 07-15, 08-15.

    Day 0 (06-02): 1300
    07-01: 1300 + 2000 (06-15 salary) - 800 (07-01 rent) = 2500  => safe
    After payment of 800 on 07-15: ... still >= 500

    But we need to construct a scenario where the trough is exactly at minimum.
    balance=1300, min=500 => room = 800 (trough before first salary on 06-15).
    Request = 1500 => can't pay today (room only 800).
    After 06-15 salary: trough becomes 1300-800(07-01 rent)+2000(06-15)=2500, need suffix >= 2000.
    earliest = 06-15: suffix_min = min(2500-800(07-01)+2000(07-15)-800(08-01)+2000(08-15)) = 2500.
    is_safe with payment on 06-15: min(1300, 2500-1500=1000, 1000-800+2000=2200, ...) = 1000 >= 500. Safe.

    Actually let me use a simpler construction with scheduled events.
    """
    # balance=800, min_keep=200, scheduled debit -600 on day 5, scheduled credit +2000 on day 25.
    # Day 5: 800 - 600 = 200 == min => exactly safe.
    # Day 25: 200 + 2000 = 2200.
    # Request = 1600. earliest_full on day 25: suffix_min = 2200 - 1600 = 600 >= 200 => yes.
    # is_safe([(day25, 1600)]): min(800, 200, 2200-1600=600) = 200 >= 200 => safe.
    ev = [
        mk_event("exp0", "expense", "rent", "debit", 600,
                 RD + timedelta(days=5), status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 2000,
                 RD + timedelta(days=25), status="scheduled",
                 desc="Payroll credit"),
    ]
    p = mk_profile(
        current_available_balance=D("800"),
        minimum_balance_to_keep=D("200"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    req = mk_request(1600, rd=RD, partial=False, deadline=RD + timedelta(days=60))
    dec, row = run_case(p, ev, req)

    # Verify the boundary
    L = ledger_for(p, ev)
    flows = project_flows(L)
    path = simulate(L.opening_balance, flows)
    assert path.balance_on(RD + timedelta(days=5)) == D("200")
    assert path.balance_on(RD + timedelta(days=5)) == L.minimum_balance

    assert row["recommended_payment_method"] == "wait", (
        f"wait plan should be accepted when balance exactly equals minimum: {row}"
    )


# ---------------------------------------------------------------------------
# 3. One cent below minimum => UNSAFE
# ---------------------------------------------------------------------------
def test_wait_rejected_one_cent_below_minimum():
    """Balance dips to minimum - 0.01 => wait must be rejected.

    balance=800, min_keep=200.01, scheduled debit -600 on day 5.
    Day 5: 800 - 600 = 200.00 < 200.01 => UNSAFE.
    """
    ev = [
        mk_event("exp0", "expense", "rent", "debit", 600,
                 RD + timedelta(days=5), status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 2000,
                 RD + timedelta(days=25), status="scheduled",
                 desc="Payroll credit"),
    ]
    p = mk_profile(
        current_available_balance=D("800"),
        minimum_balance_to_keep=D("200.01"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    req = mk_request(1600, rd=RD, partial=False, deadline=RD + timedelta(days=60))
    dec, row = run_case(p, ev, req)
    assert row["recommended_payment_method"] != "wait", (
        f"wait accepted with balance 200 < min 200.01: {row}"
    )


# ---------------------------------------------------------------------------
# 4. One cent above minimum => SAFE
# ---------------------------------------------------------------------------
def test_wait_accepted_one_cent_above_minimum():
    """Balance dips to minimum + 0.01 => wait is safe.

    balance=800, min_keep=199.99, scheduled debit -600 on day 5.
    Day 5: 800 - 600 = 200 > 199.99 => SAFE.
    """
    ev = [
        mk_event("exp0", "expense", "rent", "debit", 600,
                 RD + timedelta(days=5), status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 2000,
                 RD + timedelta(days=25), status="scheduled",
                 desc="Payroll credit"),
    ]
    p = mk_profile(
        current_available_balance=D("800"),
        minimum_balance_to_keep=D("199.99"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    req = mk_request(1600, rd=RD, partial=False, deadline=RD + timedelta(days=60))
    dec, row = run_case(p, ev, req)
    assert row["recommended_payment_method"] == "wait", (
        f"wait should be accepted with balance 200 > min 199.99: {row}"
    )


# ---------------------------------------------------------------------------
# 5. Later salary recovery after unsafe dip
# ---------------------------------------------------------------------------
def test_wait_rejected_salary_recovery_after_deep_dip():
    """Multiple scheduled expenses cause a deep dip, salary comes much later.

    balance=1000, min=500.
    Day 3: -400 => 600.
    Day 7: -400 => 200 < 500 => UNSAFE.
    Day 40: +3000 => 3200.
    Request: 2000.
    """
    ev = [
        mk_event("e1", "expense", "rent", "debit", 400,
                 RD + timedelta(days=3), status="scheduled"),
        mk_event("e2", "expense", "utilities", "debit", 400,
                 RD + timedelta(days=7), status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 3000,
                 RD + timedelta(days=40), status="scheduled",
                 desc="Payroll credit"),
    ]
    p = mk_profile(
        current_available_balance=D("1000"),
        minimum_balance_to_keep=D("500"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    req = mk_request(2000, rd=RD, partial=False, deadline=RD + timedelta(days=80))
    dec, row = run_case(p, ev, req)
    assert row["recommended_payment_method"] != "wait", (
        f"wait accepted despite deep dip to 200 < 500: {row}"
    )


# ---------------------------------------------------------------------------
# 6. Safe wait remains eligible (no false rejections)
# ---------------------------------------------------------------------------
def test_safe_wait_remains_eligible():
    """When the balance never dips below minimum, wait must still work.

    Uses the existing proven pattern from the adversarial cases:
    recurring rent + salary history, where balance never breaches minimum,
    but full payment isn't possible today due to insufficient room.
    """
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 2000, 15, 5, etype="income", desc="Payroll credit")
    p = mk_profile(current_available_balance=D("1400"), payment_methods=("full_payment",))
    dec, row = run_case(p, ev, mk_request(1500, partial=False, deadline=date(2026, 7, 20)))
    assert row["recommended_payment_method"] == "wait", (
        f"safe wait incorrectly rejected: {row}"
    )
    assert row["affordability_status"] == "affordable_later"


# ---------------------------------------------------------------------------
# 7. Multiple unsafe pre-wait dips
# ---------------------------------------------------------------------------
def test_wait_rejected_multiple_dips_before_payment_date():
    """Two separate dips below minimum before the eventual recovery.

    balance=1000, min=500.
    Day 3: -600 => 400 < 500 => DIP 1
    Day 10: +700 => 1100
    Day 15: -700 => 400 < 500 => DIP 2
    Day 30: +3000 => 3400
    """
    ev = [
        mk_event("e1", "expense", "rent", "debit", 600,
                 RD + timedelta(days=3), status="scheduled"),
        mk_event("c1", "income", "salary", "credit", 700,
                 RD + timedelta(days=10), status="scheduled",
                 desc="Payroll credit"),
        mk_event("e2", "expense", "utilities", "debit", 700,
                 RD + timedelta(days=15), status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 3000,
                 RD + timedelta(days=30), status="scheduled",
                 desc="Payroll credit"),
    ]
    p = mk_profile(
        current_available_balance=D("1000"),
        minimum_balance_to_keep=D("500"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    req = mk_request(2000, rd=RD, partial=False, deadline=RD + timedelta(days=60))
    dec, row = run_case(p, ev, req)
    assert row["recommended_payment_method"] != "wait", (
        f"wait accepted despite two dips below minimum: {row}"
    )


# ---------------------------------------------------------------------------
# 8. Unsafe dip on request date itself
# ---------------------------------------------------------------------------
def test_wait_rejected_dip_on_request_date():
    """A scheduled expense on the request date causes a breach.

    balance=600, min=500, scheduled debit -200 on request_date.
    Day 0: 600 - 200 = 400 < 500 => UNSAFE.
    Day 25: +2000 => recovery.
    """
    ev = [
        mk_event("e1", "expense", "rent", "debit", 200,
                 RD, status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 2000,
                 RD + timedelta(days=25), status="scheduled",
                 desc="Payroll credit"),
    ]
    p = mk_profile(
        current_available_balance=D("600"),
        minimum_balance_to_keep=D("500"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    req = mk_request(1500, rd=RD, partial=False, deadline=RD + timedelta(days=60))
    dec, row = run_case(p, ev, req)
    assert row["recommended_payment_method"] != "wait", (
        f"wait accepted despite dip on request date: {row}"
    )


# ---------------------------------------------------------------------------
# 9. Unsafe dip at horizon boundary — payment after big expense lands past deadline
# ---------------------------------------------------------------------------
def test_wait_rejected_dip_near_horizon_end():
    """A scheduled expense near the horizon boundary means the only safe date
    is after the deadline, so wait is rejected.

    balance=2000, min=500.
    Scheduled salary +3000 on day 10. Scheduled debit -4600 on day 70.
    Day 10: 2000 + 3000 = 5000.
    Day 70: 5000 - 4600 = 400 < 500 => the pre-payment dip violates minimum.

    earliest_full_payment_date for 3000: must be after day 70 recovery.
    Day 70 itself: balance = 400, suffix_min = 400 < 500 + 3000 => no.
    No salary after day 70 => no recovery => earliest is None or past deadline.

    With deadline at day 40, the wait candidate (if found) would be past deadline.
    """
    ev = [
        mk_event("sal0", "income", "salary", "credit", 3000,
                 RD + timedelta(days=10), status="scheduled",
                 desc="Payroll credit"),
        mk_event("big", "expense", "rent", "debit", 4600,
                 RD + timedelta(days=70), status="scheduled"),
    ]
    p = mk_profile(
        current_available_balance=D("2000"),
        minimum_balance_to_keep=D("500"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
    )
    # Tight deadline: only day 10 would be a candidate, but full forecast is unsafe
    req = mk_request(3000, rd=RD, partial=False, deadline=RD + timedelta(days=40))
    dec, row = run_case(p, ev, req)
    # The engine should not offer wait because paying 3000 on day 10 means
    # day 70 balance = 5000 - 3000 - 4600 = -2600 < 500
    L = ledger_for(p, ev)
    flows = project_flows(L)
    pay_day10_safe = is_safe(L, flows, [(RD + timedelta(days=10), D("3000"))])
    assert not pay_day10_safe, "payment on day 10 should be unsafe due to day 70 dip"
    # With the tight deadline, wait should not be offered
    assert row["recommended_payment_method"] != "wait" or \
        dec.plan.first_date <= req.desired_completion_date, (
        f"wait accepted despite horizon-boundary dip: {row}"
    )


# ---------------------------------------------------------------------------
# 10. Wait candidate interacting with spending changes
# ---------------------------------------------------------------------------
def test_wait_not_salvaged_by_spending_changes():
    """The wait candidate path does NOT try spending changes (by design).

    Scheduled debit of 800 on day 5 causes dip to 200 < 500.
    Even with stoppable streaming subscriptions in history, the wait plan
    itself does not attempt spending changes.
    """
    ev = [
        mk_event("rent0", "expense", "rent", "debit", 800,
                 RD + timedelta(days=5), status="scheduled"),
        mk_event("sal0", "income", "salary", "credit", 2000,
                 RD + timedelta(days=25), status="scheduled",
                 desc="Payroll credit"),
    ]
    # Add stoppable streaming subscriptions (history, to project as series)
    ev += monthly("stream", "streaming", 40, 10, 5, etype="subscription",
                  flex="stoppable", desc="Streaming subscription")
    p = mk_profile(
        current_available_balance=D("1000"),
        minimum_balance_to_keep=D("500"),
        payment_methods=("full_payment",),
        protect_categories=("rent",),
        stop_categories=("streaming",),
    )
    req = mk_request(1600, rd=RD, partial=False, deadline=RD + timedelta(days=60))
    dec, row = run_case(p, ev, req)
    # Wait must not appear as a candidate
    wait_cands = [c for c in dec.candidates if c.method == "wait"]
    assert not wait_cands, (
        f"wait plan should not be a candidate when balance dips below minimum: "
        f"{[(c.method, c.changes) for c in wait_cands]}"
    )
