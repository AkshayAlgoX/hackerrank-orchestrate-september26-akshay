from datetime import date, timedelta
from decimal import Decimal as D

import pytest
from hypothesis import given, settings, strategies as st

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly, run_case
from buyorwait.extraction.gather import EvidenceBundle
from buyorwait.forecast import Path, amount_safe_to_pay, earliest_full_payment_date, is_safe, project_flows, simulate
from buyorwait.ledger import Flow, build_ledger
from buyorwait.models import PaymentOption
from buyorwait.planning import Plan, decide
from buyorwait.spending import candidate_changes, select_changes

RD = date(2026, 6, 2)


def ledger_for(profile, events, rd=RD, evidence=()):
    ds = mk_dataset(profile, events, mk_request(1, rd=rd))
    return build_ledger(ds, "u1", rd, list(evidence))


def test_simulate_path_and_minimum():
    flows = [Flow(date(2026, 6, 5), D("-300"), "a"), Flow(date(2026, 6, 15), D("1000"), "b"), Flow(date(2026, 6, 20), D("-900"), "c")]
    p = simulate(D("1000"), flows)
    assert p.points == [(date(2026, 6, 5), D("700")), (date(2026, 6, 15), D("1700")), (date(2026, 6, 20), D("800"))]
    assert p.minimum == D("700")
    assert p.balance_on(date(2026, 6, 10)) == D("700")
    assert p.suffix_minimum(date(2026, 6, 15)) == D("800")


def test_amount_safe_and_earliest():
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    L = ledger_for(mk_profile(current_available_balance=D("1400"), minimum_balance_to_keep=D("500")), ev)
    flows = project_flows(L)
    # rent 07-01 and 08-01 (1600) vs salaries 06-15, 07-15, 08-15 (4500): trough is before 06-15 -> room 900
    assert amount_safe_to_pay(L, flows, D("5000")) == D("900")
    assert amount_safe_to_pay(L, flows, D("100")) == D("100")
    assert earliest_full_payment_date(L, flows, D("900")) == RD
    assert earliest_full_payment_date(L, flows, D("1500")) == date(2026, 6, 15)
    assert earliest_full_payment_date(L, flows, D("10000")) is None


def test_minimum_balance_violation_detected_per_day():
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    L = ledger_for(mk_profile(current_available_balance=D("1400"), minimum_balance_to_keep=D("500")), ev)
    flows = project_flows(L)
    assert is_safe(L, flows, [(RD, D("900"))])
    assert not is_safe(L, flows, [(RD, D("900.01"))])
    assert is_safe(L, flows, [(date(2026, 6, 15), D("1500"))])
    assert not is_safe(L, flows, [(date(2026, 6, 14), D("1500"))])


@settings(max_examples=60)
@given(st.decimals(min_value=0, max_value=5000, places=2, allow_nan=False, allow_infinity=False))
def test_safe_amount_is_maximal_and_bounded(requested):
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    L = ledger_for(mk_profile(current_available_balance=D("1400"), minimum_balance_to_keep=D("500")), ev)
    flows = project_flows(L)
    safe = amount_safe_to_pay(L, flows, requested)
    assert D("0") <= safe <= requested
    assert is_safe(L, flows, [(RD, safe)])
    if safe < requested:
        assert not is_safe(L, flows, [(RD, safe + D("0.01"))])


def test_earliest_is_first_date_with_safe_suffix():
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    L = ledger_for(mk_profile(current_available_balance=D("1400"), minimum_balance_to_keep=D("500")), ev)
    flows = project_flows(L)
    for amt in (D("900"), D("1500"), D("2000")):
        e = earliest_full_payment_date(L, flows, amt)
        if e is not None:
            assert is_safe(L, flows, [(e, amt)])
            assert not is_safe(L, flows, [(e - timedelta(days=1), amt)]) or e == RD


def test_recurrence_detection_fixed_monthly_vs_variable_weekly_and_one_offs():
    ev = monthly("rent", "rent", 800, 1, 5, desc="Apartment rent")
    for i in range(10):
        ev.append(mk_event(f"g{i}", "expense", "groceries", "debit", 50 + i, date(2026, 3, 25) + timedelta(days=7 * i), desc=f"shop {i}"))
    ev.append(mk_event("big", "expense", "groceries", "debit", 900, date(2026, 5, 20), desc="bulk"))
    ev.append(mk_event("auth", "expense", "shopping", "debit", 70, date(2026, 5, 20), status="cancelled", desc="Card authorization"))
    ev.append(mk_event("one", "expense", "shopping", "debit", 70, date(2026, 5, 21), desc="Settled card purchase", linked="auth"))
    L = ledger_for(mk_profile(), ev)
    cats = {s.category: s for s in L.series}
    assert cats["rent"].period_days is None and cats["rent"].amount == D("800")
    assert cats["groceries"].period_days == 7
    assert cats["groceries"].amount == D("54.50")  # mean of 50..59, outlier 900 excluded
    assert "shopping" not in cats
    assert any("unusual" in a for a in L.audit)


def test_salary_projection_uses_scheduled_row_and_mode_amount():
    ev = monthly("sal", "salary", 1500, 15, 4, etype="income", desc="Payroll credit")
    ev[-1] = mk_event("sal3", "income", "salary", "credit", 700, date(2026, 5, 15), desc="Payroll credit")  # one reduced month
    ev.append(mk_event("next", "income", "salary", "credit", 1600, date(2026, 6, 15), status="scheduled", desc="Next confirmed salary"))
    L = ledger_for(mk_profile(), ev)
    assert [(f.on, f.amount) for f in L.salary_flows] == [(date(2026, 6, 15), D("1600")), (date(2026, 7, 15), D("1600")), (date(2026, 8, 15), D("1600"))]
    ev2 = monthly("sal", "salary", 1500, 15, 4, etype="income", desc="Payroll credit")
    ev2[-1] = mk_event("sal3", "income", "salary", "credit", 700, date(2026, 5, 15), desc="Payroll credit")
    L2 = ledger_for(mk_profile(), ev2)
    assert [f.amount for f in L2.salary_flows] == [D("1500")] * 3  # mode, not last


def test_terminal_payroll_and_non_recurring_income_not_projected():
    ev = monthly("sal", "salary", 1500, 15, 4, etype="income", desc="Payroll credit")
    ev.append(mk_event("fin", "income", "salary", "credit", 1500, date(2026, 5, 15), desc="Final employer payroll"))
    ev.append(mk_event("bonus", "income", "salary", "credit", 9000, date(2026, 5, 20), desc="Quarterly performance bonus"))
    L = ledger_for(mk_profile(), ev)
    assert L.salary_flows == [] and not any(s.is_income for s in L.series)


def test_variable_income_is_never_projected():
    # Platform/freelance payouts are unconfirmed income (sample request_10: the reference counts
    # none of them, not even after the one payout a message calls "pending").
    ev = [mk_event(f"p{i}", "income", "salary", "credit", 500, date(2026, 3, 7) + timedelta(days=15 * i), desc="Freelance milestone payment") for i in range(6)]
    L = ledger_for(mk_profile(), ev)
    assert not any(s.is_income for s in L.series) and L.salary_flows == []
    assert any("variable-income" in a for a in L.audit)


def test_duplicate_salary_representation_deduplicated_by_month():
    ev = monthly("sal", "salary", 1500, 15, 4, etype="income", desc="Payroll credit")
    ev.append(mk_event("slip", "income", "salary", "credit", 1500, date(2026, 5, 31), desc="May 2026 net salary"))
    L = ledger_for(mk_profile(), ev)
    assert [f.on for f in L.salary_flows] == [date(2026, 6, 15), date(2026, 7, 15), date(2026, 8, 15)]
    assert any("second salary record" in a for a in L.audit)


def test_installment_option_schedule_and_eligibility():
    opt = PaymentOption("payment_option_7", "r1", "installments", D("100"), 3, date(2026, 6, 5), 30, D("10"), D("300"))
    assert opt.schedule() == [(date(2026, 6, 5), D("100")), (date(2026, 7, 5), D("100")), (date(2026, 8, 4), D("100"))]
    assert opt.option_index == 7
    from buyorwait.planning import installment_eligible
    assert installment_eligible(opt, mk_profile(max_installment_months=3))[0]
    assert not installment_eligible(opt, mk_profile(max_installment_months=2))[0]
    assert not installment_eligible(opt, mk_profile(max_installment_months=None))[0]
    assert not installment_eligible(opt, mk_profile(payment_methods=("full_payment",)))[0]


def test_installment_cadence_is_a_day_count_not_a_calendar_month_step():
    """Leg k is `first_payment_date + k * payment_frequency_days`, exactly as the samples show.

    Every reference installment plan in sample_requests.csv advances by the option's own day
    count (28/30/31-day gaps -- e.g. a 30-day option starting 2025-08-08 pays 2025-09-07 and
    2025-10-07), never by calendar-month steps. A calendar-month rewrite of `schedule()` would
    give 2026-01-31 -> 2026-02-28 -> 2026-03-31 here and silently stop reproducing the reference.
    """
    def sched(n, freq, first, amt="100"):
        return PaymentOption("payment_option_1", "r1", "installments", D(amt), n, first, freq,
                             D("0"), D(amt) * n).schedule()

    first, freq = date(2026, 1, 31), 30
    assert [d for d, _ in sched(4, freq, first)] == [first + timedelta(days=freq * k) for k in range(4)]
    # a 30-day cadence crosses a month early; a 28-day one enters March when February ends
    assert [d for d, _ in sched(3, 30, date(2026, 1, 31))] == [date(2026, 1, 31), date(2026, 3, 2), date(2026, 4, 1)]
    assert [d for d, _ in sched(2, 28, date(2025, 2, 1))] == [date(2025, 2, 1), date(2025, 3, 1)]
    # the cadence rolls over 31 December without losing or duplicating a leg
    assert [d for d, _ in sched(3, 30, date(2025, 12, 15))] == [date(2025, 12, 15), date(2026, 1, 14), date(2026, 2, 13)]
    # N == 1 is the only shape a blank interval takes in the supplied data, and it is unambiguous
    assert sched(1, None, date(2026, 7, 4)) == [(date(2026, 7, 4), D("100"))]


@pytest.mark.xfail(strict=True, reason=(
    "schedule() guards the advance with `if self.payment_frequency_days:`, falsy for None AND "
    "for 0, so a multi-payment option with a blank interval writes every leg on "
    "first_payment_date -- which also makes completes_by_deadline trivially true and is "
    "undetectable by validate_row, since it re-derives its check from the same schedule(). "
    "Unreachable from dataset/request_payment_options.csv (all 275 blank-frequency rows are "
    "full_payment with number_of_payments == 1; no row is blank AND multi-payment) and its "
    "resolution is unresolved: the spec states no default interval and the solved samples show "
    "only day-count arithmetic, so no replacement cadence is warranted without a decision"))
def test_blank_frequency_never_stacks_every_leg_on_one_date():
    opt = PaymentOption("payment_option_1", "r1", "installments", D("100"), 3, date(2026, 1, 31), None, D("0"), D("300"))
    dates = [d for d, _ in opt.schedule()]
    assert dates == sorted(set(dates)), f"legs stacked on {dates[0]}"


def test_rank_key_follows_exact_order():
    a = Plan("wait", [(date(2026, 7, 1), D("100"))], [], None, D("100"), False)
    b = Plan("installments", [(date(2026, 6, 5), D("60")), (date(2026, 7, 5), D("60"))], [], None, D("120"), True)
    c = Plan("full_payment", [(RD, D("100"))], ["x"], None, D("100"), True)
    d = Plan("partial_payment", [(RD, D("40")), (date(2026, 6, 15), D("60"))], [], None, D("100"), True)
    assert sorted([a, b, c, d], key=lambda p: p.rank_key()) == [d, b, c, a]


def test_spending_change_candidates_respect_permissions_and_ordering():
    ev = monthly("stream", "streaming", 40, 10, 5, etype="subscription", flex="reducible_or_stoppable", minallowed=20, desc="Family streaming plan")
    ev += monthly("dine", "dining", 120, 5, 5, flex="reducible", minallowed=60, desc="Weekend dinner")
    ev += monthly("gym", "gym", 50, 12, 5, etype="subscription", flex="stoppable", desc="Gym plan")
    ev += monthly("rent", "rent", 800, 1, 5, flex="reducible", minallowed=10)  # protected category
    L = ledger_for(mk_profile(reduce_categories=("dining", "streaming"), stop_categories=("streaming",)), ev)
    cands = candidate_changes(L)
    assert [(c.action, c.category) for c in cands] == [("reduce_to", "streaming"), ("stop", "streaming"), ("reduce_to", "dining")]
    assert cands[0].event_id == "stream4" and cands[0].new_amount == D("20")
    assert cands[0].saving == D("60") and cands[1].saving == D("120") and cands[2].saving == D("180")


def test_select_changes_greedy_then_prune_and_stop_replaces_reduce():
    ev = monthly("stream", "streaming", 40, 10, 5, etype="subscription", flex="reducible_or_stoppable", minallowed=20, desc="Family streaming plan")
    ev += monthly("dine", "dining", 120, 5, 5, flex="reducible", minallowed=60, desc="Weekend dinner")
    ev += monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    p = mk_profile(current_available_balance=D("660"), reduce_categories=("dining", "streaming"), stop_categories=("streaming",))
    L = ledger_for(p, ev)
    # trough before 06-15: dining 120 (06-05) + streaming 40 (06-10) => room 0; need 30 => reduce streaming (20) insufficient
    chs = select_changes(L, [(RD, D("30"))])
    assert [c.render() for c in chs] == ["stop:stream4"]                # stop replaced the insufficient reduce
    chs = select_changes(L, [(RD, D("95"))])
    assert [c.render() for c in chs] == ["stop:stream4", "reduce_to:dine4:60"]  # stop replaced reduce on streaming
    assert select_changes(L, [(RD, D("500"))]) is None


def test_decide_status_mapping_and_partial_shape():
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    p = mk_profile(current_available_balance=D("1400"), payment_methods=("full_payment", "partial_payment"))
    dec, row = run_case(p, ev, mk_request(1200, partial=True, deadline=date(2026, 7, 1)))
    assert row["recommended_payment_method"] == "partial_payment"
    assert row["payment_plan"] == "2026-06-02:900|2026-06-15:300"
    assert row["earliest_date_for_full_payment"] == "2026-06-15"
    dec, row = run_case(p, ev, mk_request(900, partial=True))
    assert row["affordability_status"] == "affordable_now" and row["payment_plan"] == "2026-06-02:900"
    dec, row = run_case(mk_profile(current_available_balance=D("600"), payment_methods=("full_payment",)), ev, mk_request(50000))
    assert row["affordability_status"] == "not_affordable" and row["payment_plan"] == "none" and row["earliest_date_for_full_payment"] == ""


def test_capacity_date_survives_a_user_who_refuses_full_payment():
    """`earliest_date_for_full_payment` is capacity, not eligibility, so a refusal must not blank it.

    The user here will consider installments only, and no installment option is supplied, so no
    eligible plan exists and the status is `not_affordable`. The full amount is nevertheless safe
    to pay in one go from 2026-06-15 (the salary date), so the date must stay populated --
    "a user refusing full payment does not mean their financial capacity date is empty".
    The field measures capacity independently of payment-method preferences (problem statement),
    which is why `affordability_status` is not a gate on it.
    """
    ev = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    p = mk_profile(current_available_balance=D("1400"), payment_methods=("installments",))
    dec, row = run_case(p, ev, mk_request(1500))
    assert row["affordability_status"] == "not_affordable"
    assert row["recommended_payment_method"] == "not_recommended"
    assert row["payment_plan"] == "none"
    # capacity: an earlier date would breach the minimum, a later one is not the earliest
    assert dec.earliest_full == date(2026, 6, 15)
    assert row["earliest_date_for_full_payment"] == "2026-06-15"
