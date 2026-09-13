"""Isolated regression tests for the hidden-test exposures named by the T3/T4 audits.

Each case builds a synthetic dataset in memory and asserts one general invariant. There are no
dataset files, no public request/user/event ids and no sample-specific amounts, so a case can
only pass by holding for the stated reason. Nothing here changes engine behaviour.

Cases marked ``xfail(strict=True)`` pin a property the engine does **not** currently provide:
the marker documents the gap (its reason names the mechanism) and fails loudly the moment the
gap is closed. They follow the same convention as the non-reproduced sample rows in
``test_sample_regression.py``.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly
from buyorwait.extraction.gather import EvidenceBundle
from buyorwait.forecast import is_safe, project_flows, simulate
from buyorwait.fx import MissingRate, lookup_rate
from buyorwait.ledger import build_ledger
from buyorwait.models import Dataset, FxTable, PaymentOption, Request
from buyorwait.output import render_row, validate_row
from buyorwait.pipeline import decide_request, run
from buyorwait.planning import decide
from buyorwait.spending import candidate_changes

RD = date(2026, 6, 2)

# The invariant counterexample the T3/T4 audit predicted: an out-of-horizon leg is dropped by
# `is_safe` (which keeps only `d <= horizon_end`), so the plan looks safer than it is.
OUT_OF_HORIZON = (
    "is_safe drops payments dated after horizon_end, so a schedule whose later legs fall "
    "outside the forecast window is judged on a prefix of itself"
)
def ledger_for(profile, events, rd=RD, evidence=()):
    ds = mk_dataset(profile, events, mk_request(1, rd=rd))
    return build_ledger(ds, profile.user_id, rd, list(evidence))


def row_of(dec):
    """The rendered output row as a plain dict (``render_row`` returns a dataclass)."""
    row = render_row(dec)
    return dict(zip(row.__dataclass_fields__, row.as_list()))


def payroll_history(months=5, amount=1500, day=15):
    return monthly("sal", "salary", amount, day, months, etype="income", desc="Payroll credit")


def history_for(uid, months=5, amount=1500, day=15):
    """A per-user monthly payroll history (``monthly()`` always tags events with the default id)."""
    return [mk_event(f"{uid}_sal{i}", "salary", "salary", "credit", amount, date(2026, m, day),
                     desc="Payroll credit", user=uid)
            for i, m in enumerate(range(1, months + 1))]


def _req(rid, uid, amount="100", rd=RD, partial=False):
    return Request(rid, uid, rd, "purchase", D(amount), rd + timedelta(days=40), partial, "?")


def _batch(*specs, fx=None):
    """Assemble a multi-request dataset: specs are (profile, request, events) triples."""
    profiles, requests, events, by_user = {}, [], [], {}
    for profile, request, evs in specs:
        profiles[profile.user_id] = profile
        requests.append(request)
        events.extend(evs)
        by_user[profile.user_id] = sorted(evs, key=lambda e: (e.event_date, e.event_id))
    return Dataset(profiles=profiles, events=events, events_by_user=by_user,
                   events_by_id={e.event_id: e for e in events}, requests=requests,
                   options_by_request={r.request_id: [] for r in requests},
                   messages=[], images=[], fx=fx or FxTable())


# ---------------------------------------------------------------------------------------
# 1. missing FX pair / reverse-pair handling
# ---------------------------------------------------------------------------------------

def test_reverse_pair_is_never_inverted_to_supply_a_missing_direction():
    """A supplied EUR->USD rate must not be inverted into a made-up USD->EUR rate."""
    fx = FxTable()
    fx.add(date(2026, 6, 1), "USD", "EUR", D("0.9"))
    assert lookup_rate(fx, date(2026, 6, 1), "USD", "EUR") == (D("0.9"), date(2026, 6, 1))
    with pytest.raises(MissingRate):
        lookup_rate(fx, date(2026, 6, 1), "EUR", "USD")


def test_rate_uses_the_settlement_date_then_the_nearest_supplied_date():
    """Settlement-date exact match wins; otherwise the most recent earlier, else earliest later."""
    fx = FxTable()
    fx.add(date(2026, 6, 1), "USD", "EUR", D("0.9"))
    fx.add(date(2026, 7, 1), "USD", "EUR", D("0.8"))
    assert lookup_rate(fx, date(2026, 7, 1), "USD", "EUR") == (D("0.8"), date(2026, 7, 1))
    # between the two rows: the most recent earlier row, reported with its own date
    assert lookup_rate(fx, date(2026, 6, 20), "USD", "EUR") == (D("0.9"), date(2026, 6, 1))
    # before every row: the earliest later row, not a fabricated 1.0
    assert lookup_rate(fx, date(2026, 5, 1), "USD", "EUR") == (D("0.9"), date(2026, 6, 1))


def test_same_currency_needs_no_rate_row():
    """Home-currency events must never depend on the FX table."""
    assert lookup_rate(FxTable(), RD, "EUR", "EUR") == (D("1"), RD)


def test_foreign_history_event_without_a_rate_is_excluded_rather_than_fatal():
    # settled history in a currency with no supplied rate: excluded from the recurrence
    # estimate and audited (never a guessed rate). A PENDING/SCHEDULED debit without a rate is
    # a different matter - it still has to leave the account, so the request fails closed
    # (see test_fail_closed_unresolved_cash.py).
    ev = payroll_history()
    ev += [mk_event("fx1", "expense", "utilities", "debit", 100, date(2026, 6, 5) - timedelta(days=30),
                    status="settled", currency="USD")]
    L = ledger_for(mk_profile(), ev)
    assert not any(s.category == "utilities" for s in L.series)
    assert any("no rates for" in a for a in L.audit)


# ---------------------------------------------------------------------------------------
# 2. a scheduled annual bonus must not become recurring salary
# ---------------------------------------------------------------------------------------

def test_settled_bonus_does_not_raise_the_projected_salary():
    """A settled bonus is history, not a pay rise: the projected payday amount stays the salary."""
    ev = payroll_history(amount=1500)
    ev.append(mk_event("bon", "bonus", "salary", "credit", 5000, date(2026, 4, 15),
                       desc="Annual performance bonus"))
    L = ledger_for(mk_profile(), ev)
    assert L.salary_flows, "the monthly payroll history must still be projected"
    assert {f.amount for f in L.salary_flows} == {D("1500")}
    assert not any(f.source_event_id == "bon" for f in L.salary_flows + L.known_flows)


def test_scheduled_bonus_is_not_counted_as_salary():
    """Closed: scheduled credits are counted only when the income classifier calls them payroll."""
    ev = payroll_history(amount=1500)
    ev.append(mk_event("bon", "bonus", "salary", "credit", 5000, date(2026, 6, 20),
                       status="scheduled", desc="Annual performance bonus"))
    L = ledger_for(mk_profile(), ev)
    assert not any(f.source_event_id == "bon" for f in L.salary_flows + L.known_flows)


def test_scheduled_payroll_is_still_counted_once():
    """The counterpart: a genuinely scheduled payroll credit is a confirmed fact."""
    ev = payroll_history(amount=1500)
    ev.append(mk_event("next", "salary", "salary", "credit", 1500, date(2026, 6, 15),
                       status="scheduled", desc="Payroll credit"))
    L = ledger_for(mk_profile(), ev)
    assert [f.on for f in L.salary_flows].count(date(2026, 6, 15)) == 1


# ---------------------------------------------------------------------------------------
# 3. a late-month request with an installment leg beyond the current horizon
# ---------------------------------------------------------------------------------------

def _long_option(request_id="r1", n=4, amount="250", first=date(2026, 7, 1), freq=31):
    return PaymentOption("opt_1", request_id, "installments", D(amount), n, first, freq,
                         D("0"), D(amount) * n)


def test_installment_plan_is_rendered_in_full_even_beyond_the_horizon():
    """The output must carry every supplied leg, including dates past the forecast window."""
    p = mk_profile(current_available_balance=D("5000"), max_installment_months=12,
                   payment_methods=("installments",))
    # the deadline covers every leg (a late schedule is rejected outright); the horizon does not
    req = mk_request(1000, rd=date(2026, 6, 25), partial=False, deadline=date(2026, 10, 15))
    ds = mk_dataset(p, payroll_history(), req, options=[_long_option()])
    dec = decide_request(ds, req, EvidenceBundle([], []))
    L = dec.ledger
    assert L.horizon_end < _long_option().schedule()[-1][0] <= req.desired_completion_date
    assert dec.method == "installments"
    assert dec.plan.payments == _long_option().schedule()
    assert row_of(dec)["payment_plan"].count("|") == len(_long_option().schedule()) - 1


def test_installment_missing_the_deadline_is_not_affordable_with_plan():
    """Closed by planning.completes_by_deadline(): a schedule whose final leg falls after
    desired_completion_date is rejected before ranking, so it can never win as the only
    candidate. Invariant: no affordable_with_plan / affordable_later decision may carry a
    mandatory leg after the deadline."""
    p = mk_profile(current_available_balance=D("5000"), max_installment_months=12,
                   payment_methods=("installments",))
    sched = _long_option().schedule()                          # 07-01, 08-01, 09-01, 10-02
    req = mk_request(1000, rd=date(2026, 6, 25), partial=False, deadline=date(2026, 8, 15))
    ds = mk_dataset(p, payroll_history(), req, options=[_long_option()])
    dec = decide_request(ds, req, EvidenceBundle([], []))
    assert sched[-1][0] > req.desired_completion_date          # the option really does miss the deadline
    assert dec.plan is None and dec.candidates == []
    assert (dec.status, dec.method) == ("not_affordable", "not_recommended")
    assert any("after desired_completion_date" in r for r in dec.rejected)
    # the same option is eligible the moment the deadline covers its final leg (even past the
    # nominal forecast window: the extended ledger verifies it)
    on_time = mk_request(1000, rd=date(2026, 6, 25), partial=False, deadline=sched[-1][0])
    dec2 = decide_request(mk_dataset(p, payroll_history(), on_time, options=[_long_option()]), on_time, EvidenceBundle([], []))
    assert dec2.status == "affordable_with_plan" and dec2.plan.payments == sched
    late = mk_request(1000, rd=date(2026, 6, 25), partial=False, deadline=sched[-1][0] - timedelta(days=1))
    dec3 = decide_request(mk_dataset(p, payroll_history(), late, options=[_long_option()]), late, EvidenceBundle([], []))
    assert dec3.plan is None and dec3.status == "not_affordable"


def test_partial_payment_second_leg_is_never_after_the_deadline():
    """partial_payment is only eligible when its second leg lands on or before the deadline."""
    ev = payroll_history(amount=1500)
    p = mk_profile(current_available_balance=D("900"),
                   payment_methods=("full_payment", "partial_payment"))
    req = mk_request(1500, rd=RD, partial=True, deadline=date(2026, 6, 3))
    ds = mk_dataset(p, ev, req)
    dec = decide_request(ds, req, EvidenceBundle([], []))
    if dec.method == "partial_payment":
        assert dec.plan.payments[-1][0] <= req.desired_completion_date
    else:
        assert any("earliest<=deadline" in r for r in dec.rejected)


# ---------------------------------------------------------------------------------------
# 4. a pending debit cancelled by a linked downstream event
# ---------------------------------------------------------------------------------------

def test_pending_debit_with_a_cancelled_linked_child_stays_reserved():
    """Adjudicated (Target #3): reservation follows the row's OWN cash state.

    AGENTS.md §6.1: "linked_event_id points to an earlier event in the same lifecycle; the link
    alone does not determine whether a row counts toward cash flow. Treat settled, pending,
    scheduled ... according to their cash state." §6.3: "Reserve pending debits"; conflict rule
    4 prefers the financially safer reading. In the dataset a cancellation is always the row's
    own status (every cancelled row is an authorisation; no cancelled child exists), so a later
    cancelled child linked to a pending debit is ignored as a cancelled transaction while the
    pending debit keeps its own state and stays reserved. The earlier strict xfail encoded the
    competing reading ("a linked cancelled child cancels its parent"), which is exactly the
    link-alone inference the specification rules out and the less safe of the two."""
    ev = payroll_history()
    ev.append(mk_event("pend", "expense", "utilities", "debit", 300, date(2026, 6, 5),
                       status="pending"))
    ev.append(mk_event("cxl", "expense", "utilities", "debit", 300, date(2026, 6, 5),
                       status="cancelled", linked="pend", desc="Card authorization cancelled"))
    L = ledger_for(mk_profile(), ev)
    assert any(f.source_event_id == "pend" and f.amount == D("-300") for f in L.known_flows)
    assert not any(f.source_event_id == "cxl" for f in L.known_flows)      # the cancelled row itself never counts


def test_pending_debit_whose_own_status_is_cancelled_is_never_reserved():
    """The explicit cancellation the specification means: the row's own status."""
    ev = payroll_history()
    ev.append(mk_event("pend", "expense", "utilities", "debit", 300, date(2026, 6, 5), status="cancelled"))
    L = ledger_for(mk_profile(), ev)
    assert not any(f.source_event_id == "pend" for f in L.known_flows)


@pytest.mark.parametrize("child_status,child_day,child_cat", [
    ("cancelled", 4, "utilities"), ("cancelled", 20, "utilities"), ("cancelled", 5, "shopping"),
    ("settled", 6, "utilities"), ("failed", 6, "utilities"),
])
def test_linked_children_of_any_status_never_change_the_parents_reservation(child_status, child_day, child_cat):
    ev = payroll_history()
    ev.append(mk_event("pend", "expense", "utilities", "debit", 300, date(2026, 6, 5), status="pending"))
    ev.append(mk_event("child", "expense", child_cat, "debit", 300, date(2026, 6, child_day), status=child_status, linked="pend"))
    L = ledger_for(mk_profile(), ev)
    assert any(f.source_event_id == "pend" and f.amount == D("-300") for f in L.known_flows)


def test_pending_debit_without_a_cancellation_is_still_reserved():
    """The counterpart: absent an explicit cancellation the pending debit stays reserved."""
    ev = payroll_history()
    ev.append(mk_event("pend", "expense", "utilities", "debit", 300, date(2026, 6, 5),
                       status="pending"))
    L = ledger_for(mk_profile(), ev)
    assert any(f.source_event_id == "pend" and f.amount == D("-300") for f in L.known_flows)


# ---------------------------------------------------------------------------------------
# 5. monthly essential bills whose descriptions vary by month
# ---------------------------------------------------------------------------------------

def _bills(descriptions, day=5, amount=100, months=(9, 10, 11, 12), year=2025):
    """Settled monthly utilities rows, one per month, with the given wording."""
    return [mk_event(f"u{i}", "expense", "utilities", "debit", amount, date(year, m, day), desc=desc)
            for i, (desc, m) in enumerate(zip(descriptions, months))]


def test_constant_description_monthly_bill_is_projected():
    """The baseline the next case is measured against: a stable description forms a series."""
    ev = payroll_history() + _bills(["Electricity bill"] * 4)
    L = ledger_for(mk_profile(), ev)
    util = [s for s in L.series if s.category == "utilities"]
    assert util and util[0].amount == D("100")
    assert [d for s in util for d in s.dates(L.request_date, L.horizon_end)]


def test_essential_bills_are_projected_when_each_month_has_a_different_description():
    """Closed (Target #4): the per-category fallback now applies the same monthly rule as the
    description path (>= 3 rows, every gap 26-35 days, no month skipped), so varying wording
    cannot hide a monthly commitment from the forecast."""
    ev = payroll_history() + _bills(["Electricity bill", "Power utility charge",
                                     "Electric bill payment", "Utility invoice"])
    L = ledger_for(mk_profile(), ev)
    util = [s for s in L.series if s.category == "utilities"]
    assert len(util) == 1 and util[0].amount == D("100") and util[0].period_days is None
    assert util[0].occurrences == 4 and util[0].note == "monthly cadence across varying descriptions"
    projected = [d for s in util for d in s.dates(L.request_date, L.horizon_end)]
    baseline = ledger_for(mk_profile(), payroll_history() + _bills(["Electricity bill"] * 4))
    base_series = [s for s in baseline.series if s.category == "utilities"]
    assert projected and projected == [d for s in base_series for d in s.dates(baseline.request_date, baseline.horizon_end)]
    assert [s.amount for s in base_series] == [util[0].amount]          # identical to the stable-wording baseline


def test_varying_amounts_with_varying_descriptions_use_the_mean_like_the_description_path():
    ev = payroll_history() + _bills(["Electricity bill", "Power utility charge", "Electric bill payment", "Utility invoice"],
                                    amount=100)
    ev[-1] = mk_event("u3", "expense", "utilities", "debit", 104, date(2025, 12, 5), desc="Utility invoice")
    L = ledger_for(mk_profile(), ev)
    assert [s.amount for s in L.series if s.category == "utilities"] == [D("101.00")]


@pytest.mark.parametrize("months", [(9, 10, 12), (8, 10, 12)])
def test_a_skipped_month_still_breaks_the_monthly_claim_for_varying_descriptions(months):
    """The bar is unchanged: a month without an occurrence is not a monthly commitment."""
    ev = payroll_history() + _bills(["Electricity bill", "Power utility charge", "Electric bill payment"], months=months)
    L = ledger_for(mk_profile(), ev)
    assert not any(s.category == "utilities" for s in L.series)


def test_one_time_purchases_are_not_promoted_by_the_category_fallback():
    ev = payroll_history() + [
        mk_event("x", "expense", "utilities", "debit", 900, date(2025, 12, 5), desc="Boiler repair"),                    # single row
        mk_event("y1", "expense", "shopping", "debit", 100, date(2025, 10, 1), desc="A"),                              # irregular trio
        mk_event("y2", "expense", "shopping", "debit", 100, date(2025, 10, 11), desc="B"),
        mk_event("y3", "expense", "shopping", "debit", 100, date(2025, 11, 25), desc="C"),
        mk_event("z1", "expense", "gifts", "debit", 50, date(2025, 11, 5), desc="A"),                                  # only two rows
        mk_event("z2", "expense", "gifts", "debit", 50, date(2025, 12, 5), desc="B"),
    ]
    L = ledger_for(mk_profile(), ev)
    assert not any(s.category in ("utilities", "shopping", "gifts") for s in L.series)


def test_sub_monthly_categories_keep_the_existing_variable_cadence_detector():
    """A weekly grocery habit with rotating merchants is still a 7-day variable series."""
    ev = payroll_history() + [mk_event(f"g{i}", "expense", "groceries", "debit", 40 + i, date(2025, 10, 6) + timedelta(days=7 * i), desc=f"Shop {i}")
                              for i in range(10)]
    L = ledger_for(mk_profile(), ev)
    g = [s for s in L.series if s.category == "groceries"]
    assert len(g) == 1 and g[0].period_days == 7


# ---------------------------------------------------------------------------------------
# 6. one request throwing must not remove the other rows
# ---------------------------------------------------------------------------------------

def test_one_bad_request_does_not_remove_the_other_rows():
    """Now that MissingRate is caught, the foreign-currency row is excluded and
    all three users produce valid decisions (no errors)."""
    a, b, c = mk_profile(user_id="u_a"), mk_profile(user_id="u_b"), mk_profile(user_id="u_c")
    # a settled history row in a currency the table has no rate for: it is excluded, not fatal
    fx_events = history_for("u_c") + [
        mk_event("x", "expense", "utilities", "debit", 100, date(2026, 5, 5),
                 currency="USD", user="u_c")]
    ds = _batch((a, _req("r_a", "u_a"), history_for("u_a")),
                (b, _req("r_b", "u_b"), history_for("u_b")),
                (c, _req("r_c", "u_c"), fx_events))
    result = run(ds, bundle=EvidenceBundle([], []))
    assert [r.request_id for r in result.rows] == ["r_a", "r_b", "r_c"]
    # No errors — the unrateable event is excluded, not fatal
    assert result.errors == {} and result.violations == {}
    # u_a's row is identical whether u_c has the foreign event or not
    assert result.rows[0].as_list() == run(_batch((a, _req("r_a", "u_a"), history_for("u_a"))),
                                           bundle=EvidenceBundle([], [])).rows[0].as_list()


# ---------------------------------------------------------------------------------------
# 7. exact minimum-balance equality
# ---------------------------------------------------------------------------------------

def test_balance_exactly_at_the_minimum_is_safe_and_one_cent_more_is_not():
    """The safety check is inclusive: landing exactly on the floor passes, a cent below fails."""
    ev = payroll_history(amount=1500) + monthly("rent", "rent", 800, 1, 5)
    L = ledger_for(mk_profile(current_available_balance=D("1400"), minimum_balance_to_keep=D("500")), ev)
    flows = project_flows(L)
    room = simulate(L.opening_balance, flows).minimum - L.minimum_balance
    assert room > D("0")
    assert is_safe(L, flows, [(RD, room)])
    assert not is_safe(L, flows, [(RD, room + D("0.01"))])


# ---------------------------------------------------------------------------------------
# 8. partial payment after the deadline
# ---------------------------------------------------------------------------------------

def test_partial_payment_is_refused_when_the_second_leg_misses_the_deadline():
    """No partial_payment above the deadline, and the two legs must still sum to the request.

    Only 900 of the 1500 is safe today, so the engine *wants* to split the payment; with a
    deadline earlier than the salary that would fund the remainder, the split is not eligible and
    a later single payment is recommended instead. Either way the row must stay contract-valid.
    """
    ev = payroll_history(amount=1500)
    p = mk_profile(current_available_balance=D("900"),
                   payment_methods=("full_payment", "partial_payment"))
    for deadline in (date(2026, 6, 3), date(2026, 6, 14)):
        req = mk_request(1500, rd=RD, partial=True, deadline=deadline)
        ds = mk_dataset(p, ev, req)
        dec = decide_request(ds, req, EvidenceBundle([], []))
        row = row_of(dec)
        assert validate_row(row, req, ()) == []
        if row["recommended_payment_method"] == "partial_payment":
            dates = [row["payment_plan"].split("|")[i].split(":")[0] for i in (0, 1)]
            assert date.fromisoformat(dates[1]) <= deadline
            assert sum(D(x.split(":")[1]) for x in row["payment_plan"].split("|")) == req.requested_amount
        else:
            # the split was refused, so this is not a partial_payment row; the contract check
            # above already guarantees whatever is recommended is internally consistent
            assert row["recommended_payment_method"] != "partial_payment"


# ---------------------------------------------------------------------------------------
# 9. protected spending category
# ---------------------------------------------------------------------------------------

def test_protected_category_is_never_offered_as_a_spending_change():
    """Even when the profile also lists the category as stoppable, protection wins."""
    ev = payroll_history(amount=1500)
    for i, month in enumerate([1, 2, 3, 4, 5]):
        ev.append(mk_event(f"med{i}", "expense", "healthcare", "debit", 200, date(2026, month, 5),
                           flex="stoppable", desc="Clinic visit"))
    p = mk_profile(current_available_balance=D("800"), protect_categories=("healthcare",),
                   stop_categories=("healthcare",))
    L = ledger_for(p, ev)
    assert not [c for c in candidate_changes(L) if c.category == "healthcare"]


def test_no_recommended_change_ever_targets_a_protected_category():
    """The protection must hold on the rendered row, not only in the candidate list."""
    ev = payroll_history(amount=1500)
    for i, month in enumerate([1, 2, 3, 4, 5]):
        ev.append(mk_event(f"med{i}", "expense", "healthcare", "debit", 200, date(2026, month, 5),
                           flex="stoppable", desc="Clinic visit"))
        ev.append(mk_event(f"fun{i}", "expense", "streaming", "debit", 60, date(2026, month, 6),
                           flex="stoppable", desc="Streaming plan"))
    p = mk_profile(current_available_balance=D("700"), protect_categories=("healthcare",),
                   stop_categories=("streaming",))
    req = mk_request(900, rd=RD)
    ds = mk_dataset(p, ev, req)
    dec = decide_request(ds, req, EvidenceBundle([], []))
    protected = {e.event_id for e in ev if e.category == "healthcare"}
    for c in (dec.plan.changes if dec.plan else []):
        assert c.event_id not in protected


# ---------------------------------------------------------------------------------------
# 10. a malformed request or payment option must not crash the batch
# ---------------------------------------------------------------------------------------

def test_zero_amount_request_is_answered_without_crashing():
    """A degenerate request still yields one contract-valid row."""
    req = mk_request(0, rd=RD, partial=False)
    ds = mk_dataset(mk_profile(), payroll_history(), req)
    dec = decide_request(ds, req, EvidenceBundle([], []))
    row = render_row(dec)
    assert D(row.amount_safe_to_pay) == D("0")
    assert row.request_id == req.request_id


def test_option_with_zero_payments_does_not_crash_the_batch():
    """Closed by planning.completes_by_deadline(): an empty schedule is rejected, not indexed."""
    p = mk_profile(current_available_balance=D("5000"), max_installment_months=12)
    req = mk_request(500, rd=RD, partial=False)
    ds = mk_dataset(p, payroll_history(), req, options=[_long_option(n=0)])
    dec = decide_request(ds, req, EvidenceBundle([], []))
    assert dec.status in ("affordable_now", "affordable_with_plan", "affordable_later", "not_affordable")
    assert any("empty payment schedule" in r for r in dec.rejected)
    assert not any(c.option is not None for c in dec.candidates)


def test_option_schedule_is_never_silently_reordered_or_rescaled():
    """A supplied option is a contract: the plan must be its legs, in its order."""
    p = mk_profile(current_available_balance=D("5000"), max_installment_months=12)
    req = mk_request(1000, rd=RD, partial=False, deadline=date(2026, 8, 15))
    opt = _long_option(n=3, amount="350", first=date(2026, 7, 3), freq=30)
    ds = mk_dataset(p, payroll_history(), req, options=[opt])
    dec = decide_request(ds, req, EvidenceBundle([], []))
    if dec.method == "installments":
        assert dec.plan.payments == opt.schedule()
        assert dec.plan.total_paid == opt.total_payable_amount

def test_missing_amount_in_recurring_series_averages_known_occurrences():
    import dataclasses
    from decimal import Decimal as D
    from datetime import date
    ev = payroll_history() + _bills(["Elec"] * 4, amount=100)
    L_known = ledger_for(mk_profile(), ev)
    
    # Drop amount for one event
    ev_missing = payroll_history() + _bills(["Elec"] * 4, amount=100)
    ev_missing = [dataclasses.replace(e, amount=None) if e.description == "Elec" and e.event_date == date(2025, 11, 5) else e for e in ev_missing]
            
    L_missing = ledger_for(mk_profile(), ev_missing)
    # the series should still be detected!
    assert [s.amount for s in L_known.series if s.category == "utilities"] == [D("100")]
    assert [s.amount for s in L_missing.series if s.category == "utilities"] == [D("100")]

def test_missing_all_amounts_in_recurring_series_fails_closed():
    from buyorwait.ledger import UnresolvedCashEvidence
    import dataclasses
    import pytest
    ev = payroll_history() + _bills(["Elec"] * 4, amount=100)
    ev = [dataclasses.replace(e, amount=None) if e.description == "Elec" else e for e in ev]
    with pytest.raises(UnresolvedCashEvidence):
        ledger_for(mk_profile(), ev)
