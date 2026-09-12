"""Every leg of a recommended installment plan is checked, including legs after the nominal
forecast window (forecast.is_safe never drops a payment; planning.decide validates a long
schedule on the same ledger projected to its last leg).

Statement: "A recommendation is safe only if the user can make every listed payment, complete
the full request by its deadline, cover essential expenses, and maintain their preferred
minimum balance"; "A plan is safe only if the balance never falls below minimum_balance_to_keep".
"""
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly
from buyorwait.extraction.gather import EvidenceBundle
from buyorwait.forecast import is_safe, project_flows
from buyorwait.ledger import build_ledger, forecast_horizon_end
from buyorwait.models import PaymentOption
from buyorwait.pipeline import decide_request
from buyorwait.planning import decide

RD = date(2026, 6, 2)
END = forecast_horizon_end(RD)            # 2026-08-31 under the calendar reading
assert END == date(2026, 8, 31)


def _events():
    # rent 800 on the 1st, salary 1500 on the 15th: the balance is lowest just after each rent
    return monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")


def _opt(oid, first, n, amount, freq=31):
    """An option paying `amount` n times from `first`, every `freq` days (a day-count cadence)."""
    return PaymentOption(f"payment_option_{oid}", "r1", "installments", D(str(amount)), n, first, freq,
                         D("0"), D(str(amount)) * n)


def _decide(req, options, balance="2000", minimum="500", **profile):
    base = dict(current_available_balance=D(balance), minimum_balance_to_keep=D(minimum),
                max_installment_months=12, payment_methods=("installments",))
    base.update(profile)
    ds = mk_dataset(mk_profile(**base), _events(), req, options=list(options))
    return decide_request(ds, req, EvidenceBundle([], []))


def _ledger(balance="2000", minimum="500", **kw):
    ds = mk_dataset(mk_profile(current_available_balance=D(balance), minimum_balance_to_keep=D(minimum)),
                    _events(), mk_request(1, rd=RD))
    return build_ledger(ds, "u1", RD, [], **kw)


# ---------------------------------------------------------------------------------------
# forecast.is_safe
# ---------------------------------------------------------------------------------------

def test_is_safe_fails_instead_of_ignoring_a_payment_after_the_window():
    L = _ledger()
    flows = project_flows(L)
    assert is_safe(L, flows, [(END, D("1"))])
    assert not is_safe(L, flows, [(END + timedelta(days=1), D("1"))])


def test_extended_ledger_projects_the_same_series_further():
    L, Lx = _ledger(), _ledger(horizon_end=date(2026, 10, 15))
    assert Lx.horizon_end == date(2026, 10, 15) and L.horizon_end == END
    rent = lambda LL: [f.on for f in project_flows(LL) if f.category == "rent"]
    assert rent(L) == [date(2026, 7, 1), date(2026, 8, 1)]
    assert rent(Lx) == [date(2026, 7, 1), date(2026, 8, 1), date(2026, 9, 1), date(2026, 10, 1)]
    assert [f.on for f in Lx.salary_flows][-1] == date(2026, 10, 15)


def test_extension_never_shrinks_the_window():
    assert _ledger(horizon_end=date(2026, 7, 1)).horizon_end == END


# ---------------------------------------------------------------------------------------
# decide(): the eight required scenarios
# ---------------------------------------------------------------------------------------

def test_1_all_payments_inside_the_window_behave_as_before():
    opt = _opt(1, date(2026, 6, 5), 3, 300, freq=30)      # 06-05, 07-05, 08-04
    dec = _decide(mk_request(900, partial=False, deadline=date(2026, 8, 10)), [opt])
    assert dec.method == "installments" and dec.plan.option is opt and dec.plan.changes == []


def test_2_last_installment_exactly_on_the_horizon_end_is_checked_on_the_nominal_ledger():
    first = END - timedelta(days=60)
    ok = _opt(1, first, 3, 300, freq=30)                  # last leg exactly on END (08-31)
    assert ok.schedule()[-1][0] == END
    dec = _decide(mk_request(900, partial=False, deadline=END), [ok])
    assert dec.method == "installments"
    # same dates, but the 08-31 leg drains the balance: opening 2000 +1500 -800 +1500 -800 +1500 = 4900
    # before the legs; three legs of 1200 leave 1300 on 08-31 ... so use 1500: 4900 - 4500 = 400 < 500
    too_big = _opt(2, first, 3, 1500, freq=30)
    dec = _decide(mk_request(4500, partial=False, deadline=END), [too_big])
    assert dec.method == "not_recommended"
    assert any(r.startswith("payment_option_2:") and "unsafe" in r for r in dec.rejected)


def test_3_one_installment_just_after_the_window_that_breaks_the_minimum_is_rejected():
    # 09-01: rent 800 is due and no salary until 09-15. Balance path with 2000 opening:
    #   06-15 +1500 -> 3500 ; 07-01 -800 ; 07-15 +1500 ; 08-01 -800 ; 08-15 +1500 -> 4900 ; 09-01 -800 -> 4100
    # a 09-01 leg of 3700 leaves 400 < 500 minimum -> unsafe; the two earlier legs alone are fine.
    opt = _opt(1, date(2026, 7, 1), 3, 3700, freq=31)   # 07-01, 08-01, 09-01
    assert opt.schedule()[-1][0] == date(2026, 9, 1) > END
    dec = _decide(mk_request(11100, partial=False, deadline=date(2026, 9, 15)), [opt], balance="10000")
    # with 10000 opening: 06-15 +1500 -> 11500; 07-01 -800-3700 -> 7000; 07-15 -> 8500; 08-01 -> 4000;
    # 08-15 -> 5500; 09-01 -800-3700 -> 1000 ... still >= 500, so raise the leg to 3900:
    #   07-01 -> 6800; 07-15 -> 8300; 08-01 -> 3600; 08-15 -> 5100; 09-01 -> 400 < 500 -> unsafe
    opt = _opt(1, date(2026, 7, 1), 3, 3900, freq=31)
    dec = _decide(mk_request(11700, partial=False, deadline=date(2026, 9, 15)), [opt], balance="10000")
    assert dec.method == "not_recommended", dec.rejected
    assert any("unsafe through its last payment (2026-09-01)" in r for r in dec.rejected)


def test_4_one_installment_after_the_window_that_is_safe_keeps_the_plan_eligible():
    opt = _opt(1, date(2026, 7, 1), 3, 300, freq=31)    # 07-01, 08-01, 09-01
    dec = _decide(mk_request(900, partial=False, deadline=date(2026, 9, 15)), [opt])
    assert dec.method == "installments" and dec.plan.option is opt
    assert dec.plan.payments == opt.schedule()            # every leg rendered
    assert dec.ledger.horizon_end == END                   # the decision's ledger is still the nominal one


def test_5_every_later_installment_is_checked_not_just_the_first_one_outside():
    # legs 09-01, 10-01, 11-01 all after the window; only the 11-01 leg is fatal
    safe = _opt(1, date(2026, 9, 1), 3, 500, freq=30)   # 09-01, 10-01, 10-31: all after END
    assert safe.schedule()[0][0] > END
    dec = _decide(mk_request(1500, partial=False, deadline=date(2026, 11, 30)), [safe])
    assert dec.method == "installments"
    # opening 2000: ... 08-15 -> 4900; 09-01 -800 -> 4100; 09-15 -> 5600; 10-01 -800 -> 4800;
    # 10-15 -> 6300; 10-31 (before the 11-01 rent). Legs of 2000: 09-01 -> 2100; 09-15 -> 3600;
    # 10-01 -> 800; 10-15 -> 2300; 10-31 -> 300 < 500: only the THIRD leg is fatal.
    fatal = _opt(2, date(2026, 9, 1), 3, 2000, freq=30)
    dec = _decide(mk_request(6000, partial=False, deadline=date(2026, 11, 30)), [fatal])
    assert dec.method == "not_recommended", dec.rejected
    assert any("unsafe through its last payment (2026-10-31)" in r for r in dec.rejected)


def test_6_late_versus_deadline_the_deadline_gate_still_wins():
    opt = _opt(1, date(2026, 7, 1), 3, 300, freq=31)    # safe beyond the window ...
    dec = _decide(mk_request(900, partial=False, deadline=date(2026, 8, 31)), [opt])  # ... but 09-01 misses the deadline
    assert dec.method == "not_recommended"
    assert any("after desired_completion_date" in r for r in dec.rejected)
    assert not any("cannot be verified" in r or "unsafe" in r for r in dec.rejected)


def test_7_partial_second_leg_on_the_boundary_is_checked_and_never_beyond_it():
    # earliest_date_for_full_payment is by construction inside the window, so the second leg is
    # at most END; the check must include it.
    req = mk_request(4000, partial=True, deadline=END)
    dec = _decide(req, [], payment_methods=("full_payment", "partial_payment"), max_installment_months=None)
    if dec.method == "partial_payment":
        assert dec.plan.payments[1][0] <= END
        assert is_safe(dec.ledger, project_flows(dec.ledger), dec.plan.payments)
    assert dec.earliest_full is None or dec.earliest_full <= END


def test_8_empty_and_malformed_schedules_are_still_rejected_safely():
    empty = PaymentOption("payment_option_1", "r1", "installments", D("100"), 0, RD, 30, D("0"), D("0"))
    dec = _decide(mk_request(300, partial=False, deadline=date(2026, 12, 31)), [empty])
    assert dec.method == "not_recommended" and any("empty payment schedule" in r for r in dec.rejected)
    zero_amount = PaymentOption("payment_option_2", "r1", "installments", D("0"), 3, date(2026, 9, 5), 30, D("0"), D("0"))
    dec = _decide(mk_request(300, partial=False, deadline=date(2026, 12, 31)), [zero_amount])
    assert dec.method == "not_recommended" and any("non-positive" in r for r in dec.rejected)


# ---------------------------------------------------------------------------------------
# without the factory, a long schedule is unverifiable and therefore rejected
# ---------------------------------------------------------------------------------------

def test_decide_without_a_ledger_factory_rejects_rather_than_assumes():
    ds = mk_dataset(mk_profile(current_available_balance=D("2000"), max_installment_months=12,
                               payment_methods=("installments",)),
                    _events(), mk_request(900, partial=False, deadline=date(2026, 9, 15)),
                    options=[_opt(1, date(2026, 7, 1), 3, 300, freq=31)])
    L = build_ledger(ds, "u1", RD, [])
    dec = decide(ds.requests[0], L, ds.options_by_request["r1"])
    assert dec.method == "not_recommended"
    assert any("cannot be verified" in r for r in dec.rejected)
