"""Recurring salary paydays must be generated from the unclamped anchor (ledger.payday_in_month).

Chaining ``add_months`` from an already-clamped date turns a 31st payday into the 28th for good
after the first February; every date below is derived from ``(anchor month, pay_day)`` instead.
"""
import calendar
from datetime import date
from decimal import Decimal as D

import pytest
from hypothesis import given, strategies as st

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request
from buyorwait.evidence import Evidence
from buyorwait.ledger import build_ledger, payday_in_month


def _payroll(dates, amount=1500, prefix="sal"):
    return [mk_event(f"{prefix}{i}", "income", "salary", "credit", amount, d, desc="Payroll credit")
            for i, d in enumerate(dates)]


def _paydays(events, rd, evidence=()):
    ds = mk_dataset(mk_profile(), events, mk_request(1, rd=rd))
    L = build_ledger(ds, "u1", rd, list(evidence))
    return [f.on for f in L.salary_flows]


# ---------------------------------------------------------------------------------------
# the helper itself
# ---------------------------------------------------------------------------------------

@given(st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 1)),
       st.integers(min_value=1, max_value=31), st.integers(min_value=0, max_value=36))
def test_payday_in_month_clamps_only_to_that_month(anchor, pay_day, k):
    d = payday_in_month(anchor, pay_day, k)
    assert d.day == min(pay_day, calendar.monthrange(d.year, d.month)[1])
    assert (d.year * 12 + d.month) - (anchor.year * 12 + anchor.month) == k


def test_payday_in_month_does_not_inherit_a_clamp():
    anchor = date(2026, 1, 31)
    assert [payday_in_month(anchor, 31, k) for k in range(1, 5)] == [
        date(2026, 2, 28), date(2026, 3, 31), date(2026, 4, 30), date(2026, 5, 31)]


# ---------------------------------------------------------------------------------------
# projected salary in a ledger
# ---------------------------------------------------------------------------------------

def test_payday_31_recovers_after_february():
    # bank history: Nov 30 (clamped), Dec 31, Jan 31 -> last settled payday is the 31st
    ev = _payroll([date(2025, 11, 30), date(2025, 12, 31), date(2026, 1, 31)])
    assert _paydays(ev, date(2026, 2, 10)) == [date(2026, 2, 28), date(2026, 3, 31), date(2026, 4, 30)]


def test_payday_31_in_a_leap_year_february():
    ev = _payroll([date(2027, 11, 30), date(2027, 12, 31), date(2028, 1, 31)])
    assert _paydays(ev, date(2028, 2, 5)) == [date(2028, 2, 29), date(2028, 3, 31), date(2028, 4, 30)]


def test_payday_30_recovers_after_february():
    ev = _payroll([date(2025, 11, 30), date(2025, 12, 30), date(2026, 1, 30)])
    assert _paydays(ev, date(2026, 2, 5)) == [date(2026, 2, 28), date(2026, 3, 30), date(2026, 4, 30)]


@pytest.mark.parametrize("pay_day", [28, 29, 30, 31])
def test_ordinary_month_lengths(pay_day):
    # anchor in March (31 days); horizon from an April request runs to June 30
    ev = _payroll([date(2026, 1, min(pay_day, 31)), date(2026, 2, min(pay_day, 28)), date(2026, 3, pay_day)])
    expected = [date(2026, 4, min(pay_day, 30)), date(2026, 5, pay_day), date(2026, 6, min(pay_day, 30))]
    assert _paydays(ev, date(2026, 4, 1)) == expected


def test_payday_on_the_request_date_is_not_projected_twice():
    # the last settled payroll landed today: the next one is a month away, not today again
    ev = _payroll([date(2026, 3, 31), date(2026, 4, 30), date(2026, 5, 31)])
    assert _paydays(ev, date(2026, 5, 31)) == [date(2026, 6, 30), date(2026, 7, 31)]


def test_payday_on_the_horizon_end_is_included_and_one_day_later_is_not():
    # request 2026-02-10 -> horizon end 2026-04-30
    ev31 = _payroll([date(2025, 12, 31), date(2026, 1, 31)])
    assert _paydays(ev31, date(2026, 2, 10))[-1] == date(2026, 4, 30)
    ev1 = _payroll([date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1)])
    assert _paydays(ev1, date(2026, 2, 10)) == [date(2026, 3, 1), date(2026, 4, 1)]  # 05-01 is outside


def test_salary_first_message_anchor_on_the_31st_does_not_drift():
    # a new job: "first salary on Jan 31" then monthly; the request comes after that first payday
    msg = Evidence("message", "m1", "u1", "salary_first", amount=D("2000"), currency="EUR",
                   effective_date=date(2026, 1, 31), sent_at="2026-01-10T09:00:00")
    assert _paydays([], date(2026, 2, 15), [msg]) == [date(2026, 2, 28), date(2026, 3, 31), date(2026, 4, 30)]


def test_salary_first_on_or_after_the_request_date_starts_on_that_day():
    msg = Evidence("message", "m1", "u1", "salary_first", amount=D("2000"), currency="EUR",
                   effective_date=date(2026, 2, 28), sent_at="2026-02-01T09:00:00")
    assert _paydays([], date(2026, 2, 28), [msg])[0] == date(2026, 2, 28)
    assert _paydays([], date(2026, 2, 27), [msg])[0] == date(2026, 2, 28)


def test_salary_date_change_moves_the_payday_from_its_effective_date():
    ev = _payroll([date(2025, 12, 15), date(2026, 1, 15), date(2026, 2, 15)])
    msg = Evidence("message", "m1", "u1", "salary_date_change", effective_date=date(2026, 3, 31),
                   sent_at="2026-02-20T09:00:00")
    assert _paydays(ev, date(2026, 3, 1), [msg]) == [date(2026, 3, 31), date(2026, 4, 30), date(2026, 5, 31)]


def test_income_ended_and_terminal_payroll_project_nothing():
    ev = _payroll([date(2025, 12, 31), date(2026, 1, 31)])
    ended = Evidence("message", "m1", "u1", "income_ended", sent_at="2026-02-01T09:00:00")
    assert _paydays(ev, date(2026, 2, 10), [ended]) == []
    final = ev + [mk_event("fin", "income", "salary", "credit", 1500, date(2026, 2, 28), desc="Final employer payroll")]
    assert _paydays(final, date(2026, 3, 1)) == []
