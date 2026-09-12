"""Scheduled credits and the recurring-salary anchor (ledger._project_salary).

Statement: "Count confirmed salary on its settlement date. Do not count pending credits,
bonuses, commissions, refunds, ... until they settle. Do not invent unsupported future
income." A scheduled credit filed under category="salary" is payroll only when its
description says so (classify.income_class); anything else neither anchors the recurring
series nor is counted before it settles.
"""
from datetime import date
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request
from buyorwait.evidence import Evidence
from buyorwait.ledger import build_ledger

RD = date(2026, 6, 2)
HISTORY = [date(2026, 2, 15), date(2026, 3, 15), date(2026, 4, 15), date(2026, 5, 15)]
NORMAL = [(date(2026, 6, 15), D("1500")), (date(2026, 7, 15), D("1500")), (date(2026, 8, 15), D("1500"))]


def _history(amount=1500):
    return [mk_event(f"sal{i}", "income", "salary", "credit", amount, d, desc="Payroll credit") for i, d in enumerate(HISTORY)]


def _sched(eid, amount, on, desc, category="salary", etype="income"):
    return mk_event(eid, etype, category, "credit", amount, on, status="scheduled", desc=desc)


def _ledger(events, evidence=(), rd=RD):
    ds = mk_dataset(mk_profile(), events, mk_request(1, rd=rd))
    return build_ledger(ds, "u1", rd, list(evidence))


def _salary(L):
    return [(f.on, f.amount) for f in sorted(L.salary_flows, key=lambda f: f.on)]


def test_scheduled_annual_bonus_beside_normal_salary():
    L = _ledger(_history() + [_sched("bon", 9000, date(2026, 6, 20), "Annual performance bonus")])
    assert _salary(L) == NORMAL
    assert not any(f.source_event_id == "bon" for f in L.salary_flows + L.known_flows)
    assert any("bon:" in a and "not counted" in a for a in L.audit)


def test_scheduled_commission_beside_normal_salary():
    # dated on the payday and larger than salary: the old anchor rule would have made 4000 recur
    L = _ledger(_history() + [_sched("com", 4000, date(2026, 6, 15), "Quarterly sales commission")])
    assert _salary(L) == NORMAL
    assert not any(f.amount == D("4000") for f in L.salary_flows + L.known_flows)


@pytest.mark.parametrize("desc", ["Lottery prize payout", "Merchant refund", "Platform payout",
                                  "Expense reimbursement", "Freelance milestone payment", "Referral bonus"])
def test_other_scheduled_non_payroll_credits_never_anchor_or_count(desc):
    L = _ledger(_history() + [_sched("x", 7777, date(2026, 6, 30), desc)])
    assert _salary(L) == NORMAL
    assert not any(f.source_event_id == "x" for f in L.salary_flows + L.known_flows)


def test_scheduled_legitimate_payroll_anchors_amount_and_payday():
    # "Next confirmed salary" of 1600 on the 20th: the freshest employer fact re-anchors the series
    L = _ledger(_history() + [_sched("next", 1600, date(2026, 6, 20), "Next confirmed salary")])
    assert _salary(L) == [(date(2026, 6, 20), D("1600")), (date(2026, 7, 20), D("1600")), (date(2026, 8, 20), D("1600"))]
    assert [f.source_event_id for f in sorted(L.salary_flows, key=lambda f: f.on)][0] == "next"


def test_scheduled_payroll_under_another_category_still_counts():
    L = _ledger(_history() + [_sched("next", 1500, date(2026, 6, 15), "Payroll credit", category="income")])
    assert _salary(L) == NORMAL and L.salary_flows[0].source_event_id == "next"


def test_mixed_scheduled_credits_only_the_payroll_row_anchors():
    ev = _history() + [
        _sched("bon", 9000, date(2026, 6, 25), "Annual performance bonus"),
        _sched("next", 1550, date(2026, 6, 15), "Next confirmed salary"),
        _sched("com", 3000, date(2026, 7, 1), "Sales commission"),
        _sched("ref", 120, date(2026, 6, 10), "Merchant refund", category="shopping", etype="refund"),
    ]
    L = _ledger(ev)
    assert _salary(L) == [(date(2026, 6, 15), D("1550")), (date(2026, 7, 15), D("1550")), (date(2026, 8, 15), D("1550"))]
    assert {f.source_event_id for f in L.salary_flows + L.known_flows} <= {"next", None}


def test_income_ended_message_stops_projection_even_with_a_scheduled_bonus():
    ev = _history() + [_sched("bon", 9000, date(2026, 6, 25), "Annual performance bonus")]
    ended = Evidence("message", "m1", "u1", "income_ended", sent_at="2026-05-30T09:00:00")
    L = _ledger(ev, [ended])
    assert L.salary_flows == [] and not any(f.source_event_id == "bon" for f in L.known_flows)


def test_scheduled_final_payroll_is_counted_once_and_nothing_recurs_after_it():
    L = _ledger(_history() + [_sched("fin", 1500, date(2026, 6, 15), "Final employer payroll")])
    assert _salary(L) == [(date(2026, 6, 15), D("1500"))]
    assert L.salary_flows[0].source_event_id == "fin"


def test_scheduled_bonus_alone_does_not_invent_a_salary_series():
    L = _ledger([_sched("bon", 9000, date(2026, 6, 25), "Annual performance bonus")])
    assert L.salary_flows == [] and L.known_flows == []
    assert any("no recurring payroll history" in a for a in L.audit)
