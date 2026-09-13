"""Linked lifecycle pairs in settled recurrence history (H2).

A settled child that carries linked_event_id is the realized cash record of its lifecycle (the
purchase an authorization became, the retry a failed attempt became). It is settled cash history
like any other row; its parent is dropped whenever a settled child exists, so the pair is counted
exactly once. A parent whose children are all pending/scheduled/cancelled keeps its own cash
state, and a child whose parent is absent from the data is ordinary settled history.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal as D

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly
from buyorwait.ledger import build_ledger

RD = date(2026, 6, 2)


def history(extra):
    ev = monthly("sal", "salary", 2000, 15, 5, etype="income", desc="Payroll credit") + list(extra)
    ds = mk_dataset(mk_profile(), ev, mk_request(100, rd=RD))
    L = build_ledger(ds, "u1", RD, [])
    return L, {s.category: s for s in L.series}


def dining(n, last=RD - timedelta(days=2), gap=10, amount=50, desc="Dining out"):
    return [mk_event(f"d{k}", "expense", "dining", "debit", amount, last - timedelta(days=gap * (n - 1 - k)), desc=desc) for k in range(n)]


def test_settled_parent_with_settled_child_counts_once_through_the_child():
    rows = dining(3) + [mk_event("parent", "expense", "dining", "debit", 50, RD - timedelta(days=33), desc="Card charge"),
                        mk_event("child", "expense", "dining", "debit", 50, RD - timedelta(days=32), desc="Card purchase", linked="parent")]
    L, s = history(rows)
    assert s["dining"].occurrences == 4 and s["dining"].amount == D("50")     # three plain rows + the child, not five


def test_cancelled_parent_with_settled_child_uses_the_child():
    rows = dining(3) + [mk_event("parent", "expense", "dining", "debit", 50, RD - timedelta(days=33), status="cancelled", desc="Card hold"),
                        mk_event("child", "expense", "dining", "debit", 50, RD - timedelta(days=32), desc="Card purchase", linked="parent")]
    L, s = history(rows)
    assert s["dining"].occurrences == 4


def test_pending_parent_with_settled_child_uses_the_child_as_history():
    rows = dining(3) + [mk_event("parent", "expense", "dining", "debit", 50, RD - timedelta(days=33), status="pending", settle=RD + timedelta(days=2), desc="Card hold"),
                        mk_event("child", "expense", "dining", "debit", 50, RD - timedelta(days=32), desc="Card purchase", linked="parent")]
    L, s = history(rows)
    assert s["dining"].occurrences == 4
    assert not any(f.source_event_id == "child" for f in L.known_flows)       # the child is history, never a reservation


def test_dangling_linked_child_is_ordinary_settled_history():
    rows = dining(3) + [mk_event("child", "expense", "dining", "debit", 50, RD - timedelta(days=32), desc="Card purchase", linked="event_not_in_data")]
    L, s = history(rows)
    assert s["dining"].occurrences == 4


def test_recurring_series_with_one_linked_occurrence_keeps_its_cadence_and_amount():
    rows = dining(4)
    rows[1] = mk_event("d1", "expense", "dining", "debit", 50, rows[1].event_date, desc="Card purchase", linked="auth_x")
    rows.append(mk_event("auth_x", "expense", "dining", "debit", 50, rows[1].event_date - timedelta(days=1), status="cancelled", desc="Card hold"))
    L, s = history(rows)
    assert s["dining"].occurrences == 4 and s["dining"].period_days == 10 and s["dining"].amount == D("50")
    L0, s0 = history(dining(4))
    assert s["dining"].amount == s0["dining"].amount and s["dining"].occurrences == s0["dining"].occurrences


def test_lifecycle_pair_is_never_double_counted():
    # parent and child on the same day with the same amount: only the child enters history,
    # so the cadence is the plain 10-day one and the amount is unchanged
    rows = dining(3)
    rows += [mk_event("parent", "expense", "dining", "debit", 50, RD - timedelta(days=32), desc="Card charge"),
             mk_event("child", "expense", "dining", "debit", 50, RD - timedelta(days=32), desc="Card purchase", linked="parent")]
    L, s = history(rows)
    assert s["dining"].occurrences == 4 and s["dining"].period_days == 10
    ids = {f.source_event_id for f in L.known_flows}
    assert "parent" not in ids and "child" not in ids


def test_parent_with_only_a_pending_child_keeps_its_own_history_row():
    # a settled original charge whose duplicate is still pending is real spending; the pending
    # duplicate is reserved by its own cash state (Target #3), not counted as history
    rows = dining(3) + [mk_event("orig", "expense", "dining", "debit", 50, RD - timedelta(days=32), desc="Restaurant charge"),
                        mk_event("dup", "expense", "dining", "debit", 50, RD - timedelta(days=1), status="pending", settle=RD + timedelta(days=2), desc="Restaurant charge", linked="orig")]
    L, s = history(rows)
    assert s["dining"].occurrences == 4
    assert any(f.source_event_id == "dup" for f in L.known_flows)
