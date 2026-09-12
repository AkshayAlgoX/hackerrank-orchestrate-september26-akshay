"""linked_event_id lifecycle rows: what is reserved, what is pooled, and why.

Statement: "When linked_event_id is present, it points to an earlier event in the same
transaction or investment lifecycle" and (AGENTS.md §6.1) "the link alone does not determine
whether a row counts toward cash flow". Cash state decides.
"""
import os
from datetime import date
from decimal import Decimal as D

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly
from buyorwait.classify import is_lifecycle_one_off
from buyorwait.evidence import Evidence
from buyorwait.ledger import build_ledger
from buyorwait.loaders import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RD = date(2026, 6, 2)


def test_disputed_duplicate_pending_charge_stays_reserved_until_a_reversal_settles():
    """Evidence classes: A - "Reserve pending debits", refunds/pending credits are not counted
    until they settle, and rule 4 (the financially safer interpretation). The competing
    reading ("Ignore ... duplicate records" covers a bank's duplicate charge) is class C: no
    solved sample contains the pattern, and the supplied bank message states the dispute is
    open and no reversal has been posted, i.e. the cash state is still pending."""
    events = monthly("rent", "rent", 800, 1, 5) + monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    events.append(mk_event("orig", "expense", "shopping", "debit", 134.75, date(2026, 5, 22),
                           settle=date(2026, 5, 23), desc="Original card charge"))
    events.append(mk_event("dup", "expense", "shopping", "debit", 134.75, date(2026, 6, 1), status="pending",
                           settle=date(2026, 6, 5), linked="orig", desc="Possible duplicate card charge"))
    ev = Evidence("message", "m1", "u1", "duplicate_charge_disputed", related_event_id="dup",
                  sent_at="2026-05-31T09:30:00")
    ds = mk_dataset(mk_profile(), events, mk_request(500, rd=RD))
    L = build_ledger(ds, "u1", RD, [ev])
    reserved = [(f.on, f.amount, f.source_event_id) for f in L.known_flows]
    assert (date(2026, 6, 5), D("-134.75"), "dup") in reserved
    # the settled original is history (already inside current_available_balance): it is neither
    # reserved again nor projected as a recurring series, so nothing is double-counted forward
    assert not any(f.source_event_id == "orig" for f in L.known_flows)
    assert not any(s.latest_event_id in ("orig", "dup") for s in L.series)


def test_settled_lifecycle_children_and_parents_never_seed_a_recurring_series():
    """A settled purchase linked to a cancelled authorization, and the parents of refund /
    reversal / duplicate rows, are one transaction each - not a spending pattern."""
    events = monthly("sal", "salary", 1500, 15, 5, etype="income", desc="Payroll credit")
    for i, day in enumerate((3, 10, 17, 24)):
        events.append(mk_event(f"auth{i}", "expense", "shopping", "debit", 50, date(2026, 5, day), status="cancelled",
                               desc="Card authorization"))
        events.append(mk_event(f"buy{i}", "expense", "shopping", "debit", 50, date(2026, 5, day + 2), linked=f"auth{i}",
                               desc="Settled card purchase"))
    ds = mk_dataset(mk_profile(), events, mk_request(500, rd=RD))
    L = build_ledger(ds, "u1", RD, [])
    assert not any(s.category == "shopping" for s in L.series)


def test_dataset_linked_settled_debits_are_all_lifecycle_one_offs_by_description():
    """Documents why the linked_event_id filter in build_ledger is currently a no-op on the
    participant data: every settled debit it removes from the recurrence pool is already
    excluded by is_lifecycle_one_off. If this ever fails, the latent case described in the
    T5 hardening report (a settled retry of a recurring bill carrying a link) has appeared and
    the filter's semantics must be decided explicitly rather than left implicit."""
    ds = load_dataset(os.path.join(ROOT, "dataset"), "requests.csv")
    parents = {e.linked_event_id for e in ds.events if e.linked_event_id}
    pooled_types = ("expense", "subscription", "debt_payment")
    offenders = [e.event_id for e in ds.events
                 if e.status == "settled" and e.is_debit and e.event_type in pooled_types
                 and (e.linked_event_id or e.event_id in parents) and not is_lifecycle_one_off(e.description)]
    assert offenders == []
