"""Grounded explanations rendered from the deterministic Decision (no model in the loop)."""
from __future__ import annotations

from datetime import date

from .money import fmt_human
from .planning import Decision


def _d(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def explain(dec: Decision) -> str:
    cur = dec.ledger.profile.home_currency
    req = dec.request
    amt = f"{cur} {fmt_human(req.requested_amount)}"
    minimum = f"{cur} {fmt_human(dec.ledger.minimum_balance)}"
    plan = dec.plan
    if dec.status == "affordable_now":
        return f"Pay {amt} today. This leaves at least {minimum} available over the next 90 days."
    if dec.method == "partial_payment":
        first, second = plan.payments
        return (f"Pay {cur} {fmt_human(first[1])} today and the remaining {cur} {fmt_human(second[1])} on {_d(second[0])}. "
                f"This completes the full request and keeps the {minimum} minimum protected.")
    if dec.method == "installments":
        opt = plan.option
        prefix = ""
        if plan.changes:
            prefix = _changes_clause(dec) + ", then use"
        else:
            prefix = "Use"
        return (f"{prefix} {opt.number_of_payments} installments of {cur} {fmt_human(opt.payment_amount)}, "
                f"starting {_d(opt.first_payment_date)}. This leaves at least {minimum} available.")
    if dec.method == "full_payment":  # with spending changes
        return f"{_changes_clause(dec)}, then pay {amt} today. This leaves at least {minimum} available."
    if dec.method == "wait":
        d = plan.payments[0][0]
        return f"Pay {amt} in full on {_d(d)}. Paying earlier would take the balance below the {minimum} minimum."
    # not_affordable
    if dec.amount_safe > 0 and "full_payment" not in dec.ledger.profile.payment_methods and dec.earliest_full is None:
        return (f"Do not proceed with the {amt} request. Although {cur} {fmt_human(dec.amount_safe)} is available today, "
                f"the full amount cannot be completed safely within 90 days.")
    return (f"Do not make this payment by {_d(req.desired_completion_date)}. "
            f"None of the available options keeps the {minimum} minimum protected.")


def _changes_clause(dec: Decision) -> str:
    cur = dec.ledger.profile.home_currency
    parts = []
    for c in dec.plan.changes:
        name = c.description.lower()
        if c.action == "stop":
            parts.append(f"stop the {name}")
        else:
            parts.append(f"reduce the {name} to {cur} {fmt_human(c.new_amount)}")
    text = " and ".join(parts)
    return text[0].upper() + text[1:]
