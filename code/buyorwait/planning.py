"""Candidate payment plans, feasibility, and the exact challenge ranking."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .forecast import SpendingChange, amount_safe_to_pay, earliest_full_payment_date, is_safe, project_flows, simulate
from .ledger import Ledger
from .models import PaymentOption, Request
from .money import ZERO, q2
from .spending import select_changes


@dataclass
class Plan:
    method: str                                  # full_payment | partial_payment | installments | wait
    payments: List[Tuple[date, Decimal]]
    changes: List[SpendingChange] = field(default_factory=list)
    option: Optional[PaymentOption] = None
    total_paid: Decimal = ZERO
    completes_by_deadline: bool = True

    @property
    def first_date(self) -> date:
        return self.payments[0][0]

    @property
    def last_date(self) -> date:
        return self.payments[-1][0]

    def rank_key(self):
        # 1. by deadline  2. no changes  3. min total  4. earlier start  5. fewer payments  6. option id
        # (1. is always True for a candidate: completes_by_deadline() rejects late schedules before
        # ranking; the term is kept so the key still mirrors the statement's list.)
        return (
            0 if self.completes_by_deadline else 1,
            1 if self.changes else 0,
            self.total_paid,
            self.first_date,
            len(self.payments),
            self.option.option_index if self.option else 0,
        )


@dataclass
class Decision:
    request: Request
    ledger: Ledger
    amount_safe: Decimal
    earliest_full: Optional[date]
    status: str
    method: str
    plan: Optional[Plan]
    candidates: List[Plan]
    rejected: List[str]
    min_projected_balance: Decimal


def completes_by_deadline(payments: Sequence[Tuple[date, Decimal]], deadline: date) -> Tuple[bool, str]:
    """Hard eligibility gate, not a ranking term.

    Statement: "A recommendation is safe only if the user can make every listed payment,
    complete the full request by its deadline, ..." and "The plan must complete the request by
    desired_completion_date". A schedule with no payments, a non-positive amount, or any
    payment dated after the deadline therefore never becomes a candidate - even when it would
    be the only one. The latest payment is taken over the whole schedule, so an unsorted
    schedule cannot slip through on its last element.
    """
    if not payments:
        return False, "empty payment schedule"
    if any(a <= ZERO for _, a in payments):
        return False, "non-positive payment amount"
    last = max(d for d, _ in payments)
    if last > deadline:
        return False, f"final payment {last.isoformat()} is after desired_completion_date {deadline.isoformat()}"
    return True, ""


def installment_eligible(opt: PaymentOption, profile) -> Tuple[bool, str]:
    if opt.payment_method != "installments":
        return False, "not an installment option"
    if "installments" not in profile.payment_methods:
        return False, "user will not consider installments"
    if profile.max_installment_months is None:
        return False, "max_installment_months blank"
    if opt.number_of_payments > profile.max_installment_months:
        return False, f"{opt.number_of_payments} payments exceed max_installment_months={profile.max_installment_months}"
    return True, ""


def decide(req: Request, L: Ledger, options: Sequence[PaymentOption],
           ledger_to: Optional[Callable[[date], Ledger]] = None) -> Decision:
    """Rank every eligible, safe plan for ``req`` on ledger ``L``.

    ``ledger_to(end)`` rebuilds the same ledger projected to ``end``. It is used only to
    validate an installment schedule whose last leg falls after ``L.horizon_end``: every
    listed payment must be checked against the forecast that reaches it (statement: "the
    user can make every listed payment ... and maintain their preferred minimum balance").
    Without the factory such a schedule cannot be verified and is rejected, never assumed safe.
    amount_safe_to_pay, earliest_date_for_full_payment, wait and partial plans stay on the
    nominal horizon.
    """
    p = L.profile
    base_flows = project_flows(L)
    extended: Dict[date, Tuple[Ledger, list]] = {}

    def ledger_for(payments) -> Optional[Tuple[Ledger, list]]:
        """(ledger, flows) able to verify every payment, or None when nothing can."""
        last = max(d for d, _ in payments)
        if last <= L.horizon_end:
            return L, base_flows
        if ledger_to is None:
            return None
        if last not in extended:
            Lx = ledger_to(last)
            extended[last] = (Lx, project_flows(Lx))
        return extended[last]
    safe = amount_safe_to_pay(L, base_flows, req.requested_amount)
    earliest = earliest_full_payment_date(L, base_flows, req.requested_amount)
    min_bal = simulate(L.opening_balance, base_flows).minimum
    rd, deadline, amt = req.request_date, req.desired_completion_date, req.requested_amount
    cands: List[Plan] = []
    rejected: List[str] = []
    accepts_full = "full_payment" in p.payment_methods

    # full payment today
    if accepts_full:
        pays = [(rd, amt)]
        if is_safe(L, base_flows, pays):
            cands.append(Plan("full_payment", pays, [], None, amt, True))
        else:
            chs = select_changes(L, pays)
            if chs:
                cands.append(Plan("full_payment", pays, chs, None, amt, True))
            else:
                rejected.append("full_payment today: unsafe even with permitted spending changes")
    else:
        rejected.append("full_payment: not accepted by user")

    # wait for the earliest safe full-payment date
    if accepts_full and earliest is not None and earliest > rd:
        ok, why = completes_by_deadline([(earliest, amt)], deadline)
        if ok:
            cands.append(Plan("wait", [(earliest, amt)], [], None, amt, True))
        else:
            rejected.append(f"wait: {why}")

    # partial: safe today + remainder on earliest date (spec-mandated shape)
    if req.allows_partial_payment and "partial_payment" in p.payment_methods:
        if ZERO < safe < amt and earliest is not None and earliest <= deadline:
            pays = [(rd, safe), (earliest, q2(amt - safe))]
            if is_safe(L, base_flows, pays):
                cands.append(Plan("partial_payment", pays, [], None, amt, True))
            else:
                rejected.append("partial_payment: combined schedule unsafe")
        else:
            rejected.append("partial_payment: conditions not met (0<safe<requested and earliest<=deadline)")
    elif req.allows_partial_payment:
        rejected.append("partial_payment: not accepted by user")

    # installment options exactly as supplied
    for opt in sorted(options, key=lambda o: o.option_index):
        ok, why = installment_eligible(opt, p)
        if not ok:
            if opt.payment_method == "installments":
                rejected.append(f"{opt.payment_option_id}: {why}")
            continue
        sched = opt.schedule()
        ok, why = completes_by_deadline(sched, deadline)
        if not ok:
            rejected.append(f"{opt.payment_option_id}: {why}")
            continue
        scope = ledger_for(sched)
        if scope is None:
            rejected.append(f"{opt.payment_option_id}: final payment {sched[-1][0].isoformat()} is after the "
                            f"forecast window {L.horizon_end.isoformat()} and cannot be verified")
            continue
        Ls, flows_s = scope
        if is_safe(Ls, flows_s, sched):
            cands.append(Plan("installments", sched, [], opt, opt.total_payable_amount, True))
        else:
            chs = select_changes(Ls, sched)
            if chs:
                cands.append(Plan("installments", sched, chs, opt, opt.total_payable_amount, True))
            else:
                rejected.append(f"{opt.payment_option_id}: unsafe through its last payment "
                                f"({sched[-1][0].isoformat()}) even with spending changes")

    cands.sort(key=lambda c: c.rank_key())
    best = cands[0] if cands else None
    if best is None:
        status, method = "not_affordable", "not_recommended"
    elif best.method == "full_payment" and not best.changes:
        status, method = "affordable_now", "full_payment"
    elif best.method == "wait":
        status, method = "affordable_later", "wait"
    else:
        status, method = "affordable_with_plan", best.method
    return Decision(req, L, safe, earliest, status, method, best, cands, rejected, min_bal)
