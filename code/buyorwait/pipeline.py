"""End-to-end orchestration: dataset -> evidence -> ledger -> decision -> validated rows."""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .atomic import atomic_write
from .extraction import EvidenceBundle, gather_evidence
from .forecast import project_flows, simulate
from .ledger import Ledger, UnresolvedCashEvidence, build_ledger
from .models import Dataset, Request
from .output import OutputRow, render_row, validate_row
from .planning import Decision, decide


@dataclass
class RunResult:
    rows: List[OutputRow]
    decisions: Dict[str, Decision]
    proofs: Dict[str, Dict[str, Any]]
    violations: Dict[str, List[str]] = field(default_factory=dict)
    bundle: Optional[EvidenceBundle] = None
    # request_id -> "ExcType: message" for requests that fell back (see fallback_row); the
    # full traceback is kept in proofs[request_id]["error"]["traceback"].
    errors: Dict[str, str] = field(default_factory=dict)


FALLBACK_EXPLANATION = ("Do not proceed: this request could not be evaluated ({error}), so no "
                        "payment is recommended until it can be re-checked.")
UNRESOLVED_EXPLANATION = ("Do not proceed: the amount of a debit still due inside the forecast window could not "
                          "be established from the available evidence ({detail}), so nothing can be certified "
                          "safe and no payment is recommended until that amount is known.")


def fallback_row(req: Request, exc: BaseException) -> OutputRow:
    """Contract-valid, most conservative row for a request whose evaluation raised.

    Nothing is recommended and nothing is claimed safe: amount_safe_to_pay 0, not_affordable,
    not_recommended, no plan, no earliest date, no spending changes. That is the only
    combination that is valid under §6.2 without any knowledge of the user's finances, and a
    reviewer can see from the explanation that it is a fallback rather than a decision.
    """
    if isinstance(exc, UnresolvedCashEvidence):
        text = UNRESOLVED_EXPLANATION.format(detail="; ".join(exc.reasons))
    else:
        text = FALLBACK_EXPLANATION.format(error=type(exc).__name__)
    return OutputRow(request_id=req.request_id, amount_safe_to_pay="0",
                     affordability_status="not_affordable", recommended_payment_method="not_recommended",
                     payment_plan="none", earliest_date_for_full_payment="", spending_changes_needed="none",
                     decision_explanation=text)


def decide_request(ds: Dataset, req: Request, bundle: EvidenceBundle) -> Decision:
    evidence = bundle.for_user(req.user_id)
    L = build_ledger(ds, req.user_id, req.request_date, evidence)
    # Same reconstruction projected further, used only to verify an installment plan whose
    # last leg falls after the nominal forecast window.
    ledger_to = lambda end: build_ledger(ds, req.user_id, req.request_date, evidence, horizon_end=end)  # noqa: E731
    return decide(req, L, ds.options_by_request.get(req.request_id, []), ledger_to=ledger_to)


def bottleneck_of(L: Ledger) -> Dict[str, Any]:
    """The day that binds amount_safe_to_pay: first date on which the projected balance is lowest.

    Derived from the same base path the decision used (project_flows + simulate, no plan, no
    changes): ``bottleneck_balance - minimum_balance`` is the room that amount_safe_to_pay is
    capped by. ``bottleneck_event_id`` names the largest debit landing that day when it comes
    from a dataset row (pending/scheduled) or the latest occurrence of the recurring series that
    produced it; otherwise it is null. Audit only: no decision reads this.
    """
    flows = project_flows(L)
    path = simulate(L.opening_balance, flows)
    lowest = path.minimum
    on = L.request_date
    for d, bal in path.points:
        if bal == lowest and bal < L.opening_balance:
            on = d
            break
    else:
        if path.points and path.points[0][0] == L.request_date and path.points[0][1] == lowest:
            on = L.request_date
    same_day = sorted((f for f in flows if f.on == on), key=lambda f: (f.amount, f.label))
    debits = [f for f in same_day if f.amount < 0]
    event_id = None
    if debits:
        biggest = debits[0]
        if biggest.source_event_id:
            event_id = biggest.source_event_id
        elif biggest.series_id:
            try:
                event_id = L.series_by_id(biggest.series_id).latest_event_id
            except KeyError:
                event_id = None
    return {
        "bottleneck_date": on.isoformat(),
        "bottleneck_balance": str(lowest),
        "minimum_balance": str(L.minimum_balance),
        "headroom": str(lowest - L.minimum_balance),
        "bottleneck_event_id": event_id,
        "flows_on_bottleneck_date": [[str(f.amount), f.label, f.source_event_id, f.series_id] for f in same_day],
    }


def proof_of(dec: Decision, bundle: EvidenceBundle) -> Dict[str, Any]:
    """Structured, machine-checkable account of the decision (feeds explanations and audits)."""
    L = dec.ledger
    return {
        "request_id": dec.request.request_id,
        "user_id": dec.request.user_id,
        "home_currency": L.profile.home_currency,
        "opening_balance": str(L.opening_balance),
        "minimum_balance": str(L.minimum_balance),
        "horizon": [L.request_date.isoformat(), L.horizon_end.isoformat()],
        "min_projected_balance": str(dec.min_projected_balance),
        "amount_safe_to_pay": str(dec.amount_safe),
        "earliest_full_payment": dec.earliest_full.isoformat() if dec.earliest_full else None,
        "status": dec.status, "method": dec.method,
        "plan": [[d.isoformat(), str(a)] for d, a in dec.plan.payments] if dec.plan else [],
        "changes": [c.render() for c in dec.plan.changes] if dec.plan else [],
        "known_flows": [[f.on.isoformat(), str(f.amount), f.label] for f in sorted(L.known_flows, key=lambda f: f.on)],
        "salary_flows": [[f.on.isoformat(), str(f.amount), f.label] for f in L.salary_flows],
        "series": [dict(id=s.series_id, category=s.category, description=s.description, amount=str(s.amount),
                        period=s.period_days or "monthly", flexibility=s.flexibility, latest_event_id=s.latest_event_id,
                        history=s.occurrences) for s in L.series],
        "evidence": [e.to_json() for e in bundle.for_user(dec.request.user_id)],
        "candidates": [dict(method=c.method, total=str(c.total_paid), first=c.first_date.isoformat(), n=len(c.payments),
                            by_deadline=c.completes_by_deadline, changes=[x.render() for x in c.changes],
                            option=c.option.payment_option_id if c.option else None) for c in dec.candidates],
        "rejected": dec.rejected,
        "audit": L.audit,
        "bottleneck": bottleneck_of(L),
        "provenance": list(L.provenance),
        "evidence_sources": {e.source_id: bundle.sources.get(e.source_id, "unknown")
                             for e in bundle.for_user(dec.request.user_id)},
    }


def run(ds: Dataset, bundle: Optional[EvidenceBundle] = None, use_model: Optional[bool] = None,
        cache_path: Optional[str] = None) -> RunResult:
    if bundle is None:
        kwargs = {} if cache_path is None else {"cache_path": cache_path}
        bundle = gather_evidence(ds, use_model=use_model, **kwargs)
    rows: List[OutputRow] = []
    decisions: Dict[str, Decision] = {}
    proofs: Dict[str, Dict[str, Any]] = {}
    violations: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    for req in ds.requests:
        # One request must never abort the batch: the contract is exactly one row per
        # request_id, so an unexpected exception yields a conservative fallback row and
        # the remaining requests are still evaluated normally.
        try:
            dec = decide_request(ds, req, bundle)
            row = render_row(dec)
            proof = proof_of(dec, bundle)
        except Exception as exc:  # noqa: BLE001 - anything unexpected in one request
            dec = None
            row = fallback_row(req, exc)
            errors[req.request_id] = f"{type(exc).__name__}: {exc}"
            proof = {"request_id": req.request_id, "user_id": req.user_id, "fallback": True,
                     "error": {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}}
        errs = validate_row(dict(zip([c for c in row.__dataclass_fields__], row.as_list())), req,
                            ds.options_by_request.get(req.request_id, []))
        if errs:
            violations[req.request_id] = errs
        rows.append(row)
        if dec is not None:
            decisions[req.request_id] = dec
        proofs[req.request_id] = proof
    return RunResult(rows, decisions, proofs, violations, bundle, errors)


def write_proofs(path: str, result: RunResult) -> None:
    """Atomic: a serialization failure leaves the previous proofs file intact."""
    atomic_write(path, lambda fh: json.dump(result.proofs, fh, indent=1), mode="w", encoding="utf-8")
