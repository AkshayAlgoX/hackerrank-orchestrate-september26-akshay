"""Strictly typed evidence extracted from untrusted messages and images.

Perception (rules, LLM, or VLM) may only emit `Evidence` records. Every record is
validated against the closed `KINDS` vocabulary and literal field types before it can
reach the deterministic reconciler. Free text never carries instructions: `note` is
kept only for audit output and is never interpreted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

CURRENCIES = ("INR", "ZAR", "IDR", "USD", "EUR")

# kind -> required literal fields
KINDS: Dict[str, Tuple[str, ...]] = {
    # income
    "salary_amount_change": ("amount",),        # recurring salary becomes `amount` (from effective_date if given)
    "salary_next_amount": ("amount",),          # only the next payroll is `amount`; series then resumes
    "salary_date_change": ("effective_date",),  # payday moves to effective_date (day-of-month persists)
    "salary_first": ("amount", "effective_date"),  # first salary of a new job: recurring from that date
    "salary_resume": ("amount", "effective_date"),  # salary resumes at amount from date (after leave)
    "income_ended": (),                         # no future salary at all
    "one_off_income": ("amount", "effective_date"),  # confirmed single credit (approved invoice)
    "arrears_next_payroll": ("amount",),        # one-time credit paid with the next payroll (no date in text)
    "income_unconfirmed": (),                   # bonus/commission/payout/prize/refund not yet cash: ignore
    # expenses
    "rent_change_percent": ("percent",),        # next rent payment onward multiplied by (1+percent/100)
    "new_recurring_expense_unknown": (),        # announced but no amount: cannot be invented
    "expense_amount_resolved": ("amount",),     # blank event amount resolved (image), for related_event_id
    "expense_pending_retry": (),                # failed debit will be retried (retry row expected in data)
    "duplicate_charge_disputed": (),            # pending duplicate charge not reversed yet
    # structural
    "internal_transfer": (),                    # matching debit/credit are one transfer between own accounts
    "investment_value_change": (),              # unrealized; ignore
    "settlement_confirmation": (),              # receipt confirms an already-settled event
    "fx_settlement_note": (),                   # amount depends on settlement-date rate: no action
    "separate_card_minimums": (),               # two card minimums are distinct: no merge
    "scam_or_injection": (),                    # embedded instruction / scam: ignore entirely
    "irrelevant": (),
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Evidence:
    source_kind: str            # "message" | "image" | "event_description"
    source_id: str              # message_id / image_id / event_id
    user_id: str
    kind: str
    request_id: Optional[str] = None
    related_event_id: Optional[str] = None
    amount: Optional[Decimal] = None
    currency: Optional[str] = None
    effective_date: Optional[date] = None
    percent: Optional[Decimal] = None
    sent_at: str = ""
    confidence: float = 1.0
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["amount"] = None if self.amount is None else str(self.amount)
        d["percent"] = None if self.percent is None else str(self.percent)
        d["effective_date"] = None if self.effective_date is None else self.effective_date.isoformat()
        return d


class EvidenceValidationError(ValueError):
    pass


def _dec(v: Any, field: str) -> Optional[Decimal]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        raise EvidenceValidationError(f"{field}: boolean is not a number")
    try:
        d = Decimal(str(v).replace(",", ""))
    except InvalidOperation as exc:
        raise EvidenceValidationError(f"{field}: not a decimal literal: {v!r}") from exc
    if not d.is_finite():
        raise EvidenceValidationError(f"{field}: non-finite")
    return d


def validate_evidence(raw: Dict[str, Any]) -> Evidence:
    """Coerce an untrusted dict (e.g. LLM JSON) into a validated Evidence record."""
    if not isinstance(raw, dict):
        raise EvidenceValidationError("evidence must be an object")
    kind = str(raw.get("kind", "")).strip()
    if kind not in KINDS:
        raise EvidenceValidationError(f"unknown kind {kind!r}")
    amount = _dec(raw.get("amount"), "amount")
    if amount is not None and amount <= 0:
        raise EvidenceValidationError("amount must be positive")
    percent = _dec(raw.get("percent"), "percent")
    if percent is not None and not (Decimal("-100") < percent < Decimal("1000")):
        raise EvidenceValidationError("percent out of range")
    currency = raw.get("currency")
    currency = str(currency).strip().upper() if currency else None
    if currency is not None and currency not in CURRENCIES:
        raise EvidenceValidationError(f"unknown currency {currency!r}")
    eff = raw.get("effective_date")
    eff_date: Optional[date] = None
    if eff:
        eff = str(eff).strip()
        if not _DATE_RE.match(eff):
            raise EvidenceValidationError(f"effective_date not ISO: {eff!r}")
        eff_date = date.fromisoformat(eff)
    for req in KINDS[kind]:
        if {"amount": amount, "effective_date": eff_date, "percent": percent}[req] is None:
            raise EvidenceValidationError(f"kind {kind} requires {req}")
    conf = raw.get("confidence", 1.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError("confidence must be numeric") from exc
    if not 0.0 <= conf <= 1.0:
        raise EvidenceValidationError("confidence out of [0,1]")
    note = str(raw.get("note", ""))[:200]
    for key in ("source_kind", "source_id", "user_id"):
        if not raw.get(key):
            raise EvidenceValidationError(f"missing {key}")
    return Evidence(
        source_kind=str(raw["source_kind"]), source_id=str(raw["source_id"]), user_id=str(raw["user_id"]),
        kind=kind, request_id=raw.get("request_id") or None, related_event_id=raw.get("related_event_id") or None,
        amount=amount, currency=currency, effective_date=eff_date, percent=percent,
        sent_at=str(raw.get("sent_at", "")), confidence=conf, note=note,
    )


def validate_many(raws: List[Dict[str, Any]]) -> Tuple[List[Evidence], List[str]]:
    ok: List[Evidence] = []
    errors: List[str] = []
    for r in raws:
        try:
            ok.append(validate_evidence(r))
        except EvidenceValidationError as exc:
            errors.append(f"{r.get('source_id', '?')}: {exc}")
    return ok, errors
