"""CSV loading into typed models. Validation happens at this boundary only."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

from .models import (
    Dataset, Event, FxTable, ImageRef, Message, PaymentOption, Profile, Request,
    EVENT_STATUSES, FLEXIBILITIES,
)
from .money import parse_money


def _date(s: str) -> date:
    return date.fromisoformat(s.strip())


def _opt_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    return _date(s) if s else None


def _split(s: str):
    return tuple(x.strip() for x in (s or "").split("|") if x.strip())


def _read(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _lenient_money(text: Optional[str]) -> Optional[Decimal]:
    """An event money cell, tolerant of a literal that cannot be parsed.

    Loading happens once, before the per-request isolation in pipeline.run, so a strict parse
    would let one unreadable cell in financial_events.csv abort the whole batch and produce no
    output at all - the contract requires one row per request_id whatever the rows contain. A
    malformed cell is therefore read exactly like a blank one (None = unresolved) and handed to
    the machinery that already exists for an unknown amount: a pending/scheduled debit with no
    usable amount fails closed (UnresolvedCashEvidence -> the conservative fallback row), a settled
    row drops out of recurring history, and a credit is never counted. Nothing is guessed from a
    malformed cell and it is never read as zero. Strictness is kept everywhere else (requests,
    profiles, payment options), where an unreadable amount is not something a row can survive.
    """
    try:
        value = parse_money(text)
    except ValueError:
        return None
    # "NaN" and "Infinity" parse as Decimal but are not quantities anyone can pay, so they are
    # read as unresolved exactly like a literal that does not parse at all.
    return value if value is None or value.is_finite() else None


def load_profiles(path: str) -> Dict[str, Profile]:
    out: Dict[str, Profile] = {}
    for r in _read(path):
        mim = (r.get("max_installment_months") or "").strip()
        out[r["user_id"]] = Profile(
            user_id=r["user_id"],
            home_currency=r["home_currency"].strip(),
            current_available_balance=parse_money(r["current_available_balance"]),
            minimum_balance_to_keep=parse_money(r["minimum_balance_to_keep"]),
            financial_priorities=_split(r.get("financial_priorities", "")),
            protect_categories=_split(r.get("expense_categories_to_protect", "")),
            reduce_categories=_split(r.get("expense_categories_user_is_willing_to_reduce", "")),
            stop_categories=_split(r.get("expense_categories_user_is_willing_to_stop", "")),
            payment_methods=_split(r.get("payment_methods_user_will_consider", "")),
            max_installment_months=int(mim) if mim else None,
        )
    return out


def load_events(path: str) -> List[Event]:
    out: List[Event] = []
    for r in _read(path):
        status = r["status"].strip()
        flex = (r.get("flexibility") or "fixed").strip() or "fixed"
        if status not in EVENT_STATUSES:
            raise ValueError(f"{r['event_id']}: unknown status {status!r}")
        if flex not in FLEXIBILITIES:
            raise ValueError(f"{r['event_id']}: unknown flexibility {flex!r}")
        out.append(Event(
            event_id=r["event_id"],
            user_id=r["user_id"],
            event_type=r["event_type"].strip(),
            description=r["description"].strip(),
            category=r["category"].strip(),
            direction=r["direction"].strip(),
            amount=_lenient_money(r.get("amount")),
            currency=r["currency"].strip(),
            event_date=_date(r["event_date"]),
            settlement_date=_opt_date(r.get("settlement_date")),
            status=status,
            linked_event_id=(r.get("linked_event_id") or "").strip() or None,
            flexibility=flex,
            minimum_allowed_amount=_lenient_money(r.get("minimum_allowed_amount")),
        ))
    return out


def load_requests(path: str) -> List[Request]:
    out: List[Request] = []
    for r in _read(path):
        out.append(Request(
            request_id=r["request_id"],
            user_id=r["user_id"],
            request_date=_date(r["request_date"]),
            request_type=r["request_type"].strip(),
            requested_amount=parse_money(r["requested_amount"]),
            desired_completion_date=_date(r["desired_completion_date"]),
            allows_partial_payment=r["allows_partial_payment"].strip().lower() == "true",
            request_text=r.get("request_text", ""),
        ))
    return out


def load_payment_options(path: str) -> Dict[str, List[PaymentOption]]:
    out: Dict[str, List[PaymentOption]] = defaultdict(list)
    for r in _read(path):
        freq = (r.get("payment_frequency_days") or "").strip()
        out[r["request_id"]].append(PaymentOption(
            payment_option_id=r["payment_option_id"],
            request_id=r["request_id"],
            payment_method=r["payment_method"].strip(),
            payment_amount=parse_money(r["payment_amount"]),
            number_of_payments=int(r["number_of_payments"]),
            first_payment_date=_date(r["first_payment_date"]),
            payment_frequency_days=int(freq) if freq else None,
            financing_fee=parse_money(r.get("financing_fee")) or Decimal("0"),
            total_payable_amount=parse_money(r["total_payable_amount"]),
        ))
    return dict(out)


def load_messages(path: str) -> List[Message]:
    return [Message(
        message_id=r["message_id"], user_id=r["user_id"],
        request_id=(r.get("request_id") or "").strip() or None,
        related_event_id=(r.get("related_event_id") or "").strip() or None,
        sent_at=r.get("sent_at", ""), source_type=r.get("source_type", ""),
        message_text=r.get("message_text", ""),
    ) for r in _read(path)]


def load_images(path: str, dataset_dir: str) -> List[ImageRef]:
    out = []
    for r in _read(path):
        out.append(ImageRef(
            image_id=r["image_id"], user_id=r["user_id"],
            request_id=(r.get("request_id") or "").strip() or None,
            related_event_id=(r.get("related_event_id") or "").strip() or None,
            path=os.path.join(dataset_dir, "media", "images", f"{r['image_id']}.png"),
        ))
    return out


def load_fx(path: str) -> FxTable:
    fx = FxTable()
    for r in _read(path):
        fx.add(_date(r["rate_date"]), r["from_currency"].strip(), r["to_currency"].strip(), Decimal(r["rate"].strip()))
    return fx


def load_dataset(dataset_dir: str, requests_file: str = "requests.csv") -> Dataset:
    p = lambda name: os.path.join(dataset_dir, name)
    events = load_events(p("financial_events.csv"))
    by_user: Dict[str, List[Event]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    for lst in by_user.values():
        lst.sort(key=lambda e: (e.event_date, e.event_id))
    return Dataset(
        profiles=load_profiles(p("financial_profiles.csv")),
        events=events,
        events_by_user=dict(by_user),
        events_by_id={e.event_id: e for e in events},
        requests=load_requests(p(requests_file)),
        options_by_request=load_payment_options(p("request_payment_options.csv")),
        messages=load_messages(p("messages.csv")),
        images=load_images(p("images.csv"), dataset_dir),
        fx=load_fx(p("exchange_rates.csv")),
        dataset_dir=dataset_dir,
    )
