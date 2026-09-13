"""Typed records for every participant-facing dataset file."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

ALLOWED_STATUS = ("affordable_now", "affordable_with_plan", "affordable_later", "not_affordable")
ALLOWED_METHODS = ("full_payment", "partial_payment", "installments", "wait", "not_recommended")
EVENT_STATUSES = ("settled", "pending", "scheduled", "failed", "cancelled", "unrealized")
FLEXIBILITIES = ("fixed", "reducible", "stoppable", "reducible_or_stoppable")
HORIZON_DAYS = 90


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: Tuple[str, ...]
    protect_categories: Tuple[str, ...]
    reduce_categories: Tuple[str, ...]
    stop_categories: Tuple[str, ...]
    payment_methods: Tuple[str, ...]
    max_installment_months: Optional[int]


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # debit | credit | non_cash
    amount: Optional[Decimal]  # None == blank in CSV (must be resolved from evidence)
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]

    @property
    def cash_date(self) -> date:
        return self.settlement_date or self.event_date

    @property
    def is_debit(self) -> bool:
        return self.direction == "debit"

    @property
    def is_credit(self) -> bool:
        return self.direction == "credit"


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str  # full_payment | installments
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal

    @property
    def schedule_defined(self) -> bool:
        """A single payment needs no interval; several payments need a stated positive day count.

        The statement defines an option by "when payments begin, the number of days between
        recurring payments, ..." and forbids inventing payment information. With more than one
        payment and no interval the dates of the later legs are unknown: the schedule is
        undefined, and no default cadence (nor a same-day collapse) is assumed for it.
        """
        return self.number_of_payments <= 1 or bool(self.payment_frequency_days and self.payment_frequency_days > 0)

    def schedule(self) -> List[Tuple[date, Decimal]]:
        from datetime import timedelta

        if not self.schedule_defined:
            return []                     # undefined: never "all legs on the first date"
        out = []
        d = self.first_payment_date
        for i in range(self.number_of_payments):
            out.append((d, self.payment_amount))
            if self.payment_frequency_days:
                d = d + timedelta(days=self.payment_frequency_days)
        return out

    @property
    def option_index(self) -> int:
        """Numeric part of payment_option_id for the final tie-break."""
        tail = self.payment_option_id.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    path: str


@dataclass
class Dataset:
    profiles: Dict[str, Profile]
    events: List[Event]
    events_by_user: Dict[str, List[Event]]
    events_by_id: Dict[str, Event]
    requests: List[Request]
    options_by_request: Dict[str, List[PaymentOption]]
    messages: List[Message]
    images: List[ImageRef]
    fx: "FxTable"
    dataset_dir: str = ""


@dataclass
class FxTable:
    rates: Dict[Tuple[str, str], Dict[date, Decimal]] = field(default_factory=dict)

    def add(self, rate_date: date, src: str, dst: str, rate: Decimal) -> None:
        self.rates.setdefault((src, dst), {})[rate_date] = rate
