"""Currency conversion with the supplied dated, directional rates."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional, Tuple

from .models import FxTable
from .money import q2


class MissingRate(LookupError):
    pass


def lookup_rate(fx: FxTable, on: date, src: str, dst: str) -> Tuple[Decimal, date]:
    """Rate for (src->dst) on `on`.

    Exact-date match is authoritative. If the exact date is absent, the most recent
    earlier rate is used, else the earliest later rate. No inversion of the reverse
    pair is attempted because the dataset provides every directional pair it expects
    us to use; inverting silently would fabricate a rate.

    A rate that is zero, negative, or not finite is not a usable rate: it is reported
    exactly like an absent one (`MissingRate`) so the caller excludes the amount and
    fails closed, instead of multiplying by it. A zero rate would silently erase a debit
    and a negative rate would flip a debit into a credit, which is worse than not knowing.
    The same applies to a rate row for the chosen date: falling back to another date's
    rate after rejecting this one would invent a rate that was never supplied.
    """
    if src == dst:
        return Decimal("1"), on
    table = fx.rates.get((src, dst))
    if not table:
        raise MissingRate(f"no rates for {src}->{dst}")
    if on in table:
        d = on
    else:
        earlier = [x for x in table if x < on]
        d = max(earlier) if earlier else min(table)
    rate = table[d]
    if not rate.is_finite() or rate <= 0:
        raise MissingRate(f"no usable rate for {src}->{dst} on {d.isoformat()} (rate {rate})")
    return rate, d


def convert(fx: FxTable, amount: Decimal, on: date, src: str, dst: str) -> Decimal:
    rate, _ = lookup_rate(fx, on, src, dst)
    return q2(amount * rate)
