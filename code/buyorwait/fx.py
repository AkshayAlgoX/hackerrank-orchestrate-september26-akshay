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
    """
    if src == dst:
        return Decimal("1"), on
    table = fx.rates.get((src, dst))
    if not table:
        raise MissingRate(f"no rates for {src}->{dst}")
    if on in table:
        return table[on], on
    earlier = [d for d in table if d < on]
    if earlier:
        d = max(earlier)
        return table[d], d
    d = min(table)
    return table[d], d


def convert(fx: FxTable, amount: Decimal, on: date, src: str, dst: str) -> Decimal:
    rate, _ = lookup_rate(fx, on, src, dst)
    return q2(amount * rate)
