"""Exact monetary arithmetic helpers. All money in this package is `Decimal`."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, InvalidOperation
from typing import Optional

CENT = Decimal("0.01")
ZERO = Decimal("0")


def parse_money(text: Optional[str]) -> Optional[Decimal]:
    """Parse a CSV amount. Blank -> None (never zero)."""
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"invalid money literal: {text!r}") from exc


def q2(x: Decimal) -> Decimal:
    """Round half-up to cents."""
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def floor2(x: Decimal) -> Decimal:
    """Round down to cents (conservative for 'safe' amounts)."""
    return x.quantize(CENT, rounding=ROUND_DOWN)


def fmt_short(x: Decimal) -> str:
    """Shortest exact representation: 603.3, 87170.56, 25256."""
    x = q2(x)
    s = format(x.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def fmt_plan(x: Decimal) -> str:
    """Plan/reduce_to amounts: integer when whole, else exactly two decimals (620.40)."""
    x = q2(x)
    if x == x.to_integral_value():
        return str(int(x))
    return format(x, "f")


def fmt_human(x: Decimal) -> str:
    """Thousands separators, two decimals when fractional: 1,574.40 / 18,000."""
    x = q2(x)
    if x == x.to_integral_value():
        return f"{int(x):,}"
    return f"{x:,.2f}"
