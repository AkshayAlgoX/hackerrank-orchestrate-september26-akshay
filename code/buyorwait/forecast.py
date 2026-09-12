"""90-day cash-flow simulation and safety checks (pure, exact arithmetic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from .ledger import Flow, Ledger, Series
from .money import ZERO, q2


@dataclass(frozen=True)
class SpendingChange:
    action: str                 # "stop" | "reduce_to"
    series_id: str
    event_id: str               # latest occurrence of the series, used in output
    description: str
    category: str
    new_amount: Decimal         # 0 for stop
    saving: Decimal             # total saving inside the horizon

    def render(self) -> str:
        from .money import fmt_plan
        if self.action == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{fmt_plan(self.new_amount)}"


@dataclass
class Path:
    """Piecewise-constant balance: `points[i] = (date, balance after all flows that day)`."""
    opening: Decimal
    points: List[Tuple[date, Decimal]]

    def balance_on(self, d: date) -> Decimal:
        b = self.opening
        for on, bal in self.points:
            if on <= d:
                b = bal
            else:
                break
        return b

    @property
    def minimum(self) -> Decimal:
        return min([self.opening] + [b for _, b in self.points])

    def suffix_minimum(self, d: date) -> Decimal:
        """min balance over all t >= d (balance on d includes that day's flows)."""
        m = self.balance_on(d)
        for on, bal in self.points:
            if on > d:
                m = min(m, bal)
        return m


def project_flows(L: Ledger, changes: Sequence[SpendingChange] = ()) -> List[Flow]:
    """All flows inside [request_date, horizon_end] after applying spending changes."""
    change_by_series: Dict[str, SpendingChange] = {c.series_id: c for c in changes}
    flows: List[Flow] = list(L.known_flows) + list(L.salary_flows)
    for s in L.series:
        amt = s.amount
        if s.series_id in change_by_series:
            c = change_by_series[s.series_id]
            amt = ZERO if c.action == "stop" else c.new_amount
        if amt == ZERO:
            continue
        for d in s.dates(L.request_date, L.horizon_end):
            flows.append(Flow(d, amt if s.is_income else -amt, f"projected {s.description}", None, s.series_id, s.category))
    return flows


# Same-day flows are netted: the statement defines the check per day ("the balance never falls
# below minimum_balance_to_keep") and gives no intraday ordering.


def simulate(opening: Decimal, flows: Sequence[Flow], payments: Sequence[Tuple[date, Decimal]] = ()) -> Path:
    net: Dict[date, Decimal] = {}
    for f in flows:
        net[f.on] = net.get(f.on, ZERO) + f.amount
    for on, amt in payments:
        net[on] = net.get(on, ZERO) - amt
    bal = opening
    pts: List[Tuple[date, Decimal]] = []
    for on in sorted(net):
        bal = bal + net[on]
        pts.append((on, bal))
    return Path(opening, pts)


def is_safe(L: Ledger, flows: Sequence[Flow], payments: Sequence[Tuple[date, Decimal]]) -> bool:
    """True iff the balance never drops below minimum_balance_to_keep inside the horizon."""
    inside = [(d, a) for d, a in payments if d <= L.horizon_end]
    return simulate(L.opening_balance, flows, inside).minimum >= L.minimum_balance


def amount_safe_to_pay(L: Ledger, flows: Sequence[Flow], requested: Decimal) -> Decimal:
    """Largest amount payable on request_date that keeps every projected day >= minimum."""
    path = simulate(L.opening_balance, flows)
    room = path.minimum - L.minimum_balance
    if room <= ZERO:
        return ZERO
    return min(requested, q2(room))


def earliest_full_payment_date(L: Ledger, flows: Sequence[Flow], amount: Decimal) -> Optional[date]:
    """First date D in the horizon on which one payment of `amount` is safe for all t >= D."""
    path = simulate(L.opening_balance, flows)
    candidates = sorted({L.request_date} | {d for d, _ in path.points if L.request_date <= d <= L.horizon_end})
    need = L.minimum_balance + amount
    for D in candidates:
        if path.suffix_minimum(D) >= need:
            return D
    return None
