"""Permitted spending changes: candidate actions and an exhaustive minimal search.

Search order: no change, then every single change, every pair, every triple (the output
allows at most three). Among the sets that make the plan safe, the least disruptive one wins:
smallest total saving inside the forecast window, then fewer changes, then event id. A stop and
a reduce_to never target the same event (statement rule); a reducible_or_stoppable series
therefore contributes two mutually exclusive candidates.

Evidence for "least total saving" rather than "fewest changes": sample request_21 needs 31.05
more; the reference picks {stop cloud 11/mo, reduce streaming 47->23.50} (34.50/mo) over the
single valid change {stop streaming} (47/mo).
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

from .forecast import SpendingChange, project_flows, is_safe
from .ledger import Ledger
from .money import ZERO, q2

MAX_CHANGES = 3


def candidate_changes(L: Ledger) -> List[SpendingChange]:
    """Every permitted change, ordered by ascending saving (least disruptive first)."""
    p = L.profile
    out: List[SpendingChange] = []
    for s in L.series:
        if s.is_income or s.category in p.protect_categories or s.flexibility == "fixed":
            continue
        n = len(s.dates(L.request_date, L.horizon_end))
        if n == 0:
            continue
        can_reduce = s.flexibility in ("reducible", "reducible_or_stoppable") and s.category in p.reduce_categories \
            and s.minimum_allowed_amount is not None and s.minimum_allowed_amount < s.amount
        can_stop = s.flexibility in ("stoppable", "reducible_or_stoppable") and s.category in p.stop_categories
        if can_reduce:
            new_amt = q2(s.minimum_allowed_amount)
            out.append(SpendingChange("reduce_to", s.series_id, s.latest_event_id, s.description, s.category,
                                      new_amt, q2((s.amount - new_amt) * n)))
        if can_stop:
            out.append(SpendingChange("stop", s.series_id, s.latest_event_id, s.description, s.category,
                                      ZERO, q2(s.amount * n)))
    out.sort(key=lambda c: (c.saving, c.event_id, c.action))
    return out


def _rank(chs: Sequence[SpendingChange]) -> Tuple:
    return (sum((c.saving for c in chs), ZERO), len(chs), tuple(sorted((c.event_id, c.action) for c in chs)))


def select_changes(L: Ledger, payments: Sequence[Tuple[date, Decimal]]) -> Optional[List[SpendingChange]]:
    """Least disruptive set of at most MAX_CHANGES permitted changes that makes `payments` safe.

    Returns [] when no change is needed and None when no permitted set works.
    """
    def safe_with(chs: Sequence[SpendingChange]) -> bool:
        return is_safe(L, project_flows(L, chs), payments)

    if safe_with([]):
        return []
    cands = candidate_changes(L)
    if not cands:
        return None
    best: Optional[List[SpendingChange]] = None
    for size in range(1, MAX_CHANGES + 1):
        for combo in combinations(cands, size):
            if len({c.series_id for c in combo}) < size:
                continue  # stop and reduce_to on the same event are mutually exclusive
            if best is not None and _rank(combo) >= _rank(best):
                continue
            if safe_with(combo):
                best = list(combo)
    if best is None:
        return None
    return sorted(best, key=lambda c: (c.saving, c.event_id, c.action))
