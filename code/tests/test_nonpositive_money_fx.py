"""Non-positive money and unusable FX rates (Target H3).

One rule, stated once: a quantity that is not a usable amount is never silently turned into one.
For every amount that reaches the ledger through `_home_amount` (the single gate the event amounts
pass through) and for every rate the FX table can answer with:

1. A debit whose amount is not a positive quantity has no usable amount. It is treated exactly
   like a blank one, so a pending/scheduled debit still due inside the window fails closed
   (UnresolvedCashEvidence -> the conservative fallback row), and a settled row drops out of the
   history it would otherwise seed.
2. An FX rate that is zero, negative or not finite behaves exactly like a missing rate
   (MissingRate): the amount is excluded, never multiplied by it. A zero rate would erase a debit
   and a negative rate would flip a debit into a credit, which is worse than not knowing.
3. A malformed amount cell at load time cannot abort the batch: `load_dataset` runs before any
   per-request isolation, so the cell loads as an unresolved amount instead.
4. Valid positive income, valid positive amounts and valid positive rates are untouched: a clean
   world gains no new audit line and no different decision.
5. Confirmed foreign income whose rate is unusable is never counted: it cannot be converted, so it
   is not income.
6. A historical occurrence with an unusable amount never lowers a recurrence forecast: it is
   dropped from the sample, and the degraded estimate is the largest known occurrence, so the
   estimate is never below what counting the invalid sample as an observation would give.

No new exception path is introduced anywhere: every case above ends in the machinery that already
exists for an unknown amount (MissingRate -> None -> unresolved cash / dropped history).
"""
from __future__ import annotations

import csv
import os
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from adversarial.cases import mk_dataset, mk_event, mk_profile, mk_request, monthly
from buyorwait.evidence import Evidence
from buyorwait.extraction.gather import EvidenceBundle
from buyorwait.ledger import _home_amount, UnresolvedCashEvidence, build_ledger
from buyorwait.loaders import load_dataset
from buyorwait.pipeline import run

RD = date(2026, 6, 2)
FAIL_CLOSED = ["0", "not_affordable", "not_recommended", "none", "", "none"]
DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")

# Amounts a dataset could state for a debit that are not quantities money can be paid with.
BAD_AMOUNTS = ["0", "-1", "-500.25", "-0.01", "-2000"]
# Rates a dataset could state that cannot be multiplied by an amount.
BAD_RATES = ["0", "-1", "-0.5", "-0.001", "NaN", "Infinity", "-Infinity"]


def world(extra, balance="3000", minimum="500", amount="1000", rates=()):
    """A small EUR world with a confirmed monthly salary and anything extra appended."""
    ev = monthly("sal", "salary", 2000, 15, 5, etype="income", desc="Payroll credit") + list(extra)
    p = mk_profile(current_available_balance=D(balance), minimum_balance_to_keep=D(minimum))
    req = mk_request(amount, rd=RD, deadline=RD + timedelta(days=40), partial=True)
    ds = mk_dataset(p, ev, req)
    for on, src, dst, rate in rates:
        ds.fx.add(on, src, dst, D(rate))
    return ds, req


def run_clean(ds, **kw):
    return run(ds, bundle=EvidenceBundle([], []), **kw)


# ---- 1. a debit with no usable amount fails closed -------------------------------------------

@pytest.mark.parametrize("amount", BAD_AMOUNTS)
@pytest.mark.parametrize("status,days", [("pending", 1), ("scheduled", 20)])
def test_non_positive_debit_amount_fails_closed(amount, status, days):
    ds, req = world([mk_event("u1", "expense", "utilities", "debit", amount, RD, status=status,
                              settle=RD + timedelta(days=days))])
    with pytest.raises(UnresolvedCashEvidence) as info:
        build_ledger(ds, "u1", RD, [])
    assert info.value.event_ids == ["u1"]
    assert "has no usable amount" in "; ".join(info.value.reasons)
    res = run_clean(ds)
    assert res.rows[0].as_list()[1:7] == FAIL_CLOSED
    assert res.errors[req.request_id].startswith("UnresolvedCashEvidence")
    assert not res.violations


@pytest.mark.parametrize("amount", BAD_AMOUNTS)
@pytest.mark.parametrize("currency,rate", [("EUR", None), ("USD", "0"), ("USD", "-1.5"), ("USD", "NaN")])
def test_any_unusable_amount_and_rate_combination_fails_closed(amount, currency, rate):
    """The row is the fallback whatever makes the amount unusable, and never a payment."""
    rates = () if rate is None else [(RD - timedelta(days=1), "USD", "EUR", rate)]
    ds, _ = world([mk_event("u1", "expense", "utilities", "debit", amount, RD, status="scheduled",
                            settle=RD + timedelta(days=5), currency=currency)], rates=rates)
    res = run_clean(ds)
    assert res.rows[0].as_list()[1:7] == FAIL_CLOSED
    assert res.rows[0].as_list()[1] == "0"          # nothing is claimed safe, not even a cent
    assert not res.violations


@pytest.mark.parametrize("amount", BAD_AMOUNTS)
def test_non_positive_amount_resolved_by_evidence_is_still_not_usable(amount):
    """Evidence that resolves a debit to a non-positive amount does not make it payable either."""
    ds, _ = world([mk_event("u1", "expense", "utilities", "debit", None, RD, status="pending",
                            settle=RD + timedelta(days=2))])
    ev = Evidence("image", "image_x", "u1", "expense_amount_resolved", request_id="r1", related_event_id="u1",
                  amount=D(amount), currency="EUR", sent_at="2026-06-01T00:00:00Z")
    with pytest.raises(UnresolvedCashEvidence):
        build_ledger(ds, "u1", RD, [ev])
    assert run(ds, bundle=EvidenceBundle([ev], [])).rows[0].as_list()[1:7] == FAIL_CLOSED


def test_the_audit_says_why_the_amount_is_unusable():
    """The refusal reuses the ledger's existing unresolved-cash reason, and the ledger audit
    records the specific cause for a reviewer."""
    ds, _ = world([])
    for amount in BAD_AMOUNTS:
        e = mk_event("x", "expense", "utilities", "debit", amount, RD)
        audit = []
        assert _home_amount(ds, e, "EUR", {}, audit) is None
        assert len(audit) == 1 and "not positive" in audit[0]


def test_an_unresolved_debit_is_never_reserved_as_zero():
    """A non-positive amount must behave exactly like an unstated one: no 0 flow is created, since
    a reservation of 0 would claim the obligation costs nothing."""
    for amount in BAD_AMOUNTS + [None]:
        ds, _ = world([mk_event("u1", "expense", "utilities", "debit", amount, RD + timedelta(days=20),
                                status="scheduled")])
        with pytest.raises(UnresolvedCashEvidence) as info:
            build_ledger(ds, "u1", RD, [])
        assert info.value.event_ids == ["u1"]


# ---- 2. an unusable rate behaves exactly like a missing rate ----------------------------------

@pytest.mark.parametrize("rate", BAD_RATES)
def test_unusable_rate_is_indistinguishable_from_a_missing_rate(rate):
    """Same ledger refusal, same reasons, same fallback row as when the pair has no rate at all."""
    extra = [mk_event("u1", "expense", "utilities", "debit", 100, RD, status="scheduled",
                      settle=RD + timedelta(days=5), currency="USD")]
    bad, _ = world(extra, rates=[(RD - timedelta(days=1), "USD", "EUR", rate)])
    missing, _ = world(extra)
    with pytest.raises(UnresolvedCashEvidence) as a:
        build_ledger(bad, "u1", RD, [])
    with pytest.raises(UnresolvedCashEvidence) as b:
        build_ledger(missing, "u1", RD, [])
    assert a.value.reasons == b.value.reasons
    assert run_clean(bad).rows[0].as_list() == run_clean(missing).rows[0].as_list()
    assert run_clean(bad).rows[0].as_list()[1:7] == FAIL_CLOSED


@pytest.mark.parametrize("rate", BAD_RATES)
@pytest.mark.parametrize("amount", ["0.01", "100", "999999.99"])
def test_home_amount_gate_returns_none_or_a_positive_decimal(rate, amount):
    """The gate's own contract: never zero, never negative, and never a fabricated value."""
    ds, _ = world([], rates=[(RD, "USD", "EUR", rate)])
    e = mk_event("x", "expense", "utilities", "debit", amount, RD, currency="USD")
    assert _home_amount(ds, e, "EUR", {}) is None


@pytest.mark.parametrize("amount", BAD_AMOUNTS)
def test_home_amount_gate_rejects_a_non_positive_amount_in_the_home_currency(amount):
    ds, _ = world([])
    e = mk_event("x", "expense", "utilities", "debit", amount, RD, currency="EUR")
    assert _home_amount(ds, e, "EUR", {}) is None


def test_home_amount_gate_leaves_valid_amounts_and_rates_alone():
    ds, _ = world([], rates=[(RD, "USD", "EUR", "0.9")])
    fx_debit = mk_event("x", "expense", "utilities", "debit", "100", RD, currency="USD")
    assert _home_amount(ds, fx_debit, "EUR", {}) == D("90")
    home_debit = mk_event("y", "expense", "utilities", "debit", "100", RD, currency="EUR")
    assert _home_amount(ds, home_debit, "EUR", {}) == D("100")


@pytest.mark.parametrize("rate", BAD_RATES)
def test_an_unusable_rate_is_rejected_on_its_own_date_without_falling_back(rate):
    """A rejected rate row is not replaced by another date's rate: that would invent a rate the
    dataset never supplied for the settlement date."""
    extra = [mk_event("u1", "expense", "utilities", "debit", 100, RD, status="scheduled",
                      settle=RD + timedelta(days=5), currency="USD")]
    ds, _ = world(extra, rates=[(RD, "USD", "EUR", rate), (RD + timedelta(days=30), "USD", "EUR", "0.9")])
    with pytest.raises(UnresolvedCashEvidence):
        build_ledger(ds, "u1", RD, [])


# ---- 5. unusable FX on income invents nothing --------------------------------------------------

@pytest.mark.parametrize("rate", ["0", "-1", "NaN"])
def test_scheduled_foreign_payroll_with_an_unusable_rate_is_never_income(rate):
    sched = [mk_event("u1", "income", "salary", "credit", 3000, RD + timedelta(days=10), status="scheduled",
                      currency="USD", desc="Next confirmed salary")]
    bad, _ = world(sched, rates=[(RD - timedelta(days=1), "USD", "EUR", rate)])
    missing, _ = world(sched)
    good, _ = world(sched, rates=[(RD - timedelta(days=1), "USD", "EUR", "0.9")])
    # with a usable rate the confirmed payroll is counted, so the checks below are not vacuous
    assert any(f.source_event_id == "u1" for f in build_ledger(good, "u1", RD, []).salary_flows)
    # without one, exactly the missing-rate outcome, and nothing is claimed safe from it
    assert run_clean(bad).rows[0].as_list() == run_clean(missing).rows[0].as_list()
    assert run_clean(bad).rows[0].as_list()[1] == "0"


# ---- 4. a clean world is untouched -------------------------------------------------------------

def test_valid_positive_income_and_rates_are_unaffected():
    ds, _ = world([mk_event("u1", "expense", "utilities", "debit", 100, RD + timedelta(days=10),
                            status="scheduled", currency="USD")],
                  rates=[(RD - timedelta(days=1), "USD", "EUR", "0.9")])
    L = build_ledger(ds, "u1", RD, [])
    assert [str(f.amount) for f in L.salary_flows] == ["2000.00", "2000.00", "2000.00"]
    assert [str(f.amount) for f in L.known_flows] == ["-90.00"]
    assert not any("not positive" in a or "no rates" in a for a in L.audit)
    res = run_clean(ds)
    assert not res.errors
    row = res.rows[0].as_list()
    assert row[2] == "affordable_now" and D(row[1]) == D("1000")


# ---- 6. invalid history never lowers a recurrence forecast -------------------------------------

def _series_amount(ds, category="utilities"):
    L = build_ledger(ds, "u1", RD, [])
    found = [s for s in L.series if s.category == category]
    assert len(found) == 1, [s.category for s in L.series]
    return found[0].amount


@pytest.mark.parametrize("amount", BAD_AMOUNTS)
def test_invalid_historical_amount_is_treated_like_an_absent_occurrence(amount):
    """The 4th monthly bill states an unusable amount. The forecast must equal the one built from
    the three real occurrences alone - the largest known one - and never the mean that counting
    the invalid row as an observation would drag down (300 here, well below 400)."""
    base = monthly("r", "utilities", 400, 10, 4)
    invalid = base[:-1] + [mk_event("r3", "expense", "utilities", "debit", amount, base[-1].event_date)]
    absent = base[:-1] + [mk_event("r3", "expense", "utilities", "debit", None, base[-1].event_date)]
    ds_invalid, _ = world(invalid)
    ds_absent, _ = world(absent)
    est = _series_amount(ds_invalid)
    assert est == _series_amount(ds_absent) == D("400")
    assert est >= D("400") > D("300")


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_more_invalid_history_never_lowers_the_estimate(k):
    """Monotone in the number of unusable occurrences: the estimate can only stay or rise."""
    base = monthly("r", "utilities", 400, 10, 4)
    events = list(base)
    for i in range(k):
        j = len(base) - 1 - i
        events[j] = mk_event(base[j].event_id, "expense", "utilities", "debit", "-1", base[j].event_date)
    ds, _ = world(events)
    assert _series_amount(ds) == D("400")


def test_all_occurrences_invalid_makes_the_series_unforecastable_and_fails_closed():
    """With no usable occurrence at all nothing can be forecast, so the request fails closed."""
    events = [mk_event(e.event_id, "expense", "utilities", "debit", "0", e.event_date)
              for e in monthly("r", "utilities", 400, 10, 4)]
    ds, _ = world(events)
    with pytest.raises(UnresolvedCashEvidence):
        build_ledger(ds, "u1", RD, [])
    assert run_clean(ds).rows[0].as_list()[1:7] == FAIL_CLOSED


def test_variable_history_with_an_invalid_occurrence_is_not_forecast_below_its_maximum():
    """The variable path (sub-monthly, rotating descriptions) uses the largest typical known
    occurrence when a sample is missing, never the mean the invalid sample would pull down."""
    start = date(2026, 4, 6)
    weekly = [mk_event(f"g{i}", "expense", "groceries", "debit", str(100 + 10 * i),
                       start + timedelta(days=7 * i), desc=f"Market basket {i}", flex="reducible")
              for i in range(6)]
    weekly[-1] = mk_event("g5", "expense", "groceries", "debit", "-100", start + timedelta(days=35),
                          desc="Market basket 5", flex="reducible")
    ds, _ = world(weekly)
    est = _series_amount(ds, "groceries")
    assert est == D("140")
    assert est > D("66.67")          # the mean if the invalid row were counted as an observation


# ---- 3. a malformed cell cannot abort the batch ------------------------------------------------

def _headers():
    out = {}
    for name in ("financial_profiles.csv", "financial_events.csv", "exchange_rates.csv",
                 "request_payment_options.csv", "messages.csv", "images.csv", "requests.csv"):
        with open(os.path.join(DATASET, name), encoding="utf-8") as fh:
            out[name] = fh.readline().strip()
    return out


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header.split(","))
        w.writerows(rows)


def temp_dataset(tmp_path, event_rows, request_rows):
    """A minimal but complete dataset directory: only what load_dataset reads."""
    h = _headers()
    profiles = [["user_01", "EUR", "3000", "500", "education", "rent", "dining", "delivery_membership",
                 "full_payment|partial_payment", ""],
                ["user_02", "EUR", "3000", "500", "education", "rent", "dining", "delivery_membership",
                 "full_payment|partial_payment", ""]]
    _write(tmp_path / "financial_profiles.csv", h["financial_profiles.csv"], profiles)
    _write(tmp_path / "financial_events.csv", h["financial_events.csv"], event_rows)
    for name in ("exchange_rates.csv", "request_payment_options.csv", "messages.csv", "images.csv"):
        _write(tmp_path / name, h[name], [])
    _write(tmp_path / "requests.csv", h["requests.csv"], request_rows)
    return str(tmp_path)


def _event_row(eid, user, etype, desc, cat, direction, amount, on, status="settled", settle=None, minallowed=""):
    return [eid, user, etype, desc, cat, direction, amount, "EUR", on.isoformat(),
            (settle or on).isoformat(), status, "", "fixed", minallowed]


def _request_row(rid, user):
    return [rid, user, RD.isoformat(), "purchase", "1000", (RD + timedelta(days=40)).isoformat(), "true", "?"]


SALARY_ROWS = [_event_row(f"sal{i}", "user_01", "income", "Payroll credit", "salary", "credit", "2000",
                          date(2026, 3 + i, 15)) for i in range(3)]


def test_a_malformed_amount_cell_does_not_abort_the_batch(tmp_path):
    """One unreadable cell used to raise at load time, before any per-request isolation, so the
    whole batch produced no output at all. It now loads as an unresolved amount: the request that
    depends on it fails closed and every other request is still answered."""
    ds_dir = temp_dataset(tmp_path,
                          SALARY_ROWS + [
                              _event_row("bad_debit", "user_01", "expense", "Power bill", "utilities", "debit",
                                         "1,2,3.4.5", RD - timedelta(days=5), status="pending",
                                         settle=RD + timedelta(days=3)),
                              _event_row("bad_history", "user_01", "expense", "Water bill", "utilities", "debit",
                                         "n/a", RD - timedelta(days=40)),
                              _event_row("clean", "user_02", "income", "Payroll credit", "salary", "credit",
                                         "2000", date(2026, 5, 15))],
                          [_request_row("r1", "user_01"), _request_row("r2", "user_02")])
    ds = load_dataset(ds_dir, "requests.csv")                      # must not raise
    assert ds.events_by_id["bad_debit"].amount is None             # unreadable is unresolved, not zero
    assert ds.events_by_id["bad_history"].amount is None
    assert ds.events_by_id["clean"].amount == D("2000")
    res = run(ds, use_model=False, cache_path=os.path.join(ds_dir, "cache.json"))
    assert [r.request_id for r in res.rows] == ["r1", "r2"]         # one row per request, still
    rows = {r.request_id: r.as_list() for r in res.rows}
    assert rows["r1"][1:7] == FAIL_CLOSED                           # the affected request fails closed
    assert rows["r2"][2] == "affordable_now"                        # the rest of the batch is unaffected
    assert not res.violations


def test_a_malformed_amount_is_reported_as_unresolved_not_silently_ignored(tmp_path):
    """The audit line the ledger already writes for a blank amount covers the malformed cell too."""
    ds_dir = temp_dataset(tmp_path,
                          SALARY_ROWS + [_event_row("bad_history", "user_01", "expense", "Water bill",
                                                    "utilities", "debit", "??", RD - timedelta(days=40))],
                          [_request_row("r1", "user_01")])
    ds = load_dataset(ds_dir, "requests.csv")
    L = build_ledger(ds, "user_01", RD, [])
    assert any("bad_history: blank amount unresolved" in a for a in L.audit)


def test_a_malformed_minimum_allowed_amount_only_removes_the_option_to_reduce(tmp_path):
    ds_dir = temp_dataset(tmp_path,
                          SALARY_ROWS + [_event_row("clean", "user_02", "income", "Payroll credit", "salary",
                                                    "credit", "2000", date(2026, 5, 15),
                                                    minallowed="about 400")],
                          [_request_row("r2", "user_02")])
    ds = load_dataset(ds_dir, "requests.csv")
    assert ds.events_by_id["clean"].minimum_allowed_amount is None
    assert not run(ds, use_model=False, cache_path=os.path.join(ds_dir, "cache.json")).violations


def test_the_shipped_dataset_has_no_non_positive_amount_or_rate():
    """The change is inert on the supplied data: the only unusable amounts there are blank ones,
    and every supplied rate is finite and positive, so no row can reach the new exclusions."""
    ds = load_dataset(DATASET, "requests.csv")
    assert ds.events and ds.fx.rates
    assert all(e.amount is None or e.amount > 0 for e in ds.events)
    assert all(r.is_finite() and r > 0 for table in ds.fx.rates.values() for r in table.values())
