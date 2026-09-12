from datetime import date
from decimal import Decimal as D

import pytest
from hypothesis import given, strategies as st

from buyorwait.fx import MissingRate, convert, lookup_rate
from buyorwait.ledger import Series, add_months
from buyorwait.models import FxTable
from buyorwait.money import fmt_human, fmt_plan, fmt_short, parse_money, q2


def test_parse_money_blank_is_none_not_zero():
    assert parse_money("") is None
    assert parse_money(None) is None
    assert parse_money(" 1,234.50 ") == D("1234.50")


def test_formats_match_sample_style():
    assert fmt_short(D("17229139.20")) == "17229139.2"
    assert fmt_short(D("603.30")) == "603.3"
    assert fmt_short(D("25256")) == "25256"
    assert fmt_plan(D("620.4")) == "620.40"
    assert fmt_plan(D("25256")) == "25256"
    assert fmt_plan(D("15952906.67")) == "15952906.67"
    assert fmt_human(D("1574.4")) == "1,574.40"
    assert fmt_human(D("18000")) == "18,000"


@given(st.decimals(min_value=0, max_value=10**9, places=4, allow_nan=False, allow_infinity=False))
def test_fmt_short_roundtrip_is_cent_exact(x):
    assert D(fmt_short(x)) == q2(x)


def test_fx_exact_date_then_fallback_and_direction():
    fx = FxTable()
    fx.add(date(2024, 3, 15), "USD", "IDR", D("15833.33"))
    fx.add(date(2024, 4, 15), "USD", "IDR", D("16000"))
    assert convert(fx, D("1800"), date(2024, 3, 15), "USD", "IDR") == D("28499994.00")
    rate, used = lookup_rate(fx, date(2024, 4, 1), "USD", "IDR")
    assert (rate, used) == (D("15833.33"), date(2024, 3, 15))
    rate, used = lookup_rate(fx, date(2024, 1, 1), "USD", "IDR")
    assert used == date(2024, 3, 15)
    with pytest.raises(MissingRate):
        lookup_rate(fx, date(2024, 3, 15), "IDR", "USD")  # never invert a pair
    assert convert(fx, D("5"), date(2024, 3, 15), "IDR", "IDR") == D("5.00")


def test_add_months_keeps_day_and_clamps():
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2024, 1, 31), 2) == date(2024, 3, 31)
    assert add_months(date(2025, 12, 15), 1) == date(2026, 1, 15)


def test_series_dates_monthly_anchor_does_not_drift():
    s = Series("s", "rent", "expense", "rent", "fixed", None, "e", date(2026, 1, 31), D("1"), None)
    assert s.dates(date(2026, 2, 1), date(2026, 5, 1)) == [date(2026, 2, 28), date(2026, 3, 31), date(2026, 4, 30)]


def test_series_dates_weekly_window_bounds_inclusive():
    s = Series("s", "groceries", "expense", "g", "fixed", None, "e", date(2026, 6, 1), D("1"), 7)
    ds = s.dates(date(2026, 6, 8), date(2026, 6, 22))
    assert ds == [date(2026, 6, 8), date(2026, 6, 15), date(2026, 6, 22)]


@given(st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 1, 1)), st.integers(1, 35))
def test_periodic_series_never_projects_before_start_or_after_end(last, period):
    s = Series("s", "c", "expense", "d", "fixed", None, "e", last, D("1"), period)
    start, end = last, date.fromordinal(last.toordinal() + 90)
    ds = s.dates(start, end)
    assert all(start <= d <= end for d in ds)
    assert ds == sorted(ds) and len(ds) == len(set(ds))
