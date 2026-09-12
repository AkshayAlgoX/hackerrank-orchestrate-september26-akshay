"""Deterministic classification of income events by description.

The dataset uses a closed vocabulary of income descriptions. Classes:
  payroll          regular employer salary; projected monthly on the observed payday
  payroll_terminal final/previous-employer payroll; nothing follows it
  variable_income  freelance / contract / platform payouts; projected only when history shows a
                   regular cadence and no evidence marks it unconfirmed or ended
  never            bonuses, commissions, arrears, prizes, refunds, reimbursements: counted only
                   when they settle (challenge rule), never projected
"""
from __future__ import annotations

import re

TERMINAL = re.compile(r"\bfinal\b|previous employer|last (salary|payroll)", re.I)
NEVER = re.compile(r"commission|bonus|arrears|prize|windfall|reimbursement|refund|proceeds|seasonal contract|peak-season", re.I)
VARIABLE = re.compile(r"payout|earnings|invoice|milestone|project payment|contract payment|retainer|independent work|freelance", re.I)
PAYROLL = re.compile(r"payroll|salary|household (salary|income)|wages|assignment pay", re.I)


def income_class(description: str) -> str:
    d = description or ""
    if NEVER.search(d):
        return "never"
    if TERMINAL.search(d):
        return "payroll_terminal"
    if VARIABLE.search(d):
        return "variable_income"
    if PAYROLL.search(d):
        return "payroll"
    return "never"


LIFECYCLE_ONE_OFF = re.compile(
    r"authori[sz]ation|reversal|reversed|duplicate|awaiting refund|original card charge|"
    r"settled card purchase|reimbursable|work expense|retry|failed",
    re.I,
)


def is_lifecycle_one_off(description: str) -> bool:
    """Expense rows that belong to a card/refund/retry lifecycle, never a spending pattern."""
    return bool(LIFECYCLE_ONE_OFF.search(description or ""))
