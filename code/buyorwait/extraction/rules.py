"""Deterministic message classifier for the dataset's closed template vocabulary.

Returns raw evidence dicts (validated later). English and Indonesian variants are
handled. Anything unmatched is returned as kind="irrelevant" with a low confidence so
the caller may escalate it to a model.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional

from ..models import Message

AMOUNT = re.compile(r"\b(INR|ZAR|IDR|USD|EUR)\s?([\d][\d,]*(?:\.\d+)?)")
ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
LONG_DATE = re.compile(r"\b(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December) (\d{4})\b")
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}


def _amounts(t: str):
    return [(m.group(1), m.group(2).replace(",", "")) for m in AMOUNT.finditer(t)]


def _dates(t: str) -> List[str]:
    out = [m.group(1) for m in ISO.finditer(t)]
    for m in LONG_DATE.finditer(t):
        out.append(date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))).isoformat())
    return out


def _base(msg: Message, kind: str, **kw) -> Dict:
    d = dict(source_kind="message", source_id=msg.message_id, user_id=msg.user_id, request_id=msg.request_id,
             related_event_id=msg.related_event_id, sent_at=msg.sent_at, kind=kind, confidence=1.0)
    d.update(kw)
    return d


# (pattern, handler) pairs evaluated in order; first match wins unless the handler returns several.
def classify_message(msg: Message) -> List[Dict]:
    t = msg.message_text
    tl = t.lower()
    amts = _amounts(t)
    dates = _dates(t)
    a0 = amts[0] if amts else (None, None)
    d0 = dates[0] if dates else None
    out: List[Dict] = []

    def ev(kind, **kw):
        out.append(_base(msg, kind, **kw))

    # scam / embedded instruction
    if re.search(r"pay the (release|processing) charge|bayar biaya (pencairan|pemrosesan)|selected for a cash prize|terpilih untuk menerima hadiah", tl):
        ev("scam_or_injection", note="unsolicited prize demanding a fee; ignored")
        return out
    # income ended
    if re.search(r"seasonal contract has ended|kontrak musiman saat ini telah berakhir|employment has ended|hubungan kerja anda telah berakhir", tl):
        ev("income_ended")
        return out
    # household record ended -> remaining salary
    if re.search(r"household employment record has ended|sumber pendapatan kerja rumah tangga telah berakhir", tl) and amts:
        ev("salary_amount_change", amount=a0[1], currency=a0[0], note="remaining confirmed monthly salary")
        return out
    # salary increase from a date
    if re.search(r"salary has increased to|gaji bulanan anda naik menjadi", tl) and amts:
        ev("salary_amount_change", amount=a0[1], currency=a0[0], effective_date=d0)
        return out
    # next salary reduced / temporary pay
    if re.search(r"next salary is reduced to|temporary monthly pay is|gaji bulanan sementara anda adalah", tl) and amts:
        ev("salary_next_amount", amount=a0[1], currency=a0[0])
        return out
    # payday moved
    if re.search(r"confirmed salary is now expected on|kini diperkirakan masuk pada", tl) and d0:
        ev("salary_date_change", effective_date=d0)
        return out
    # first salary (new job)
    if re.search(r"first salary|gaji pertama", tl) and amts and d0:
        ev("salary_first", amount=a0[1], currency=a0[0], effective_date=d0)
        return out
    # salary resumes + new recurring childcare (amount unknown)
    if re.search(r"regular salary of .* resumes on", tl) and amts and d0:
        ev("salary_resume", amount=a0[1], currency=a0[0], effective_date=d0)
        ev("new_recurring_expense_unknown", note="childcare payment announced without an amount")
        return out
    # base salary restated, commission pending. The operative fact is that the commission is not
    # cash. The restated base figure is not adopted: it is a message figure that conflicts with the
    # settled payroll history, and the conflict rules prefer the settled record and, failing that,
    # the financially safer (lower) reading. The figure is kept in the note for audit only.
    if re.search(r"confirmed base salary is|gaji pokok yang dikonfirmasi adalah", tl) and amts:
        ev("income_unconfirmed", note=f"commission pending; base salary restated as {a0[0]} {a0[1]}")
        return out
    # regular salary + one-time arrears in the same payroll
    if re.search(r"one-time arrears adjustment|penyesuaian tunggakan satu kali", tl) and len(amts) >= 2:
        ev("salary_amount_change", amount=a0[1], currency=a0[0])
        ev("arrears_next_payroll", amount=amts[1][1], currency=amts[1][0])
        return out
    # confirmed salary on a date, converted at settlement rate (foreign payroll)
    if re.search(r"salary of [A-Z]{3} [\d,.]+ is confirmed for|gaji sebesar [A-Z]{3} [\d,.]+ dikonfirmasi untuk|confirmed a [A-Z]{3} [\d,.]+ salary credit for", tl.replace("usd", "USD").replace("eur", "EUR").replace("inr", "INR").replace("zar", "ZAR").replace("idr", "IDR")) and amts and d0:
        ev("salary_amount_change", amount=a0[1], currency=a0[0], effective_date=d0)
        if re.search(r"receipt (has|contains) the final", tl):
            ev("settlement_confirmation")
        return out
    # bonus / commission / payout / prize / refund not yet cash
    if re.search(r"bonus is still subject|bonus kuartalan anda masih|payout is still pending|masih tertunda|"
                 r"prize claim has been verified|klaim hadiah anda sudah diverifikasi|refund has been initiated|"
                 r"pengembalian dana sudah diproses|refund is still processing", tl):
        ev("income_unconfirmed")
        return out
    # approved invoice with expected settlement date
    if re.search(r"approved an invoice payment of|menyetujui pembayaran faktur sebesar", tl) and amts and d0:
        ev("one_off_income", amount=a0[1], currency=a0[0], effective_date=d0, note="approved invoice")
        return out
    # rent increase
    if re.search(r"increases monthly rent by|menaikkan biaya sewa bulanan sebesar", tl):
        m = PERCENT.search(t)
        if m:
            ev("rent_change_percent", percent=m.group(1))
            return out
    if re.search(r"transfer between your two accounts|transfer antara dua rekening anda", tl):
        ev("internal_transfer")
        return out
    if re.search(r"market value has increased|value of the investment has fallen|nilai investasi yang ditampilkan telah turun", tl):
        ev("investment_value_change")
        return out
    if re.search(r"investment sale have settled|hasil penjualan investasi anda sudah masuk|prize proceeds have reached|"
                 r"reimbursement for your earlier work expense|penggantian atas biaya kerja|was paid in|payment was received on|"
                 r"receipt (has|contains) the final", tl):
        ev("settlement_confirmation")
        return out
    if re.search(r"previous debit attempt failed", tl):
        ev("expense_pending_retry")
        return out
    if re.search(r"extra card charge is still being investigated|tagihan kartu tambahan masih dalam penyelidikan", tl):
        ev("duplicate_charge_disputed")
        return out
    if re.search(r"charged in a foreign currency|dikenakan dalam mata uang asing", tl):
        ev("fx_settlement_note")
        return out
    if re.search(r"two separate card accounts", tl):
        ev("separate_card_minimums")
        return out
    if re.search(r"gaji rutin untuk penggajian berikutnya sudah dikonfirmasi", tl):
        ev("settlement_confirmation", note="regular salary confirmed; payslip carries the figures")
        return out
    ev("irrelevant", confidence=0.2, note="no template matched")
    return out
