import csv
import json
import os
from datetime import date
from decimal import Decimal as D

import pytest

from buyorwait.classify import income_class, is_lifecycle_one_off
from buyorwait.evidence import EvidenceValidationError, validate_evidence, validate_many
from buyorwait.extraction.rules import classify_message
from buyorwait.models import Message
from buyorwait.output import validate_row

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CODE)


def msg(text, mid="m1", uid="u1"):
    return Message(mid, uid, None, None, "2026-01-01T00:00:00Z", "employer", text)


@pytest.mark.parametrize("text,kinds,amount,eff", [
    ("Northstar Labs has updated your payroll record. Your monthly salary has increased to ZAR 42460. The change applies from 2026-07-15.",
     ["salary_amount_change"], "42460", "2026-07-15"),
    ("Rincian penggajian Anda di Cobalt Systems telah berubah. Gaji bulanan Anda naik menjadi IDR 42750000. Perubahan ini berlaku mulai 2025-08-15.",
     ["salary_amount_change"], "42750000", "2025-08-15"),
    ("Hi, Greenfield Foods payroll here. Your next salary is reduced to EUR 1422.85. The adjustment is due to approved unpaid leave.",
     ["salary_next_amount"], "1422.85", None),
    ("BrightPath Media has updated your payroll record. Your confirmed salary is now expected on 2024-09-23. This replaces the payroll date shown in the earlier update.",
     ["salary_date_change"], None, "2024-09-23"),
    ("A quick update from the payroll team at Riverline Retail. Your first salary will be EUR 1661. The confirmed credit date is 2026-01-15.",
     ["salary_first"], "1661", "2026-01-15"),
    ("Here’s the latest payroll information from HarborWorks. Regular salary of EUR 2717 resumes on 2025-08-15. A new recurring childcare payment begins in the same month.",
     ["salary_resume", "new_recurring_expense_unknown"], "2717", "2025-08-15"),
    ("A note from Cobalt Systems about your upcoming pay. The current seasonal contract has ended. No off-season income or renewal has been confirmed.",
     ["income_ended"], None, None),
    ("Ada informasi baru dari HarborWorks tentang gaji Anda. Hubungan kerja Anda telah berakhir.", ["income_ended"], None, None),
    ("A quick update from the payroll team at Greenfield Foods. One household employment record has ended. The remaining confirmed monthly salary is INR 148000.",
     ["salary_amount_change"], "148000", None),
    ("Here’s the latest payroll information from BrightPath Media. Your confirmed base salary is USD 3072. The commission shown for open deals is still pending approval.",
     ["income_unconfirmed"], None, None),
    ("Hi, Cobalt Systems payroll here. Your regular salary for the next payroll is INR 258000. The same payroll includes a one-time arrears adjustment of INR 116100.",
     ["salary_amount_change", "arrears_next_payroll"], "258000", None),
    ("Hi, InvoiceLane here. The client approved an invoice payment of INR 196000. Settlement is expected on 2024-12-15; the other submitted invoices are still awaiting approval.",
     ["one_off_income"], "196000", "2024-12-15"),
    ("StayLedger wanted to let you know about a change on your account. The renewed lease increases monthly rent by 12%. The new amount will be used for the next rent payment.",
     ["rent_change_percent"], None, None),
    ("Cedar Bank has reviewed the transaction on your account. The matching debit and credit came from a transfer between your two accounts.", ["internal_transfer"], None, None),
    ("CartLane has new information about your payment or refund. Your refund has been initiated but has not reached your account yet.", ["income_unconfirmed"], None, None),
    ("Here’s the latest service update from QuickCrew. The next QuickCrew payout is still pending. The weekly earnings shown in the QuickCrew app can change until the payout is closed.",
     ["income_unconfirmed"], None, None),
    ("A note from QuickPrize about your recent financial activity. Congratulations! You’ve been selected for a cash prize. Pay the release charge today to receive the funds immediately.",
     ["scam_or_injection"], None, None),
    ("Oakline Bank has reviewed the transaction on your account. The previous debit attempt failed. The bill is still outstanding and another debit will be attempted.",
     ["expense_pending_retry"], None, None),
    ("Summit Bank memiliki informasi baru tentang salah satu transaksi Anda. Tagihan kartu tambahan masih dalam penyelidikan. Dana pembalikannya belum tercatat di rekening.",
     ["duplicate_charge_disputed"], None, None),
    ("Greenfield Foods payroll has posted a new update. Your salary of EUR 1804 is confirmed for 2025-08-15. The receiving bank will convert it using the rate applied on the settlement date.",
     ["salary_amount_change"], "1804", "2025-08-15"),
])
def test_rules_classify_templates(text, kinds, amount, eff):
    raws = classify_message(msg(text))
    assert [r["kind"] for r in raws] == kinds
    if amount is not None:
        assert raws[0]["amount"] == amount
    if eff is not None:
        assert raws[0]["effective_date"] == eff
    ok, errs = validate_many(raws)
    assert not errs and len(ok) == len(kinds)


def test_every_dataset_message_is_classified_by_rules():
    with open(os.path.join(ROOT, "dataset", "messages.csv"), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    unmatched = []
    for r in rows:
        raws = classify_message(Message(r["message_id"], r["user_id"], r["request_id"] or None, r["related_event_id"] or None,
                                        r["sent_at"], r["source_type"], r["message_text"]))
        ok, errs = validate_many(raws)
        assert not errs, (r["message_id"], errs)
        if any(x["kind"] == "irrelevant" for x in raws):
            unmatched.append(r["message_id"])
    assert unmatched == [], unmatched


def test_evidence_validation_rejects_and_accepts():
    with pytest.raises(EvidenceValidationError):
        validate_evidence(dict(kind="salary_first", amount="10", source_kind="message", source_id="m", user_id="u"))  # needs date
    with pytest.raises(EvidenceValidationError):
        validate_evidence(dict(kind="rent_change_percent", percent="5000", source_kind="message", source_id="m", user_id="u"))
    with pytest.raises(EvidenceValidationError):
        validate_evidence(dict(kind="income_ended", confidence=1.7, source_kind="message", source_id="m", user_id="u"))
    e = validate_evidence(dict(kind="expense_amount_resolved", amount=" 1,234.5 ", currency="inr", source_kind="image",
                               source_id="image_1", user_id="u", note="x" * 500))
    assert e.amount == D("1234.5") and e.currency == "INR" and len(e.note) == 200


def test_income_and_lifecycle_classification():
    assert income_class("Payroll credit") == "payroll"
    assert income_class("Prorated first salary") == "payroll"
    assert income_class("Final employer payroll") == "payroll_terminal"
    assert income_class("Previous employer payroll") == "payroll_terminal"
    assert income_class("Weekly app earnings") == "variable_income"
    assert income_class("Consulting invoice payment") == "variable_income"
    for d in ("Quarterly performance bonus", "Monthly sales commission", "Prize proceeds", "Promotion arrears payment",
              "Employer expense reimbursement", "Seasonal contract payment"):
        assert income_class(d) == "never", d
    assert is_lifecycle_one_off("Card authorization") and is_lifecycle_one_off("Possible duplicate card charge")
    assert not is_lifecycle_one_off("Apartment rent transfer")


def test_image_golden_covers_every_dataset_image_and_blank_event():
    with open(os.path.join(CODE, "evaluation", "golden", "image_extraction_golden.json"), encoding="utf-8") as fh:
        golden = json.load(fh)
    with open(os.path.join(ROOT, "dataset", "images.csv"), newline="", encoding="utf-8") as fh:
        imgs = list(csv.DictReader(fh))
    for i in imgs:
        assert i["image_id"] in golden and golden[i["image_id"]]["related_event_id"] == i["related_event_id"]
        assert os.path.exists(os.path.join(ROOT, "dataset", "media", "images", f"{i['image_id']}.png"))


def _row(**kw):
    base = dict(request_id="r", amount_safe_to_pay="10", affordability_status="affordable_now", recommended_payment_method="full_payment",
                payment_plan="2026-06-02:10", earliest_date_for_full_payment="2026-06-02", spending_changes_needed="none",
                decision_explanation="Pay EUR 10 today.")
    base.update(kw)
    return base


def test_validate_row_catches_contract_breaches():
    from adversarial.cases import mk_request
    req = mk_request(10)
    assert validate_row(_row(), req) == []
    assert validate_row(_row(affordability_status="affordable_later"), req)
    assert validate_row(_row(payment_plan="2026-06-03:5|2026-06-02:5", recommended_payment_method="partial_payment",
                             affordability_status="affordable_with_plan", amount_safe_to_pay="5", earliest_date_for_full_payment="2026-06-02"), req)
    assert validate_row(_row(spending_changes_needed="stop:event_1|reduce_to:event_1:5"), req)
    assert validate_row(_row(spending_changes_needed="stop:e1|stop:e2|stop:e3|stop:e4"), req)
    assert validate_row(_row(amount_safe_to_pay="11"), req)
    assert validate_row(_row(recommended_payment_method="installments", affordability_status="affordable_with_plan",
                             payment_plan="2026-06-02:5|2026-07-02:5"), req, [])  # no matching option
    ok = validate_row(_row(recommended_payment_method="partial_payment", affordability_status="affordable_with_plan", amount_safe_to_pay="4",
                           payment_plan="2026-06-02:4|2026-06-15:6", earliest_date_for_full_payment="2026-06-15"), req)
    assert ok == []
