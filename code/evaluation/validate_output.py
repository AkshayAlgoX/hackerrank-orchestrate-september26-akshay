#!/usr/bin/env python3
"""Rigorous validator for a submission ``output.csv``.

    python3 code/evaluation/validate_output.py --out output.csv [--requests dataset/requests.csv]
    python3 code/evaluation/validate_output.py --out output.csv --strict      # warnings fail too
    python3 code/evaluation/validate_output.py --out output.csv --no-recompute  # skip the 90-day check

Layers, cheapest first:

1. **File/serialization** - byte-level hygiene (BOM, CRLF, lone CR, undecodable bytes,
   missing trailing newline) and cell-level hygiene (``NaN``/``Infinity`` tokens, scientific
   notation, thousands separators, embedded newlines, stray whitespace, ``None``/``null``
   leakage, control characters).
2. **Shape** - exact header text and order; exactly one row for every ``request_id`` in the
   requests file; no missing, duplicate, or extra ids; every id present in the dataset.
3. **Row contract** - delegated to :func:`buyorwait.output.validate_row`, the engine's own
   contract checker, so the validator and the writer can never disagree. The call is wrapped:
   a malformed row is reported as a problem instead of crashing the validator.
4. **Independent consistency** - calendar-validity of every date, ``amount_safe_to_pay``
   bounds and precision, spending-change targets that actually exist for the right user,
   and ``earliest_date_for_full_payment`` not predating the request.
5. **Semantic invariants** (``--recompute``, default on) - the recommended plan is replayed
   through the engine's own 90-day forecast and must never breach ``minimum_balance_to_keep``.

Read-only: nothing here rewrites the file it inspects.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from datetime import date
from decimal import Decimal, InvalidOperation

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, CODE)

from buyorwait.loaders import load_dataset  # noqa: E402
from buyorwait.output import COLUMNS, validate_row  # noqa: E402

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PLAN_ITEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}):(\d+(?:\.\d{1,2})?)$")
# Plain non-negative decimal, at most two fraction digits: no exponent, no sign,
# no thousands separator, no leading zeros.
PLAIN_AMOUNT_RE = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d{1,2})?$")
NON_FINITE = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
NULLISH = {"none", "null", "nil", "nan", "na", "n/a"}
# `payment_plan` and `spending_changes_needed` use the literal token `none` by spec.
NONE_IS_LEGAL = {"payment_plan", "spending_changes_needed"}
MAX_EXPLANATION = 1000


def validate(out_path: str, dataset_dir: str, requests_file: str, recompute: bool = True) -> dict:
    """Validate ``out_path``. Returns ``{ok, rows, expected_rows, problems, warnings}``."""
    problems: list = []
    warnings: list = []

    raw = _read_bytes(out_path, problems)
    if raw is None:
        return _result(problems, warnings, 0, None)
    _check_bytes(raw, out_path, problems, warnings)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"file is not valid UTF-8: {exc}")
        return _result(problems, warnings, 0, None)

    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        all_rows = list(reader)
    except csv.Error as exc:
        problems.append(f"CSV parse error: {exc}")
        return _result(problems, warnings, 0, None)

    if not all_rows:
        problems.append("file is empty (no header row)")
        return _result(problems, warnings, 0, None)

    header, rows = all_rows[0], all_rows[1:]
    if header != COLUMNS:
        problems.append(f"header mismatch: expected {COLUMNS}, got {header}")
        if sorted(header) == sorted(COLUMNS):
            problems.append("header has the right columns but the wrong order")
        return _result(problems, warnings, len(rows), None)
    if len(all_rows) - 1 != len(rows):  # pragma: no cover - csv.reader is consistent
        problems.append("row count mismatch")

    try:
        ds = load_dataset(dataset_dir, requests_file)
    except Exception as exc:  # dataset unusable => cannot check ids; report, do not crash
        problems.append(f"could not load {requests_file} from {dataset_dir}: {exc}")
        return _result(problems, warnings, len(rows), None)

    req_by_id = {r.request_id: r for r in ds.requests}
    seen = _check_rows(rows, req_by_id, ds, requests_file, problems, warnings)

    missing = [rid for rid in req_by_id if rid not in seen]
    if missing:
        problems.append(f"missing {len(missing)} request ids, e.g. {sorted(missing)[:5]}")
    unknown = [rid for rid in seen if rid not in req_by_id]
    if unknown:
        problems.append(f"{len(unknown)} row ids are not in {requests_file}, e.g. {sorted(unknown)[:5]}")

    if recompute:
        rows_by_id = {rid: dict(zip(COLUMNS, rows[line - 2]))
                      for rid, line in seen.items() if len(rows[line - 2]) == len(COLUMNS)}
        _recompute_invariants(ds, rows_by_id, problems)
    return _result(problems, warnings, len(rows), len(req_by_id))


# ---------------------------------------------------------------------------------------
# layer 1: bytes and cells
# ---------------------------------------------------------------------------------------

def _read_bytes(path: str, problems: list):
    if not os.path.exists(path):
        problems.append(f"{path}: does not exist")
        return None
    if os.path.isdir(path):
        problems.append(f"{path}: is a directory")
        return None
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        problems.append(f"{path}: cannot read ({exc})")
        return None


def _check_bytes(raw: bytes, path: str, problems: list, warnings: list) -> None:
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("file starts with a UTF-8 BOM; the header row will not match")
    if b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""):
        # Every file the organizers ship (dataset/output.csv, sample_requests.csv) and the
        # golden CSV use LF. CRLF parses identically under csv.reader/pandas but breaks any
        # grader that splits on "\n" directly, so it is reported rather than assumed safe.
        warnings.append("file uses CRLF line endings; every organizer-supplied CSV "
                        "(dataset/output.csv, the golden file) uses LF")
    elif b"\r" in raw:
        problems.append("file contains a lone CR character")
    if raw and not raw.endswith(b"\n"):
        warnings.append("file does not end with a newline")
    if not raw.strip():
        problems.append(f"{path}: file has no content")


def _check_cells(row: dict, line: int, problems: list, warnings: list) -> None:
    for col in COLUMNS:
        cell = row[col]
        if cell != cell.strip():
            problems.append(f"line {line}: {col} has leading/trailing whitespace")
        if "\n" in cell or "\r" in cell:
            problems.append(f"line {line}: {col} contains a line break")
        if any(ord(ch) < 32 and ch not in "\t" for ch in cell):
            problems.append(f"line {line}: {col} contains a control character")
        bare = cell.strip().lower()
        if bare in NON_FINITE:
            problems.append(f"line {line}: {col} is not finite ({cell.strip()!r})")
        elif bare in NULLISH and col not in NONE_IS_LEGAL:
            problems.append(f"line {line}: {col} contains the literal {cell.strip()!r}")

    amount = row["amount_safe_to_pay"].strip()
    if amount and not PLAIN_AMOUNT_RE.match(amount):
        problems.append(f"line {line}: amount_safe_to_pay {amount!r} is not a plain non-negative "
                        f"amount with at most two decimals")
    elif amount:
        try:
            if Decimal(amount) < 0:
                problems.append(f"line {line}: amount_safe_to_pay is negative")
        except InvalidOperation:
            problems.append(f"line {line}: amount_safe_to_pay {amount!r} is not numeric")

    earliest = row["earliest_date_for_full_payment"].strip()
    if earliest and not _valid_date(earliest):
        problems.append(f"line {line}: earliest_date_for_full_payment {earliest!r} is not a valid date")

    plan = row["payment_plan"].strip()
    if plan and plan != "none":
        for item in plan.split("|"):
            m = PLAN_ITEM_RE.match(item)
            if m and not _valid_date(m.group(1)):
                problems.append(f"line {line}: payment_plan item {item!r} is not a valid calendar date")

    text = row["decision_explanation"]
    if len(text) > MAX_EXPLANATION:
        warnings.append(f"line {line}: decision_explanation is {len(text)} chars (>{MAX_EXPLANATION})")


def _valid_date(value: str) -> bool:
    if not DATE_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------------------
# layers 2-4: shape, row contract, independent consistency
# ---------------------------------------------------------------------------------------

def _check_rows(rows, req_by_id, ds, requests_file, problems, warnings) -> dict:
    """Check every row; returns ``{request_id: physical_line_number}`` for unique ids."""
    seen = {}
    for i, r in enumerate(rows, start=2):
        if len(r) != len(COLUMNS):
            problems.append(f"line {i}: {len(r)} columns, expected {len(COLUMNS)}")
            continue
        row = dict(zip(COLUMNS, r))
        rid = row["request_id"]
        _check_cells(row, i, problems, warnings)
        if rid in seen:
            problems.append(f"{rid}: duplicate row (also on line {seen[rid]})")
        else:
            seen[rid] = i
        req = req_by_id.get(rid)
        if req is None:
            problems.append(f"{rid}: not in {requests_file}")
            continue
        options = ds.options_by_request.get(rid, [])
        try:
            for err in validate_row(row, req, options):
                problems.append(f"{rid}: {err}")
        except Exception as exc:  # a malformed row must not crash the validator
            problems.append(f"{rid}: contract check raised {type(exc).__name__}: {exc}")
        _check_consistency(row, req, ds, problems, warnings)
        if not row["decision_explanation"].strip():
            problems.append(f"{rid}: empty explanation")
        else:
            try:
                _check_explanation(row, req, ds, problems)
            except Exception as exc:
                warnings.append(f"{rid}: explanation check skipped ({type(exc).__name__}: {exc})")
    return seen


def _check_consistency(row, req, ds, problems, warnings) -> None:
    """Checks the engine's own contract checker does not make, kept rule-free."""
    earliest = row["earliest_date_for_full_payment"].strip()
    if earliest and _valid_date(earliest):
        if date.fromisoformat(earliest) < req.request_date:
            problems.append(f"{req.request_id}: earliest_date_for_full_payment {earliest} "
                            f"is before request_date {req.request_date.isoformat()}")
    if row["affordability_status"] == "not_affordable" and earliest:
        warnings.append(f"{req.request_id}: not_affordable but earliest_date_for_full_payment is set")

    changes = row["spending_changes_needed"].strip()
    if changes and changes != "none":
        for part in changes.split("|"):
            event_id = _change_event_id(part)
            if event_id is None:
                continue  # grammar errors are already reported by validate_row
            event = ds.events_by_id.get(event_id)
            if event is None:
                problems.append(f"{req.request_id}: spending change references unknown event {event_id!r}")
            elif event.user_id != req.user_id:
                problems.append(f"{req.request_id}: spending change targets {event_id} owned by "
                                f"{event.user_id}, not {req.user_id}")


def _change_event_id(part: str):
    if part.startswith("stop:"):
        return part[5:] or None
    if part.startswith("reduce_to:"):
        body = part[len("reduce_to:"):]
        return body.rsplit(":", 1)[0] or None
    return None


# ---------------------------------------------------------------------------------------
# layer 5: replay the recommended plan through the engine's forecast
# ---------------------------------------------------------------------------------------

def _check_explanation(row, req, ds, problems) -> None:
    """Explanation must mention the home currency and the plan's key amount/date facts."""
    cur = ds.profiles[req.user_id].home_currency
    text = row["decision_explanation"]
    if cur not in text:
        problems.append(f"{req.request_id}: explanation lacks home currency {cur}")
    plan = row["payment_plan"]
    if plan != "none":
        first_amt = Decimal(plan.split("|")[0].split(":")[1])
        if f"{first_amt:,.2f}".rstrip("0").rstrip(".") not in text.replace(",", "").replace(".00", "") \
                and f"{first_amt:,}" not in text and f"{first_amt:,.2f}" not in text:
            problems.append(f"{req.request_id}: explanation does not state the first payment amount")
    changes = row["spending_changes_needed"]
    if changes != "none" and not re.search(r"\b(stop|reduce)\b", text, re.I):
        problems.append(f"{req.request_id}: explanation does not mention the spending change")


def _recompute_invariants(ds, rows, problems) -> None:
    from buyorwait.extraction import gather_evidence
    from buyorwait.forecast import is_safe, project_flows
    from buyorwait.ledger import build_ledger
    try:
        bundle = gather_evidence(ds, use_model=False, cache_path=None)
    except Exception as exc:
        problems.append(f"could not rebuild evidence for the 90-day check: {exc}")
        return
    for req in ds.requests:
        row = rows.get(req.request_id)
        if row is None or row["payment_plan"] == "none":
            continue
        try:
            L = build_ledger(ds, req.user_id, req.request_date, bundle.for_user(req.user_id))
            payments = []
            for item in row["payment_plan"].split("|"):
                d, a = item.split(":")
                payments.append((date.fromisoformat(d), Decimal(a)))
            changes = []
            if row["spending_changes_needed"] != "none":
                from buyorwait.spending import candidate_changes
                wanted = row["spending_changes_needed"].split("|")
                for c in candidate_changes(L):
                    if c.render() in wanted:
                        changes.append(c)
                if len(changes) != len(wanted):
                    problems.append(f"{req.request_id}: spending change not permitted for this user/series: {wanted}")
            if not is_safe(L, project_flows(L, changes), payments):
                problems.append(f"{req.request_id}: recommended plan breaches minimum balance in the 90-day forecast")
        except Exception as exc:
            problems.append(f"{req.request_id}: 90-day replay raised {type(exc).__name__}: {exc}")


def _result(problems, warnings, rows, expected):
    return {"ok": not problems, "rows": rows, "expected_rows": expected,
            "problems": problems, "warnings": warnings}


def line_endings(path: str) -> str:
    """Report the file's line terminator: ``lf``, ``crlf``, ``mixed`` or ``unknown``."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return "unknown"
    crlf, lf = raw.count(b"\r\n"), raw.count(b"\n")
    if crlf and lf > crlf:
        return "mixed"
    if crlf:
        return "crlf"
    return "lf" if lf else "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate a submission output.csv against the contract")
    ap.add_argument("--out", default=os.path.join(ROOT, "output.csv"))
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--requests", default="requests.csv")
    ap.add_argument("--no-recompute", action="store_true", help="skip the 90-day plan replay")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--json", default=None, help="also write the full result here")
    a = ap.parse_args(argv)
    res = validate(a.out, a.dataset, a.requests, recompute=not a.no_recompute)
    res["line_endings"] = line_endings(a.out)
    ok = res["ok"] and (not a.strict or not res["warnings"])
    print(json.dumps({k: v for k, v in res.items() if k not in ("problems", "warnings")}, indent=1))
    for p in res["problems"][:200]:
        print("  ERROR", p)
    for w in res["warnings"][:200]:
        print("  warn ", w)
    print(f"problems: {len(res['problems'])}  warnings: {len(res['warnings'])}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(res, ok=ok), fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
