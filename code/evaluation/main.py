#!/usr/bin/env python3
"""Evaluation runner.

    python3 code/evaluation/main.py            # samples regression + adversarial cases + report
    python3 code/evaluation/main.py --output output.csv   # also validate a full-dataset output

Writes code/evaluation/reports/evaluation_report.json and .md. Never calls a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from buyorwait.atomic import atomic_write  # noqa: E402
from buyorwait.loaders import load_dataset  # noqa: E402
from buyorwait.output import write_csv  # noqa: E402
from buyorwait.pipeline import run  # noqa: E402
from score import score  # noqa: E402
from validate_output import validate  # noqa: E402

REPORTS = os.path.join(HERE, "reports")
GOLD = os.path.join(HERE, "golden", "sample_golden.csv")
SNAPSHOT = os.path.join(HERE, "regression", "samples_snapshot.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--output", default=None, help="full-dataset output.csv to validate")
    ap.add_argument("--update-snapshot", action="store_true", help="refresh the regression snapshot")
    a = ap.parse_args(argv)
    os.makedirs(REPORTS, exist_ok=True)
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # 1. samples: run, validate, score against the 25 official solved rows
    ds = load_dataset(a.dataset, "sample_requests.csv")
    t0 = time.time()
    res = run(ds, use_model=False, cache_path=None)
    pred = os.path.join(REPORTS, "samples_output.csv")
    write_csv(pred, res.rows)
    report["samples_runtime_s"] = round(time.time() - t0, 2)
    report["samples_contract"] = validate(pred, a.dataset, "sample_requests.csv")
    sc = score(pred, GOLD)
    report["samples_score"] = sc["summary"]
    report["samples_rows"] = sc["rows"]
    report["samples_first_divergence"] = sc["first_divergence"]
    report["samples_drawdown"] = _drawdown_audit(ds, res, GOLD)

    # 2. regression snapshot: behaviour must not drift silently
    if a.update_snapshot or not os.path.exists(SNAPSHOT):
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        write_csv(SNAPSHOT, res.rows)
        report["snapshot"] = "updated"
    else:
        snap = score(pred, SNAPSHOT)
        drift = [r for r in snap["rows"] if r[1] != "OK"]
        report["snapshot"] = {"drifted_rows": drift}

    # 3. adversarial / property cases through the whole pipeline
    from adversarial.cases import run_all
    adv = run_all()
    report["adversarial"] = {"passed": sum(1 for _, ok, _ in adv if ok), "total": len(adv),
                             "failures": [(n, m) for n, ok, m in adv if not ok]}

    # 4. optional full output validation
    if a.output:
        report["full_output_contract"] = validate(a.output, a.dataset, "requests.csv")

    atomic_write(os.path.join(REPORTS, "evaluation_report.json"), lambda fh: json.dump(report, fh, indent=1))
    _write_md(report)
    ok = report["samples_contract"]["ok"] and report["adversarial"]["passed"] == report["adversarial"]["total"] \
        and (a.output is None or report["full_output_contract"]["ok"])
    print(json.dumps({k: report[k] for k in ("samples_score", "adversarial", "snapshot")}, indent=1, default=str))
    print("contract ok:", report["samples_contract"]["ok"], "| overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _drawdown_audit(ds, res, gold_path):
    """Reference vs forecast drawdown (opening - lowest projected balance) for every sample.

    amount_safe_to_pay = opening - minimum - drawdown, so when the reference amount is below the
    request the reference drawdown is known exactly and the forecast error can be isolated.
    """
    from decimal import Decimal
    import csv as _csv
    with open(gold_path, newline="", encoding="utf-8") as fh:
        gold = {r["request_id"]: r for r in _csv.DictReader(fh)}
    out = {}
    for req in ds.requests:
        dec = res.decisions.get(req.request_id)
        if dec is None:  # fallback row: the engine raised, there is no ledger to audit
            out[req.request_id] = {"fallback": res.errors.get(req.request_id, "no decision")}
            continue
        L = dec.ledger
        p = L.profile
        safe_ref = Decimal(gold[req.request_id]["amount_safe_to_pay"])
        ours = L.opening_balance - dec.min_projected_balance
        entry = {"forecast_drawdown": str(ours), "horizon_end": L.horizon_end.isoformat()}
        if safe_ref < req.requested_amount:
            ref = p.current_available_balance - p.minimum_balance_to_keep - safe_ref
            entry["reference_drawdown"] = str(ref)
            entry["rel_error"] = f"{float((ours - ref) / ref):+.2%}" if ref else "n/a"
        else:
            entry["reference_drawdown"] = f"<= {p.current_available_balance - p.minimum_balance_to_keep - safe_ref} (full amount safe)"
        out[req.request_id] = entry
    return out


def _write_md(report):
    lines = ["# Evaluation report", "", f"Generated: {report['generated_at']}", "",
             "## Sample regression (25 official solved requests)", ""]
    for k, v in report["samples_score"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", f"Contract validation: {'OK' if report['samples_contract']['ok'] else 'FAIL'}"]
    for p in report["samples_contract"]["problems"]:
        lines.append(f"- {p}")
    lines += ["", "## Per-request differences vs golden (first divergence in column order first)", ""]
    for rid, d in report["samples_rows"]:
        lines.append(f"- {rid}: {d}")
    lines += ["", "## Forecast drawdown vs reference drawdown", "",
              "| request | reference drawdown | forecast drawdown | rel. error | horizon end |", "|---|---|---|---|---|"]
    for rid, e in report.get("samples_drawdown", {}).items():
        lines.append(f"| {rid} | {e['reference_drawdown']} | {e['forecast_drawdown']} | {e.get('rel_error', '')} | {e['horizon_end']} |")
    adv = report["adversarial"]
    lines += ["", f"## Adversarial / property cases: {adv['passed']}/{adv['total']} passed", ""]
    for n, m in adv["failures"]:
        lines.append(f"- FAIL {n}: {m}")
    if "full_output_contract" in report:
        c = report["full_output_contract"]
        lines += ["", f"## Full output contract: {'OK' if c['ok'] else 'FAIL'} ({c.get('rows')} rows / {c.get('expected_rows')} expected)", ""]
        for p in c["problems"][:100]:
            lines.append(f"- {p}")
    atomic_write(os.path.join(REPORTS, "evaluation_report.md"), lambda fh: fh.write("\n".join(lines) + "\n"))


if __name__ == "__main__":
    sys.exit(main())
