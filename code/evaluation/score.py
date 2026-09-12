#!/usr/bin/env python3
"""Score an output CSV against a golden CSV (same columns), per field.

    python3 code/evaluation/score.py --pred code/evaluation/reports/samples_output.csv \
        --gold code/evaluation/golden/sample_golden.csv [--verbose]

Exact-match fields: affordability_status, recommended_payment_method, payment_plan,
earliest_date_for_full_payment, spending_changes_needed. amount_safe_to_pay is scored
exact and within relative tolerances (1% / 5% of requested_amount) because the hidden
scorer is likely tolerance-based and the variable-spend forecast is an estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FIELDS = ["amount_safe_to_pay", "affordability_status", "recommended_payment_method", "payment_plan",
          "earliest_date_for_full_payment", "spending_changes_needed"]


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["request_id"]: r for r in csv.DictReader(fh)}


def score(pred_path: str, gold_path: str, verbose: bool = False) -> dict:
    pred, gold = read(pred_path), read(gold_path)
    n = len(gold)
    hits = {f: 0 for f in FIELDS}
    tol = {"1pct": 0, "5pct": 0}
    abs_rel_err = []
    rows = []
    first_divergence = {}
    for rid, g in gold.items():
        p = pred.get(rid)
        if p is None:
            rows.append((rid, "MISSING"))
            first_divergence[rid] = "missing row"
            continue
        diffs = []
        for f in FIELDS:
            if f == "amount_safe_to_pay":
                gs, ps = Decimal(g[f] or "0"), Decimal(p[f] or "0")
                req = Decimal(g.get("requested_amount") or "0") or max(gs, Decimal("1"))
                if gs == ps:
                    hits[f] += 1
                rel = abs(ps - gs) / req if req else Decimal(0)
                abs_rel_err.append(float(rel))
                if rel <= Decimal("0.01"):
                    tol["1pct"] += 1
                if rel <= Decimal("0.05"):
                    tol["5pct"] += 1
                if gs != ps:
                    diffs.append(f"{f}: {ps} vs {gs} (rel {float(rel):.3%})")
            else:
                if (p[f] or "") == (g[f] or ""):
                    hits[f] += 1
                else:
                    diffs.append(f"{f}: {p[f]!r} vs {g[f]!r}")
        rows.append((rid, "; ".join(diffs) if diffs else "OK"))
        if diffs:
            first_divergence[rid] = diffs[0]
    summary = {f: f"{hits[f]}/{n}" for f in FIELDS}
    summary["amount_safe_within_1pct"] = f"{tol['1pct']}/{n}"
    summary["amount_safe_within_5pct"] = f"{tol['5pct']}/{n}"
    summary["amount_safe_mean_rel_err"] = f"{(sum(abs_rel_err) / len(abs_rel_err)) if abs_rel_err else 0:.4f}"
    all_exact = sum(1 for _, d in rows if d == "OK")
    summary["rows_all_fields_exact"] = f"{all_exact}/{n}"
    if verbose:
        for rid, d in rows:
            print(f"{rid}: {d}")
    summary["discrete_fields_all_exact"] = f"{sum(1 for rid, d in rows if d == 'OK' or d.startswith('amount_safe_to_pay') and ';' not in d)}/{n}"
    return {"summary": summary, "rows": rows, "first_divergence": first_divergence}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default=os.path.join(HERE, "golden", "sample_golden.csv"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = score(a.pred, a.gold, a.verbose)
    print(json.dumps(res["summary"], indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
