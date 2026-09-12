#!/usr/bin/env python3
"""Render ``evaluation/usage_report.md`` from the usage JSON of the final full-dataset run.

    python3 code/evaluation/write_usage_report.py                        # render the report
    python3 code/evaluation/write_usage_report.py --check                # is it current and complete?
    python3 code/evaluation/write_usage_report.py --usage reports/usage_last_run.json

``code/main.py`` writes ``reports/usage_last_run.json`` at the end of every run. This tool turns
that JSON into the single file the challenge requires, containing every mandated figure:

* model provider(s) and model name(s),
* model calls,
* input tokens and output tokens,
* total and average tokens per request,
* estimated total and per-request cost,

plus per-model breakdowns, the pricing assumptions behind the estimate, and an integrity
section that proves the numbers describe the run that produced ``output.csv`` rather than an
earlier sample run. ``--check`` exits non-zero when the on-disk report is missing, stale, or
does not describe a full-dataset run.
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
from buyorwait.extraction.llm import PRICING_PER_MTOK  # noqa: E402

MARKER = "# Token usage and cost report"


def _rows_in(path: str) -> int:
    """Data rows in a CSV (header excluded); -1 when the file is unreadable."""
    try:
        with open(path, "rb") as fh:
            body = fh.read().strip()
    except OSError:
        return -1
    if not body:
        return 0
    return max(0, len(body.splitlines()) - 1)


def gather_context(usage: dict, dataset: str, output: str, usage_path: str) -> dict:
    ds_requests = os.path.join(dataset, "requests.csv")
    return {
        "dataset_dir": dataset,
        "requests_file": ds_requests,
        "requests_rows": _rows_in(ds_requests),
        "output_path": output,
        "output_rows": _rows_in(output) if output else -1,
        "usage_path": usage_path,
    }


def integrity(usage: dict, ctx: dict) -> list:
    """``(ok, message)`` pairs proving the report matches the final full-dataset run."""
    n = usage.get("requests", 0)
    provider = usage.get("provider") or "none"
    model = usage.get("model") or "none"
    checks = []

    full = ctx["requests_rows"]
    if full < 0:
        checks.append((False, f"could not read {ctx['requests_file']}"))
    elif n == full:
        checks.append((True, f"run covers the full dataset: {n} requests == {full} rows in requests.csv"))
    else:
        checks.append((False, f"run covers {n} requests but requests.csv has {full} rows - this is not "
                              f"the final full-dataset run"))

    out_rows = ctx["output_rows"]
    if out_rows < 0:
        checks.append((False, f"{ctx['output_path']}: missing or unreadable"))
    elif out_rows == full:
        checks.append((True, f"output.csv has one row per request: {out_rows}"))
    else:
        checks.append((False, f"output.csv has {out_rows} rows but requests.csv has {full}"))

    checks.append((True, f"provider recorded: {provider} (model {model})"))

    per_model = usage.get("usage", {}) or {}
    unknown = [m for m, r in per_model.items() if not r.get("pricing_known", True)]
    if unknown:
        checks.append((False, f"no pricing configured for {', '.join(unknown)}; their cost is reported as 0"))
    else:
        checks.append((True, "every model that was called has known pricing"))

    total_calls = sum(r.get("calls", 0) for r in per_model.values())
    if provider != "none" and total_calls == 0:
        checks.append((False, "a provider is configured but the run recorded zero model calls"))
    return checks


def render(usage: dict, ctx: dict | None = None) -> str:
    ctx = ctx or {}
    n = usage.get("requests", 0) or 1
    per_model = usage.get("usage", {}) or {}
    tot_calls = sum(m["calls"] for m in per_model.values())
    tot_in = sum(m["input_tokens"] for m in per_model.values())
    tot_out = sum(m["output_tokens"] for m in per_model.values())
    tot_cache = sum(m.get("cache_read_tokens", 0) for m in per_model.values())
    tot_cost = sum(m["cost_usd"] for m in per_model.values())
    tot_tokens = tot_in + tot_out
    srcs: dict = {}
    for v in (usage.get("sources") or {}).values():
        srcs[v] = srcs.get(v, 0) + 1

    checks = integrity(usage, ctx) if ctx else []
    lines = [MARKER, "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}", "",
             "This file summarises **the final full-dataset run that produced `output.csv`**.", ""]

    # --- the mandated figures, stated once, unambiguously -------------------------------
    lines += ["## Required figures", "",
              "| Requirement | Value |",
              "|---|---|",
              f"| Model provider(s) | `{usage.get('provider') or 'none'}` |",
              f"| Model name(s) | `{usage.get('model') or 'none'}`"
              + (f" (+ {', '.join(sorted(set(per_model) - {usage.get('model')}))})" if per_model else "")
              + " |",
              f"| Model calls (total) | {tot_calls} |",
              f"| Input tokens (total) | {tot_in:,} |",
              f"| Output tokens (total) | {tot_out:,} |",
              f"| Total tokens (input + output) | {tot_tokens:,} |",
              f"| Requests processed | {usage.get('requests', 0):,} |",
              f"| Total tokens per request | {tot_tokens / n:,.2f} |",
              f"| Model calls per request | {tot_calls / n:,.4f} |",
              f"| Estimated total cost (USD) | {tot_cost:,.6f} |",
              f"| Estimated cost per request (USD) | {tot_cost / n:,.8f} |", ""]

    # --- how the numbers were produced ---------------------------------------------------
    lines += ["## Architecture note", "",
              "The decision engine is deterministic (Python `Decimal` arithmetic). Models are used only as",
              "a bounded perception layer: extracting literal facts (amounts, dates, intents) from untrusted",
              "messages and images into a validated schema. Message templates are handled first by a",
              "deterministic rules classifier; a model is called only for images and for messages no rule",
              "matches. Results are cached by content hash, so a re-run makes zero calls.", "",
              f"Evidence sources: {', '.join(f'{k}={v}' for k, v in sorted(srcs.items())) or 'n/a'}", ""]

    # --- per model ------------------------------------------------------------------------
    lines += ["## Usage by model", "",
              "| Model | Calls | Input tokens | Output tokens | Total tokens | Cache-read tokens | Est. cost (USD) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for m, r in sorted(per_model.items()):
        total = r["input_tokens"] + r["output_tokens"]
        lines.append(f"| `{m}` | {r['calls']:,} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                     f"{total:,} | {r.get('cache_read_tokens', 0):,} | {r['cost_usd']:,.6f} |")
    if not per_model:
        lines.append("| _(no model calls in this run)_ | 0 | 0 | 0 | 0 | 0 | 0.000000 |")
    lines.append(f"| **Total** | **{tot_calls:,}** | **{tot_in:,}** | **{tot_out:,}** | **{tot_tokens:,}** "
                 f"| **{tot_cache:,}** | **{tot_cost:,.6f}** |")

    if per_model:
        lines += ["", "### Per-request averages by model", "",
                  "| Model | Calls/request | Input tokens/request | Output tokens/request | Cost/request (USD) |",
                  "|---|---:|---:|---:|---:|"]
        for m, r in sorted(per_model.items()):
            lines.append(f"| `{m}` | {r['calls'] / n:.4f} | {r['input_tokens'] / n:,.2f} | "
                         f"{r['output_tokens'] / n:,.2f} | {r['cost_usd'] / n:,.8f} |")

    # --- pricing ---------------------------------------------------------------------------
    lines += ["", "## Pricing assumptions (USD per 1M tokens)", ""]
    for m, (pin, pout) in PRICING_PER_MTOK.items():
        lines.append(f"- `{m}`: input {pin:.2f}, output {pout:.2f} (built-in list price)")
    lines.append("- any other model: `BUYORWAIT_LLM_PRICE_IN` / `BUYORWAIT_LLM_PRICE_OUT` from the "
                 "environment of the run")
    unknown = [m for m, r in per_model.items() if not r.get("pricing_known", True)]
    if unknown:
        lines.append(f"- **pricing was NOT configured for: {', '.join(unknown)}; their cost is reported as 0**")
    lines += ["", "Cache-read tokens are billed at a reduced rate by most providers; the estimate above",
              "conservatively prices them as regular input tokens.", ""]

    # --- integrity -------------------------------------------------------------------------
    if checks:
        lines += ["## Integrity checks", ""]
        for ok, msg in checks:
            lines.append(f"- {'PASS' if ok else '**FAIL**'} - {msg}")
        lines.append("")
        if ctx:
            lines += [f"- usage JSON: `{os.path.relpath(ctx['usage_path'], ROOT)}`",
                      f"- dataset: `{os.path.relpath(ctx['requests_file'], ROOT)}` "
                      f"({ctx['requests_rows']} rows)",
                      f"- predictions: `{os.path.relpath(ctx['output_path'], ROOT)}` "
                      f"({ctx['output_rows']} rows)", ""]

    if usage.get("rejected_evidence"):
        lines += ["## Rejected evidence (failed schema validation)", ""]
        lines += [f"- {r}" for r in usage["rejected_evidence"]]
        lines.append("")

    fp = usage.get("engine_fingerprint") or {}
    lines += ["## Reproducibility fingerprint", ""]
    if fp.get("files"):
        lines += [f"Combined {fp.get('algorithm', 'sha256')} over the engine files: `{fp.get('combined')}`", "",
                  "| File | SHA-256 |", "|---|---|"]
        lines += [f"| `{k}` | `{v}` |" for k, v in sorted(fp["files"].items())]
        lines.append("")
        if fp.get("current_combined") and fp["current_combined"] != fp.get("combined"):
            lines += ["**Warning:** the engine files on disk no longer match the run "
                      f"(now `{fp['current_combined']}`); regenerate output.csv before submitting.", ""]
    else:
        lines += ["Not recorded by this run (older usage JSON); re-run code/main.py.", ""]
    if usage.get("provider_errors"):
        lines += ["## Provider errors (calls that fell back to rules/golden)", ""]
        lines += [f"- {e}" for e in usage["provider_errors"]]
        lines.append("")
    if usage.get("fallback_rows"):
        lines += ["## Fallback rows (requests whose evaluation raised)", ""]
        lines += [f"- {k}: {v}" for k, v in usage["fallback_rows"].items()]
        lines.append("")

    lines += ["## Reproducing this run", "",
              "```bash",
              "python3 code/main.py            # writes output.csv and reports/usage_last_run.json",
              "python3 code/evaluation/write_usage_report.py",
              "python3 code/evaluation/validate_output.py --out output.csv",
              "```", "",
              "No API keys or credentials are stored in this repository or in this report; the",
              "provider, model, endpoint and key are read from environment variables at run time.", ""]
    return "\n".join(lines)


def check_report(usage_path: str = None, report_path: str = None, dataset: str = None,
                 output: str = None) -> dict:
    """Is the on-disk report present, current, and about the final full-dataset run?"""
    usage_path = usage_path or os.path.join(HERE, "reports", "usage_last_run.json")
    report_path = report_path or os.path.join(HERE, "usage_report.md")
    dataset = dataset or os.path.join(ROOT, "dataset")
    output = output or os.path.join(ROOT, "output.csv")
    problems = []
    if not os.path.exists(usage_path):
        return {"ok": False, "problems": [f"{usage_path}: no usage JSON; run code/main.py first"]}
    try:
        with open(usage_path, encoding="utf-8") as fh:
            usage = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "problems": [f"{usage_path}: unreadable ({exc})"]}
    ctx = gather_context(usage, dataset, output, usage_path)
    for ok, msg in integrity(usage, ctx):
        if not ok:
            problems.append(msg)
    fp = usage.get("engine_fingerprint") or {}
    if fp.get("combined"):
        try:
            from buyorwait.fingerprint import engine_fingerprint
            now = engine_fingerprint()["combined"]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"could not recompute the engine fingerprint: {exc}")
        else:
            if now != fp["combined"]:
                problems.append("engine files changed since the run that produced output.csv "
                                f"(run {fp['combined'][:12]}..., now {now[:12]}...); regenerate before submitting")
    if not os.path.exists(report_path):
        problems.append(f"{report_path}: missing (run this tool without --check)")
    else:
        with open(report_path, encoding="utf-8") as fh:
            on_disk = fh.read()
        expected = _strip_generated(render(usage, ctx))
        if _strip_generated(on_disk) != expected:
            problems.append(f"{report_path}: out of date with {os.path.basename(usage_path)}; "
                            f"re-run this tool to regenerate it")
    return {"ok": not problems, "problems": problems, "requests": usage.get("requests"),
            "provider": usage.get("provider"), "model": usage.get("model")}


def _strip_generated(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("Generated:"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render or verify evaluation/usage_report.md")
    ap.add_argument("--usage", default=os.path.join(HERE, "reports", "usage_last_run.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "usage_report.md"))
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--predictions", default=os.path.join(ROOT, "output.csv"),
                    help="the output.csv this run produced (used by the integrity checks)")
    ap.add_argument("--check", action="store_true", help="verify instead of writing; exit 1 if stale")
    a = ap.parse_args(argv)
    if a.check:
        res = check_report(a.usage, a.out, a.dataset, a.predictions)
        print(json.dumps(res, indent=1, default=str))
        for p in res["problems"]:
            print("  ERROR", p)
        print("->", "PASS" if res["ok"] else "FAIL")
        return 0 if res["ok"] else 1
    with open(a.usage, encoding="utf-8") as fh:
        usage = json.load(fh)
    ctx = gather_context(usage, a.dataset, a.predictions, a.usage)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render(usage, ctx))
    bad = [m for ok, m in integrity(usage, ctx) if not ok]
    print(f"wrote {a.out}")
    for m in bad:
        print("  WARNING", m)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
