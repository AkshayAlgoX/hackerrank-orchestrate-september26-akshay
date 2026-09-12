#!/usr/bin/env python3
"""One-command evaluation runner for the whole submission.

    python3 code/evaluation/run_all.py                    # every stage
    python3 code/evaluation/run_all.py --stages pytest     # just the unit suite
    python3 code/evaluation/run_all.py --regenerate        # rebuild output.csv first
    python3 code/evaluation/run_all.py --with-support      # + log, packaging and usage checks
    python3 code/evaluation/run_all.py --strict            # warnings are failures too

Stages:

* ``pytest``      - the full unit/property suite under ``code/tests/`` in a subprocess.
* ``samples``     - the 25 solved samples: contract validation, per-field score against the
                    golden CSV, and drift against the frozen regression snapshot.
* ``adversarial`` - the contract/property cases in ``evaluation/adversarial/cases.py``.
* ``output``      - the full-dataset ``output.csv`` validated against ``dataset/requests.csv``,
                    including the 90-day plan replay.

``samples``, ``adversarial`` and ``output`` are produced by ``evaluation/main.py`` so the
runner and the standalone tools can never disagree; ``pytest`` and the optional support
checks are layered around them. The runner is deterministic and never calls a model unless
``--allow-model`` is passed. Writes ``reports/run_all_report.json`` and ``.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
# HERE must end up ahead of CODE: `import main` has to resolve to evaluation/main.py, not
# code/main.py. Inserting CODE last would silently shadow it.
for p in (CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

REPORTS = os.path.join(HERE, "reports")
ALL_STAGES = ("pytest", "samples", "adversarial", "output")


# ---------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------

def stage_pytest(pytest_args=(), targets=None, timeout=1800) -> dict:
    """Run the unit suite in a subprocess so its state cannot leak into this process.

    ``targets`` replaces the default ``code/tests`` collection root. Keep it pointed at a
    leaf test file when calling the runner from inside the suite: targeting the whole
    directory would re-enter this test and recurse.
    """
    roots = list(targets) if targets else [os.path.join(CODE, "tests")]
    cmd = [sys.executable, "-m", "pytest", *roots, "-q", *pytest_args]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "cmd": " ".join(cmd), "seconds": round(time.time() - t0, 2),
                "summary": f"timed out after {timeout}s", "counts": {}}
    except OSError as exc:
        return {"ok": False, "cmd": " ".join(cmd), "seconds": 0.0, "summary": str(exc), "counts": {}}
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return {"ok": proc.returncode == 0, "cmd": " ".join(cmd), "returncode": proc.returncode,
            "seconds": round(time.time() - t0, 2), "summary": tail[-1] if tail else "",
            "counts": _parse_pytest(tail[-1] if tail else ""),
            "failures": [ln for ln in tail if ln.startswith(("FAILED", "ERROR"))][:50],
            "output_tail": "\n".join(tail[-25:])}


PYTEST_OUTCOMES = ("passed", "failed", "error", "errors", "skipped", "xfailed", "xpassed",
                   "warning", "warnings", "deselected")
# pytest reports the plural for a count and the singular for one; normalise so a report
# always has a stable key regardless of how many tests landed in that bucket.
_OUTCOME_ALIASES = {"errors": "error", "warnings": "warning"}


def _parse_pytest(summary: str) -> dict:
    """Pull ``{'passed': 77, 'xfailed': 4}`` out of a pytest summary line.

    The count is the token *before* the outcome word ("77 passed"), so the pairs are
    scanned together -- matching outcome words on their own would silently yield {}.
    """
    counts = {}
    tokens = summary.replace(",", " ").split()
    for count, word in zip(tokens, tokens[1:]):
        if word not in PYTEST_OUTCOMES or not count.isdigit():
            continue
        counts.setdefault(_OUTCOME_ALIASES.get(word, word), int(count))
    return counts


def ensure_output(output: str, dataset: str, requests_file: str, regenerate: bool,
                  allow_model: bool) -> dict:
    """Produce ``output.csv`` when it is missing or ``--regenerate`` was given."""
    if os.path.exists(output) and not regenerate:
        return {"regenerated": False, "path": output, "reason": "already present"}
    from buyorwait.loaders import load_dataset
    from buyorwait.output import write_csv
    from buyorwait.pipeline import run
    ds = load_dataset(dataset, requests_file)
    res = run(ds, use_model=True if allow_model else False)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    write_csv(output, res.rows)
    return {"regenerated": True, "path": output, "rows": len(res.rows),
            "model_calls": bool(allow_model)}


def _load_eval_main():
    """Load ``evaluation/main.py`` by path so ``code/main.py`` can never shadow it."""
    import importlib.util
    path = os.path.join(HERE, "main.py")
    spec = importlib.util.spec_from_file_location("evaluation_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluation_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def stage_evaluation(dataset: str, output: str) -> dict:
    """Run evaluation/main.py in-process: samples regression + adversarial + full validation."""
    eval_main = _load_eval_main()
    buf = io.StringIO()
    started = time.time()
    # main.py validates whatever --output it is given; a missing file is reported there.
    effective_output = output if os.path.exists(output) else None
    with contextlib.redirect_stdout(buf):
        rc = eval_main.main(["--dataset", dataset, "--output", effective_output] if effective_output
                            else ["--dataset", dataset])
    report_path = os.path.join(REPORTS, "evaluation_report.json")
    try:
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        report = {}
        buf.write(f"\ncould not read {report_path}: {exc}\n")
    return {"ok": rc == 0, "returncode": rc, "seconds": round(time.time() - started, 2),
            "evaluated_output": effective_output, "report": report, "stdout": buf.getvalue()}


def stage_output_validation(output: str, dataset: str, requests_file: str, strict: bool) -> dict:
    """Independent strict pass over the full-dataset output."""
    import validate_output as v
    if not os.path.exists(output):
        return {"ok": False, "problems": [f"{output}: does not exist (run with --regenerate)"],
                "warnings": [], "line_endings": "unknown"}
    res = v.validate(output, dataset, requests_file)
    res["line_endings"] = v.line_endings(output)
    if strict and res["warnings"]:
        res["ok"] = False
    return res


def stage_support() -> dict:
    """Optional: log integrity, packaging and usage-report freshness."""
    out = {}
    try:
        import check_log
        out["log"] = check_log.check(os.path.join(ROOT, "log.txt"))
    except Exception as exc:
        out["log"] = {"ok": False, "errors": [f"checker failed: {type(exc).__name__}: {exc}"], "warnings": []}
    try:
        import check_package
        out["package"] = check_package.check(ROOT, zip_path=os.path.join(ROOT, "code.zip"))
    except Exception as exc:
        out["package"] = {"ok": False, "errors": [f"checker failed: {type(exc).__name__}: {exc}"], "warnings": []}
    try:
        import write_usage_report as wur
        out["usage_report"] = wur.check_report(dataset=os.path.join(ROOT, "dataset"),
                                               output=os.path.join(ROOT, "output.csv"))
    except Exception as exc:
        out["usage_report"] = {"ok": False, "problems": [f"checker failed: {type(exc).__name__}: {exc}"]}
    return out


# ---------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------

def run(stages=ALL_STAGES, dataset=None, output=None, requests_file="requests.csv",
        regenerate=False, allow_model=False, strict=False, pytest_args=(),
        pytest_targets=None, with_support=False) -> dict:
    dataset = dataset or os.path.join(ROOT, "dataset")
    output = output or os.path.join(ROOT, "output.csv")
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "dataset": dataset, "output": output, "requests_file": requests_file,
              "strict": strict, "stages": list(stages)}
    t0 = time.time()

    if "output" in stages or "samples" in stages or "adversarial" in stages:
        report["prepare"] = ensure_output(output, dataset, requests_file, regenerate, allow_model)

    if "pytest" in stages:
        report["pytest"] = stage_pytest(pytest_args, targets=pytest_targets)
    if {"samples", "adversarial"} & set(stages):
        ev = stage_evaluation(dataset, output)
        rep = ev.pop("report", {})
        report["evaluation_run"] = ev
        report["samples"] = {"ok": bool(rep.get("samples_score")) and rep.get("samples_contract", {}).get("ok", False),
                             "score": rep.get("samples_score"), "contract": rep.get("samples_contract"),
                             "snapshot": rep.get("snapshot"), "rows": rep.get("samples_rows"),
                             "drawdown": rep.get("samples_drawdown")}
        report["adversarial"] = dict(rep.get("adversarial", {}),
                                     ok=rep.get("adversarial", {}).get("passed") == rep.get("adversarial", {}).get("total")
                                     if rep.get("adversarial") else False)
    if "output" in stages:
        report["output_validation"] = stage_output_validation(output, dataset, requests_file, strict)

    if with_support:
        report["support"] = stage_support()

    report["seconds"] = round(time.time() - t0, 2)
    report["ok"], report["failed_stages"] = _verdict(report, stages, with_support)
    return report


def _verdict(report: dict, stages, with_support: bool):
    failed = []
    if "pytest" in stages and not report.get("pytest", {}).get("ok"):
        failed.append("pytest")
    if "samples" in stages and not report.get("samples", {}).get("ok"):
        failed.append("samples")
    if "adversarial" in stages and not report.get("adversarial", {}).get("ok"):
        failed.append("adversarial")
    if "output" in stages and not report.get("output_validation", {}).get("ok"):
        failed.append("output")
    if with_support:
        for name, res in (report.get("support") or {}).items():
            if not res.get("ok"):
                failed.append(f"support:{name}")
    return not failed, failed


def print_summary(report: dict) -> None:
    print("=" * 72)
    print(f"evaluation run {report['generated_at']}  ({report['seconds']}s)")
    print("=" * 72)
    if "prepare" in report and report["prepare"].get("regenerated"):
        print(f"regenerated {report['prepare']['path']} ({report['prepare'].get('rows')} rows)")
    if "pytest" in report:
        p = report["pytest"]
        print(f"[{'PASS' if p['ok'] else 'FAIL'}] pytest   {p.get('summary', '')}")
        for f in p.get("failures", []):
            print(f"        {f}")
    if "samples" in report:
        s = report["samples"]
        print(f"[{'PASS' if s['ok'] else 'FAIL'}] samples  {json.dumps(s.get('score') or {}, default=str)}")
        snap = s.get("snapshot")
        if isinstance(snap, dict):
            drifted = snap.get("drifted_rows") or []
            print(f"        snapshot drift: {len(drifted)} row(s)")
        else:
            print(f"        snapshot: {snap}")
    if "adversarial" in report:
        a = report["adversarial"]
        print(f"[{'PASS' if a.get('ok') else 'FAIL'}] adversarial {a.get('passed')}/{a.get('total')}")
        for name, msg in a.get("failures", []) or []:
            print(f"        FAIL {name}: {msg}")
    if "output_validation" in report:
        o = report["output_validation"]
        print(f"[{'PASS' if o.get('ok') else 'FAIL'}] output   {o.get('rows')}/{o.get('expected_rows')} rows, "
              f"line endings={o.get('line_endings')}, {len(o.get('problems', []))} problem(s), "
              f"{len(o.get('warnings', []))} warning(s)")
        for pr in o.get("problems", [])[:20]:
            print(f"        ERROR {pr}")
        for w in o.get("warnings", [])[:20]:
            print(f"        warn  {w}")
    if "support" in report:
        for name, res in report["support"].items():
            findings = res.get("errors") or res.get("problems") or []
            n_warn = len(res.get("warnings") or [])
            print(f"[{'PASS' if res.get('ok') else 'FAIL'}] {name} "
                  f"({len(findings)} error(s), {n_warn} warning(s))")
            for e in findings[:10]:
                print(f"        ERROR {e}")
            for w in (res.get("warnings") or [])[:10]:
                print(f"        warn  {w}")
    print("-" * 72)
    print("OVERALL:", "PASS" if report["ok"] else f"FAIL {report['failed_stages']}")


def write_reports(report: dict) -> None:
    os.makedirs(REPORTS, exist_ok=True)
    trimmed = dict(report)
    if "evaluation_run" in trimmed:
        trimmed["evaluation_run"] = {k: v for k, v in trimmed["evaluation_run"].items() if k != "report"}
    with open(os.path.join(REPORTS, "run_all_report.json"), "w", encoding="utf-8") as fh:
        json.dump(trimmed, fh, indent=1, default=str)
    lines = ["# Evaluation run", "", f"Generated: {report['generated_at']} ({report['seconds']}s)", "",
             f"Overall: **{'PASS' if report['ok'] else 'FAIL'}**"
             + (f" - failed: {', '.join(report['failed_stages'])}" if report["failed_stages"] else ""), ""]
    if "pytest" in report:
        lines += [f"## pytest", "", f"- {'PASS' if report['pytest']['ok'] else 'FAIL'}: {report['pytest'].get('summary')}", ""]
    if "samples" in report:
        lines += ["## Samples regression (25 solved requests)", ""]
        lines += [f"- {k}: {v}" for k, v in (report["samples"].get("score") or {}).items()]
        snap = report["samples"].get("snapshot")
        lines += ["", f"- snapshot drift: {len(snap.get('drifted_rows') or []) if isinstance(snap, dict) else snap}", ""]
    if "adversarial" in report:
        a = report["adversarial"]
        lines += [f"## Adversarial cases: {a.get('passed')}/{a.get('total')}", ""]
    if "output_validation" in report:
        o = report["output_validation"]
        lines += [f"## Output validation: {'OK' if o.get('ok') else 'FAIL'}", "",
                  f"- rows: {o.get('rows')} / expected {o.get('expected_rows')}",
                  f"- line endings: {o.get('line_endings')}",
                  f"- problems: {len(o.get('problems', []))}", f"- warnings: {len(o.get('warnings', []))}", ""]
        for pr in o.get("problems", []):
            lines.append(f"- ERROR {pr}")
        for w in o.get("warnings", []):
            lines.append(f"- warn {w}")
        lines.append("")
    if "support" in report:
        lines += ["## Support checks", ""]
        for name, res in report["support"].items():
            lines.append(f"### {name}: {'PASS' if res.get('ok') else 'FAIL'}")
            for e in (res.get("errors") or res.get("problems") or []):
                lines.append(f"- ERROR {e}")
            for w in (res.get("warnings") or []):
                lines.append(f"- warn {w}")
            lines.append("")
    with open(os.path.join(REPORTS, "run_all_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the full evaluation suite")
    ap.add_argument("--stages", default=",".join(ALL_STAGES),
                    help=f"comma list of {','.join(ALL_STAGES)} (default: all)")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--output", default=os.path.join(ROOT, "output.csv"))
    ap.add_argument("--requests", default="requests.csv")
    ap.add_argument("--regenerate", action="store_true", help="rebuild output.csv before validating")
    ap.add_argument("--allow-model", action="store_true", help="let the configured provider be called")
    ap.add_argument("--strict", action="store_true", help="treat validation warnings as failures")
    ap.add_argument("--with-support", action="store_true", help="also run the log/package/usage checkers")
    ap.add_argument("--pytest-args", default="", help="extra arguments passed to pytest")
    a = ap.parse_args(argv)
    stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        print(f"unknown stage(s): {unknown}; expected any of {list(ALL_STAGES)}", file=sys.stderr)
        return 2
    report = run(stages=stages, dataset=a.dataset, output=a.output, requests_file=a.requests,
                 regenerate=a.regenerate, allow_model=a.allow_model, strict=a.strict,
                 pytest_args=tuple(a.pytest_args.split()) if a.pytest_args else (), with_support=a.with_support)
    write_reports(report)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
