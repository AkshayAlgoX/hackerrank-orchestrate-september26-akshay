#!/usr/bin/env python3
"""Buy or Wait? entry point.

    python3 code/main.py                      # dataset/requests.csv -> output.csv (repo root)
    python3 code/main.py --samples            # run the 25 solved samples -> code/evaluation/reports/samples_output.csv
    python3 code/main.py --no-model           # never call a model (rules + cache + golden only)
    python3 code/main.py --proofs path.json   # also dump the per-request decision proofs

Model access (optional) is configured only through environment variables:

    BUYORWAIT_LLM_PROVIDER=openai            # OpenAI-compatible wire format (DeepSeek, OpenAI, gateways) | anthropic | none
    BUYORWAIT_LLM_MODEL=<model id>           # e.g. the DeepSeek vision model id you have access to
    BUYORWAIT_LLM_BASE_URL=https://api.deepseek.com   # default for the openai protocol
    BUYORWAIT_LLM_API_KEY=...                # or DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
    BUYORWAIT_LLM_PRICE_IN / BUYORWAIT_LLM_PRICE_OUT  # USD per million tokens, for the usage report

Without a provider the run is fully deterministic and offline (rules + content-hash cache +
hand-verified image readings). `--provider-check` sends one tiny extraction and prints usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from buyorwait import finalize  # noqa: E402
from buyorwait.atomic import atomic_write  # noqa: E402
from buyorwait.fingerprint import engine_fingerprint  # noqa: E402
from buyorwait.loaders import load_dataset  # noqa: E402
from buyorwait.output import write_csv  # noqa: E402
from buyorwait.pipeline import run, write_proofs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Buy or Wait? decision engine")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--requests", default=None, help="requests file name inside the dataset dir")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--samples", action="store_true", help="run sample_requests.csv instead of requests.csv")
    ap.add_argument("--no-model", action="store_true", help="disable model calls (rules/cache/golden only)")
    ap.add_argument("--model", action="store_true", help="force model calls (requires credentials)")
    ap.add_argument("--proofs", default=None, help="write decision proofs JSON here")
    ap.add_argument("--usage", default=None, help="write token-usage JSON here")
    ap.add_argument("--report", default=None, help="write the usage report (markdown) here")
    ap.add_argument("--provider-check", action="store_true", help="send one small extraction to the configured provider and exit")
    a = ap.parse_args(argv)
    if a.provider_check:
        return provider_check()

    requests_file = a.requests or ("sample_requests.csv" if a.samples else "requests.csv")
    out = a.out or (os.path.join(HERE, "evaluation", "reports", "samples_output.csv") if a.samples
                    else os.path.join(ROOT, "output.csv"))
    ds = load_dataset(a.dataset, requests_file)
    use_model = False if a.no_model else (True if a.model else None)
    result = run(ds, use_model=use_model)

    # ---- finalization protocol: stage -> validate -> publish -> sign (see buyorwait/finalize.py)
    # A sample run (or a run redirected with --out/--usage) never overwrites the canonical
    # full-dataset metadata, report or manifest under evaluation/reports.
    canonical = not a.samples and a.out is None and a.usage is None
    usage_path = a.usage or os.path.join(HERE, "evaluation", "reports",
                                         "usage_samples_run.json" if a.samples else "usage_last_run.json")
    report_path = a.report or (os.path.join(HERE, "evaluation", "usage_report.md") if canonical
                               else os.path.join(os.path.dirname(os.path.abspath(usage_path)), "usage_report.md"))
    manifest_path = finalize.manifest_path_for(usage_path)
    run_dir = finalize.RunDir(usage_path)
    try:
        staged = {"output": run_dir.staged("output.csv"), "usage": run_dir.staged("usage_last_run.json"),
                  "report": run_dir.staged("usage_report.md")}
        final = {"output": out, "usage": usage_path, "report": report_path}
        if a.proofs:
            staged["proofs"], final["proofs"] = run_dir.staged("proofs.json"), a.proofs
        write_csv(staged["output"], result.rows)
        # Bind the run metadata to the bytes that were actually written: the report tooling refuses
        # to describe an output.csv whose digest differs from the one recorded here.
        output_sha256 = finalize.sha256_of(staged["output"])
        if a.proofs:
            write_proofs(staged["proofs"], result)
        usage = {"provider": result.bundle.provider, "model": result.bundle.model, "requests": len(ds.requests),
                 "usage": result.bundle.usage, "sources": result.bundle.sources,
                 "rejected_evidence": result.bundle.rejected,
                 # context for evaluation/write_usage_report.py: which run these numbers describe
                 "dataset": os.path.abspath(a.dataset), "requests_file": requests_file,
                 "output": os.path.abspath(out),
                 "output_sha256": output_sha256, "output_rows": len(result.rows),
                 "output_bytes": os.path.getsize(staged["output"]),
                 "proofs": os.path.abspath(a.proofs) if a.proofs else None,
                 "report": os.path.abspath(report_path), "manifest": os.path.abspath(manifest_path),
                 "provider_errors": result.bundle.provider_errors,
                 "recovery": {k: v for k, v in result.bundle.recovery.items() if k != "details"},
                 "recovery_details": result.bundle.recovery.get("details", []),
                 "fallback_rows": result.errors,
                 # reproducibility: hashes of the engine files this run executed (no secrets)
                 "engine_fingerprint": engine_fingerprint(),
                 "status": "complete",
                 "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_write(staged["usage"], lambda fh: json.dump(usage, fh, indent=1), mode="w", encoding="utf-8")
        text = _render_report(usage, a.dataset, staged["output"], out, usage_path)
        atomic_write(staged["report"], lambda fh: fh.write(text), mode="w", encoding="utf-8", newline="\n")
        problems = finalize.validate_staged(staged, len(ds.requests), usage)
        if problems:
            print("REFUSING TO PUBLISH: the staged artefact set is inconsistent:", file=sys.stderr)
            for pr in problems:
                print("  ", pr, file=sys.stderr)
            return 3
        manifest = finalize.build_manifest({n: (staged[n], final[n]) for n in staged}, len(result.rows),
                                           usage["engine_fingerprint"]["combined"])
        finalize.publish(staged, final, manifest_path, manifest)
    finally:
        run_dir.cleanup()
    print(f"wrote {len(result.rows)} rows -> {out}")
    rec = result.bundle.recovery
    print(f"evidence: provider={result.bundle.provider} model={result.bundle.model} "
          f"sources={_counts(result.bundle.sources)} rejected={len(result.bundle.rejected)} "
          f"recovery: entered={rec.get('entered', 0)} calls={rec.get('calls', 0)} recovered={rec.get('recovered', 0)}")
    if result.errors:
        print(f"FALLBACK ROWS for {len(result.errors)} request(s) whose evaluation raised "
              f"(conservative not_affordable/not_recommended written):", file=sys.stderr)
        for rid, err in result.errors.items():
            print(f"  {rid}: {err}", file=sys.stderr)
    if result.violations:
        print(f"CONTRACT VIOLATIONS in {len(result.violations)} rows:", file=sys.stderr)
        for rid, errs in result.violations.items():
            print(f"  {rid}: {errs}", file=sys.stderr)
        return 2
    return 0


sha256_of = finalize.sha256_of


def _render_report(usage: dict, dataset: str, staged_output: str, final_output: str, usage_path: str) -> str:
    """Render evaluation/usage_report.md from the *staged* output bytes, displaying final paths."""
    eval_dir = os.path.join(HERE, "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import write_usage_report as wur
    ctx = wur.gather_context(usage, dataset, staged_output, usage_path)
    ctx["output_path"] = os.path.abspath(final_output)      # what the report names
    ctx["output_check_path"] = staged_output                 # what the binding check hashes
    return wur.render(usage, ctx)


def provider_check() -> int:
    """One text extraction against the configured endpoint; prints config and usage, never the key."""
    from buyorwait.extraction.llm import ModelExtractor, ProviderConfig
    from buyorwait.evidence import validate_many
    from buyorwait.models import Message

    cfg = ProviderConfig.from_env()
    print("config:", cfg.describe())
    if not cfg.usable:
        print("provider not usable: set BUYORWAIT_LLM_PROVIDER, BUYORWAIT_LLM_MODEL and an API key in the environment", file=sys.stderr)
        return 2
    ex = ModelExtractor(cfg)
    msg = Message("probe", "probe", None, None, "2026-01-01T00:00:00Z", "employer",
                  "Your monthly salary has increased to EUR 2100. The change applies from 2026-02-15.")
    raws = ex.extract_message(msg)
    ok, errs = validate_many(raws)
    print("evidence:", [e.to_json() for e in ok])
    if errs:
        print("rejected:", errs)
    print("usage:", json.dumps(ex.usage.to_json()))
    return 0


def _counts(sources):
    c = {}
    for v in sources.values():
        c[v] = c.get(v, 0) + 1
    return c


if __name__ == "__main__":
    sys.exit(main())
