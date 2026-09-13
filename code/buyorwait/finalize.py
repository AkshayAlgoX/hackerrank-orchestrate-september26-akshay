"""Run finalization: stage every artefact, validate the set, publish, then sign a manifest.

The submission is a *set* of files - output.csv, the run metadata (usage_last_run.json), the
usage report and optionally the proofs - and the set is only meaningful when every member
comes from the same run. Files are replaced one at a time (POSIX gives no multi-file atomic
swap), so a process that dies between two replacements leaves a mixed set on disk. The
protocol below makes that mixed state *detectable*, and makes the fully published state
*self-consistent*:

    STAGE     every artefact is written into a private run directory
    VALIDATE  the staged set is checked together (row count, per-row contract, the
              metadata's recorded digest equals the staged output's digest, report renders)
    PUBLISH   staged files are moved into place with os.replace, one by one
    SIGN      last of all, run_manifest.json is written with the SHA-256 of every published
              file, the row count and the engine fingerprint

State machine (what ``classify`` answers for the artefacts on disk):

    no manifest                                           -> INCOMPLETE (no completed run)
    manifest unreadable / wrong version / missing fields  -> INCOMPLETE
    manifest present, some listed file missing            -> INCOMPLETE
    manifest present, any digest or the row count differs -> INCOMPLETE (mixed set)
    manifest present, every digest and the row count match-> COMPLETE

Death at any point before SIGN leaves either the previous set untouched (still COMPLETE and
still describing the previous output) or a mixed set whose members no longer match the
previous manifest (INCOMPLETE). Death after SIGN leaves the new set fully published and
signed (COMPLETE). Nothing in between can be classified as complete, because the manifest
is written only after every other file is already in place and is the last file written.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import Callable, Dict, List, Optional

from .atomic import atomic_write

MANIFEST_VERSION = 1
MANIFEST_NAME = "run_manifest.json"
ARTIFACTS = ("output", "usage", "report", "proofs")   # publication order; output last, then the manifest
STAGE_SUFFIX = ".tmp"                                  # staged names end with .tmp: never a valid artefact


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: str) -> int:
    with open(path, "rb") as fh:
        body = fh.read().strip()
    return max(0, len(body.splitlines()) - 1) if body else 0


class RunDir:
    """A private staging directory beside the run metadata; removed on exit unless kept."""

    def __init__(self, beside: str):
        base = os.path.dirname(os.path.abspath(beside)) or "."
        os.makedirs(base, exist_ok=True)
        self.path = tempfile.mkdtemp(prefix=".run-", suffix=STAGE_SUFFIX, dir=base)

    def staged(self, name: str) -> str:
        return os.path.join(self.path, name + STAGE_SUFFIX)

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def build_manifest(files: Dict[str, str], output_rows: int, engine_fingerprint: str) -> dict:
    """Manifest for a set of *staged* files keyed by artefact name (paths may be staged paths;
    only the digests are recorded, together with the final path each will be published to)."""
    return {
        "version": MANIFEST_VERSION,
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_fingerprint": engine_fingerprint,
        "output_rows": int(output_rows),
        "files": {name: {"path": os.path.abspath(final), "sha256": sha256_of(staged)}
                  for name, (staged, final) in files.items()},
    }


def validate_staged(staged: Dict[str, str], expected_rows: int, usage: dict) -> List[str]:
    """Problems that must be empty before anything is published."""
    problems: List[str] = []
    out = staged.get("output")
    if not out or not os.path.exists(out):
        return ["staged output.csv is missing"]
    rows = csv_rows(out)
    if rows != expected_rows:
        problems.append(f"staged output has {rows} rows, expected {expected_rows}")
    digest = sha256_of(out)
    if usage.get("output_sha256") != digest:
        problems.append("staged run metadata does not record the staged output's digest")
    if usage.get("output_rows") != rows:
        problems.append("staged run metadata records a different row count")
    if usage.get("status") != "complete":
        problems.append("staged run metadata is not marked complete")
    for name in ("usage", "report", "proofs"):
        p = staged.get(name)
        if p is not None and (not os.path.exists(p) or os.path.getsize(p) == 0):
            problems.append(f"staged {name} is missing or empty")
    return problems


def publish(staged: Dict[str, str], final: Dict[str, str], manifest_path: str, manifest: dict,
            replace: Callable[[str, str], None] = os.replace, before_step: Optional[Callable[[str], None]] = None) -> List[str]:
    """Move every staged file into place (metadata first, output last), then sign.

    ``before_step(name)`` is a test hook invoked right before each replacement and before the
    manifest write; raising from it simulates the process dying at that exact point.
    Returns the publication order actually performed (for audits).
    """
    done: List[str] = []
    order = [n for n in ("proofs", "usage", "report", "output") if n in staged and staged[n] is not None]
    for name in order:
        if before_step:
            before_step(name)
        os.makedirs(os.path.dirname(os.path.abspath(final[name])) or ".", exist_ok=True)
        replace(staged[name], final[name])
        done.append(name)
    if before_step:
        before_step("manifest")
    atomic_write(manifest_path, lambda fh: json.dump(manifest, fh, indent=1), mode="w", encoding="utf-8")
    done.append("manifest")
    return done


def load_manifest(manifest_path: str) -> tuple:
    """(manifest or None, problem or None)"""
    if not os.path.exists(manifest_path):
        return None, (f"{manifest_path}: no completion manifest (the last run did not finish publishing, "
                      f"or none has been made); run code/main.py")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, f"{manifest_path}: unreadable completion manifest ({type(exc).__name__})"
    if not isinstance(m, dict):
        return None, f"{manifest_path}: completion manifest is not an object"
    for key in ("version", "status", "engine_fingerprint", "output_rows", "files"):
        if key not in m:
            return None, f"{manifest_path}: completion manifest is incomplete (missing {key})"
    if m["version"] != MANIFEST_VERSION:
        return None, f"{manifest_path}: completion manifest version {m['version']!r} is not {MANIFEST_VERSION}"
    if m["status"] != "complete":
        return None, f"{manifest_path}: completion manifest status is {m['status']!r}"
    if not isinstance(m["files"], dict) or "output" not in m["files"] or "usage" not in m["files"]:
        return None, f"{manifest_path}: completion manifest lists no output/usage entry"
    return m, None


def classify(manifest_path: str, paths: Dict[str, str], engine_fingerprint: Optional[str] = None,
             ignore: tuple = ()) -> dict:
    """COMPLETE iff the manifest is valid and every artefact on disk matches it.

    ``paths`` maps artefact names to the files that are expected on disk (the caller decides
    which output.csv / usage / report it is asking about). An artefact the manifest lists but
    the caller does not ask about is still verified at the path the manifest recorded.
    """
    problems: List[str] = []
    m, problem = load_manifest(manifest_path)
    if m is None:
        return {"state": "INCOMPLETE", "problems": [problem], "manifest": None}
    for name, entry in m["files"].items():
        if name in ignore:
            continue
        path = paths.get(name) or entry.get("path")
        if not path or not os.path.exists(path):
            problems.append(f"{name}: {path or '(no path)'} listed in the manifest is missing")
            continue
        actual = sha256_of(path)
        if actual != entry.get("sha256"):
            problems.append(f"{name}: {path} has SHA-256 {actual[:16]}... but the manifest recorded "
                            f"{str(entry.get('sha256'))[:16]}... - not the file this run published")
    for name in ("output", "usage"):
        if name in paths and name not in m["files"]:
            problems.append(f"{name}: not listed in the manifest")
    out = paths.get("output") or m["files"]["output"].get("path")
    if out and os.path.exists(out) and csv_rows(out) != m["output_rows"]:
        problems.append(f"output: {csv_rows(out)} rows on disk, manifest recorded {m['output_rows']}")
    if engine_fingerprint and m["engine_fingerprint"] != engine_fingerprint:
        problems.append(f"engine files changed since the run was signed (signed {m['engine_fingerprint'][:12]}..., "
                        f"now {engine_fingerprint[:12]}...); regenerate before submitting")
    return {"state": "COMPLETE" if not problems else "INCOMPLETE", "problems": problems, "manifest": m}


def resign(manifest_path: str, name: str, path: str) -> None:
    """Update one artefact's digest in an existing valid manifest (used when the report tool
    re-renders the report from the same, still-matching run metadata)."""
    m, problem = load_manifest(manifest_path)
    if m is None:
        raise ValueError(problem)
    m["files"][name] = {"path": os.path.abspath(path), "sha256": sha256_of(path)}
    m["resigned_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write(manifest_path, lambda fh: json.dump(m, fh, indent=1), mode="w", encoding="utf-8")


def manifest_path_for(usage_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(usage_path)), MANIFEST_NAME)
