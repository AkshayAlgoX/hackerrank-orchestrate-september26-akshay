#!/usr/bin/env python3
"""Submission packaging checker: what actually goes into ``code.zip``.

    python3 code/evaluation/check_package.py                 # report the current state
    python3 code/evaluation/check_package.py --build         # (re)build code.zip, then verify it
    python3 code/evaluation/check_package.py --zip out.zip --build
    python3 code/evaluation/check_package.py --exclude-reports

Reports, in order:

1. **Required artifacts** - ``code/main.py``, ``README.md``, ``requirements.txt`` and the
   mandatory ``evaluation/usage_report.md`` must all be present and non-empty.
2. **Forbidden content** - a virtualenv, ``__pycache__``/``*.pyc``, ``.git``, editor or
   build directories, ``log.txt``, and any dotenv/credential file. Each one is listed
   explicitly rather than silently dropped.
3. **Credentials** - every text file that would be zipped is scanned by
   ``secret_scan.py``; matches are reported redacted, and ``--build`` refuses to run.
4. **Manifest** - exactly what will be submitted, with sizes, so the list can be eyeballed.
5. **Zip verification** (when ``code.zip`` exists) - the archive is reopened and its member
   list, forbidden entries, credentials and required-artifact presence are re-checked, so
   the check applies to the bytes that will actually be uploaded.

Read-only unless ``--build`` is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, HERE)

import secret_scan  # noqa: E402

# Top-level items that make up the submitted solution. The challenge asks for the runnable
# solution, its README and the evaluation/ folder - nothing else needs to ship.
INCLUDE_ROOTS = ("code", "README.md")

EXCLUDE_DIRS = {".venv", "venv", "env", ".env.d", "__pycache__", ".pytest_cache", ".hypothesis",
                ".git", ".hg", ".svn", "node_modules", ".mypy_cache", ".ruff_cache", ".tox",
                ".idea", ".vscode", ".ipynb_checkpoints", "build", "dist", ".eggs", ".cache"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".o", ".a", ".class", ".zip", ".log", ".gyp",
                    ".egg-info", ".swp", ".swo", ".bak", ".orig", ".rej", ".DS_Store")
# Names that must never be shipped, reported by name rather than silently skipped.
FORBIDDEN_NAMES = {".env", ".env.local", ".env.production", "log.txt", "id_rsa", "id_dsa",
                   "credentials.json", "secrets.json", "service-account.json", ".netrc",
                   ".npmrc", ".pypirc", "token.json", "code.zip"}

# (label, accepted archive paths, why it is required)
REQUIRED = (
    ("code/main.py", ("code/main.py",), "the runnable entry point"),
    ("evaluation/usage_report.md", ("code/evaluation/usage_report.md", "evaluation/usage_report.md"),
     "the mandatory token-usage report"),
    ("README", ("README.md", "code/README.md", "README", "code/README"), "setup and run instructions"),
    ("code/requirements.txt", ("code/requirements.txt",), "declared dependencies"),
)


def rel(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def collect(root: str, exclude_reports: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(included, excluded, forbidden)`` relative paths under ``root``."""
    included: List[str] = []
    excluded: List[str] = []
    forbidden: List[str] = []
    for entry in INCLUDE_ROOTS:
        full = os.path.join(root, entry)
        if not os.path.exists(full):
            continue
        if os.path.isfile(full):
            included.append(rel(root, full))
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            # Pruned directories are recorded by name (with a trailing slash) rather than
            # expanded: a withdrawn .venv must show up in the report without listing
            # thousands of files, and the walk must not descend into it.
            keep = []
            for d in sorted(dirnames):
                if d in EXCLUDE_DIRS or d.endswith(".egg-info"):
                    excluded.append(rel(root, os.path.join(dirpath, d)) + "/")
                else:
                    keep.append(d)
            dirnames[:] = keep
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                r = rel(root, path)
                parts = set(r.split("/"))
                if name in FORBIDDEN_NAMES:
                    forbidden.append(r)
                elif parts & EXCLUDE_DIRS or name.endswith(EXCLUDE_SUFFIXES):
                    excluded.append(r)
                elif exclude_reports and r.startswith("code/evaluation/reports/"):
                    excluded.append(r)
                else:
                    included.append(r)
    return included, excluded, forbidden


def verify_zip(zip_path: str) -> dict:
    """Re-open an existing archive and check the bytes that would actually be uploaded."""
    problems: List[str] = []
    warnings: List[str] = []
    if not os.path.exists(zip_path):
        return {"ok": False, "exists": False, "problems": [f"{zip_path}: not present"],
                "warnings": [], "members": []}
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        return {"ok": False, "exists": True, "problems": [f"{zip_path}: unreadable ({exc})"],
                "warnings": [], "members": []}
    with zf:
        if zf.testzip() is not None:
            problems.append("archive is corrupt (CRC failure)")
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for n in names:
            parts = n.split("/")
            if set(parts) & EXCLUDE_DIRS or n.endswith(EXCLUDE_SUFFIXES):
                problems.append(f"archive contains an excluded artifact: {n}")
            if os.path.basename(n) in FORBIDDEN_NAMES:
                problems.append(f"archive contains a file that must not be submitted: {n}")
        for label, accepted, why in REQUIRED:
            if not any(a in names for a in accepted):
                problems.append(f"archive is missing {label} ({why}); expected one of {list(accepted)}")
        for f in secret_scan.scan_zipfile(zf):
            problems.append(f"archive contains a possible credential: {f.render()}")
        for info in zf.infolist():
            if info.file_size == 0 and not info.is_dir():
                warnings.append(f"archive member is empty: {info.filename}")
        members = [{"name": i.filename, "size": i.file_size} for i in zf.infolist() if not i.is_dir()]
    return {"ok": not problems, "exists": True, "problems": problems, "warnings": warnings,
            "members": members, "bytes": os.path.getsize(zip_path),
            "sha256": _sha256(zip_path)}


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check(root: str, zip_path: Optional[str] = None, exclude_reports: bool = False) -> dict:
    """Report exactly what will be submitted. Read-only."""
    errors: List[str] = []
    warnings: List[str] = []
    included, excluded, forbidden = collect(root, exclude_reports)
    zip_path = zip_path or os.path.join(root, "code.zip")

    for f in forbidden:
        errors.append(f"present in the tree and must not be submitted: {f}")
    if "log.txt" in forbidden:
        warnings.append("log.txt is the chat transcript: submit it as a separate file, never inside code.zip")

    for label, accepted, why in REQUIRED:
        present = [a for a in accepted if a in included]
        if not present:
            errors.append(f"missing required artifact {label} ({why}); expected one of {list(accepted)}")
        elif all(os.path.getsize(os.path.join(root, a)) == 0 for a in present):
            errors.append(f"{label} is empty ({why})")

    if "code/evidence_cache.json" in included:
        warnings.append("code/evidence_cache.json is a runtime content-hash cache; it makes re-runs "
                        "model-free. Keep it only if you want the shipped copy to reproduce offline.")

    credentials = []
    for r in included:
        credentials.extend(secret_scan.scan_file(os.path.join(root, r)))
    for f in credentials:
        errors.append(f"possible credential in {f.render()}")

    manifest = []
    total = 0
    for r in included:
        size = os.path.getsize(os.path.join(root, r))
        total += size
        manifest.append({"path": r, "bytes": size})
    existing_zip = verify_zip(zip_path)
    if not existing_zip["exists"]:
        warnings.append(f"{zip_path}: not built yet (run with --build)")

    return {"ok": not errors, "root": root, "zip": zip_path, "errors": errors, "warnings": warnings,
            "credentials": [f.render() for f in credentials],
            "manifest": manifest, "manifest_bytes": total, "excluded": excluded,
            "forbidden_present": forbidden, "zip_check": existing_zip,
            "root_skipped": _root_skipped(root), "coverage": _coverage(manifest)}


def _root_skipped(root: str) -> List[str]:
    """Top-level entries that exist but are deliberately outside the archive."""
    out = []
    for name in sorted(os.listdir(root)):
        if name in INCLUDE_ROOTS:
            continue
        kind = "dir" if os.path.isdir(os.path.join(root, name)) else "file"
        out.append(f"{name} ({kind})")
    return out


def _coverage(manifest) -> Dict[str, int]:
    """Sanity summary of what the archive will contain, by top-level folder."""
    out: Dict[str, int] = {}
    for m in manifest:
        top = m["path"].split("/")[0]
        out[top] = out.get(top, 0) + 1
    return out


def build(root: str, zip_path: str, exclude_reports: bool = False) -> dict:
    """Write ``code.zip`` from the manifest. Refuses when the check reports errors."""
    res = check(root, zip_path=zip_path, exclude_reports=exclude_reports)
    if not res["ok"]:
        res["built"] = False
        res["errors"].append("refusing to build: fix the errors above first")
        return res
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in res["manifest"]:
            src = os.path.join(root, entry["path"])
            info = zipfile.ZipInfo(entry["path"], date_time=time.localtime(os.path.getmtime(src))[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(src, "rb") as fh:
                zf.writestr(info, fh.read())
    res["built"] = True
    res["zip_check"] = verify_zip(zip_path)
    res["ok"] = res["zip_check"]["ok"]
    return res


def print_report(res: dict, show_manifest: bool = True) -> None:
    print(f"package check: {res['root']}")
    for e in res["errors"]:
        print("  ERROR", e)
    for w in res["warnings"]:
        print("  warn ", w)
    if show_manifest:
        print(f"\nwill submit {len(res['manifest'])} files, {res['manifest_bytes'] / 1024:.1f} KiB "
              f"uncompressed ({', '.join(f'{k}={v}' for k, v in sorted(res['coverage'].items()))}):")
        for m in res["manifest"]:
            print(f"  {m['bytes']:>9,}  {m['path']}")
    if res.get("root_skipped"):
        print(f"\nrepo-root entries not submitted: {', '.join(res['root_skipped'])}")
    if res["excluded"]:
        print(f"\nexcluded (kept out of the archive): {len(res['excluded'])} file(s)")
        for p in res["excluded"][:20]:
            print(f"  - {p}")
        if len(res["excluded"]) > 20:
            print(f"  ... and {len(res['excluded']) - 20} more")
    z = res["zip_check"]
    if z["exists"]:
        print(f"\ncode.zip: {z.get('bytes', 0) / 1024:.1f} KiB, {len(z['members'])} members, "
              f"sha256={z.get('sha256', '')[:16]}...")
        print(f"code.zip verified: {'OK' if z['ok'] else 'FAIL'}")
    print("\n->", "PASS" if res["ok"] else "FAIL")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check (and optionally build) the submission code.zip")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--build", action="store_true", help="write code.zip from the checked manifest")
    ap.add_argument("--exclude-reports", action="store_true", help="leave evaluation/reports/ out")
    ap.add_argument("--quiet-manifest", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    zip_path = a.zip or os.path.join(a.root, "code.zip")
    res = build(a.root, zip_path, a.exclude_reports) if a.build \
        else check(a.root, zip_path=zip_path, exclude_reports=a.exclude_reports)
    print_report(res, show_manifest=not a.quiet_manifest)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
