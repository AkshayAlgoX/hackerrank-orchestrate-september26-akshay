"""Reproducibility fingerprint: SHA-256 of every deterministic engine file used by a run.

Only file contents are hashed - never environment variables, credentials, or the dataset.
The combined digest changes when any engine source changes, so a usage report or proof
carrying it can be tied to the exact code that produced it.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)

# The engine: the package itself plus the entry point. Evaluation tooling and tests are not
# part of the decision path and are left out on purpose.
ENGINE_ROOTS = ("buyorwait", "main.py")
# Data the offline path depends on besides the dataset: hand-verified image readings.
DATA_INPUTS = (os.path.join("evaluation", "golden", "image_extraction_golden.json"),)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_files(code_dir: str = CODE) -> List[str]:
    out: List[str] = []
    for root in ENGINE_ROOTS:
        full = os.path.join(code_dir, root)
        if os.path.isfile(full):
            out.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith(".py"):
                    out.append(os.path.relpath(os.path.join(dirpath, name), code_dir).replace(os.sep, "/"))
    for rel in DATA_INPUTS:
        if os.path.exists(os.path.join(code_dir, rel)):
            out.append(rel)
    return sorted(set(out))


def engine_fingerprint(code_dir: str = CODE) -> Dict[str, object]:
    files = {rel: _sha256(os.path.join(code_dir, rel)) for rel in engine_files(code_dir)}
    combined = hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode("utf-8")).hexdigest()
    return {"algorithm": "sha256", "files": files, "combined": combined}
