"""Shared credential scanner used by the packaging and log-integrity checkers.

The scanner is deliberately conservative: it reports the *kind* of credential and a
redacted excerpt, never the value itself, so a finding can be pasted into a report or a
chat transcript without leaking what it found. Placeholder values that appear throughout
the documentation (``...``, ``<your-key>``, ``REDACTED``, ``changeme``) are not reported.

This module is read-only. It never rewrites the files it inspects.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Ordered most-specific first: the first pattern that matches a given line wins, so a
# DeepSeek-style key is not also reported as a generic assigned credential.
PATTERNS: Sequence[tuple] = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("url_with_credentials", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]{1,64}:[^/\s:@]{6,}@")),
    # No leading \b: an underscore-prefixed name such as DEEPSEEK_API_KEY has no word
    # boundary before "api", and those prefixed env vars are exactly the ones that leak.
    ("assigned_credential", re.compile(
        r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret"
        r"|password|passwd|bearer)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-./+]{16,})")),
)

# Values that look like an assignment but are obviously documentation placeholders.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:[.\-_*xX]+|<[^>]*>|\[?redacted\]?|your[_-]?[a-z]*|changeme|example|placeholder"
    r"|none|null|todo|xxx+|abc+|123+|test[_-]?key)$"
)

# Only these extensions are scanned; everything else is treated as binary.
TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".txt", ".rst", ".cfg", ".ini", ".toml", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".sh", ".bash", ".zsh", ".env", ".js", ".ts", ".jsx", ".tsx", ".html",
    ".css", ".sql", ".xml", ".properties", ".conf", ".gitignore", ".zip", ".ipynb",
})
MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    """One suspected credential. ``excerpt`` is always redacted."""

    path: str
    line: int
    kind: str
    excerpt: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.excerpt}"


def redact(value: str) -> str:
    """Mask the middle of a value, keeping a short prefix/suffix for identification."""
    value = value.strip().strip("\"'")
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * min(len(value) - 8, 24)}{value[-4:]} ({len(value)} chars)"


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER.match(value.strip().strip("\"'")))


def scan_text(text: str, path: str = "<memory>") -> List[Finding]:
    """Scan already-decoded text; returns one finding per (line, kind)."""
    out: List[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind, pat in PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            value = m.group(1) if m.groups() else m.group(0)
            if _is_placeholder(value):
                continue
            out.append(Finding(path, lineno, kind, redact(value)))
            break  # one finding per line is enough to act on
    return out


def looks_textual(path: str) -> bool:
    name = os.path.basename(path)
    if name in {"log.txt", ".gitignore", ".env", "Makefile", "Dockerfile"}:
        return True
    return os.path.splitext(name)[1].lower() in TEXT_SUFFIXES


def scan_bytes(data: bytes, path: str = "<memory>") -> List[Finding]:
    if b"\x00" in data[:4096]:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return scan_text(text, path)


def scan_file(path: str) -> List[Finding]:
    """Scan one file on disk. Unreadable or oversized files are skipped silently."""
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return []
        with open(path, "rb") as fh:
            return scan_bytes(fh.read(), path)
    except OSError:
        return []


def scan_paths(paths: Sequence[str], root: Optional[str] = None) -> List[Finding]:
    """Scan every textual file under ``paths`` (files or directories)."""
    out: List[Finding] = []
    for p in paths:
        full = os.path.join(root, p) if root else p
        if os.path.isdir(full):
            for dirpath, dirnames, filenames in os.walk(full):
                dirnames[:] = sorted(d for d in dirnames if d not in {".git", ".venv", "node_modules"})
                for name in sorted(filenames):
                    fp = os.path.join(dirpath, name)
                    if looks_textual(fp) and not name.endswith((".png", ".jpg", ".zip")):
                        out.extend(scan_file(fp))
        elif os.path.isfile(full) and looks_textual(full):
            out.extend(scan_file(full))
    return out


def scan_zipfile(zf) -> List[Finding]:
    """Scan the textual members of an open :class:`zipfile.ZipFile` (``zf``)."""
    out: List[Finding] = []
    for info in zf.infolist():
        if info.is_dir() or not looks_textual(info.filename) or info.file_size > MAX_BYTES:
            continue
        try:
            out.extend(scan_bytes(zf.read(info), f"{zf.filename}!{info.filename}"))
        except (OSError, RuntimeError):
            continue
    return out
