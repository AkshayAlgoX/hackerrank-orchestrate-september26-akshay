#!/usr/bin/env python3
"""Integrity checker for the root ``log.txt`` chat transcript.

    python3 code/evaluation/check_log.py                    # repo-root log.txt
    python3 code/evaluation/check_log.py --log path/to/log.txt --strict
    python3 code/evaluation/check_log.py --json out.json

Checks the AGENTS.md section 5 contract:

* every block opens with ``## [<ISO-8601 timestamp>] <title>`` and the timestamp parses;
* ``SESSION START`` blocks carry all seven metadata fields;
* per-turn blocks carry a user prompt, a response summary, an actions list and a
  ``Context:`` section containing all five context fields;
* every ``tool=`` line is non-empty, is not a placeholder or a generic label, is not a bare
  model name, and matches the tool that opened the enclosing session;
* entry timestamps are non-decreasing, and titles respect the 80-character limit;
* no credential appears anywhere in the transcript.

**This tool never writes to the log.** It reads the file, reports, and exits. Transcript
history must be preserved exactly as written - a checker that could edit the file would
defeat the purpose of the transcript.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, HERE)

import secret_scan  # noqa: E402  (code/evaluation/secret_scan.py)

BLOCK_HEADER = re.compile(r"^## \[(?P<ts>[^\]]*)\]\s?(?P<title>.*)$")
SESSION_FIELDS = ("Repo Root:", "Branch:", "Worktree:", "Parent Agent:", "Language:", "Time Remaining:")
CONTEXT_FIELDS = ("tool=", "branch=", "repo_root=", "worktree=", "parent_agent=")
TURN_SECTIONS = ("User Prompt", "Agent Response Summary:", "Actions:", "Context:")
MAX_TITLE = 80

PLACEHOLDER = re.compile(r"^(?:<[^>]*>|\{\{.*\}\}|\.\.\.|tbd|todo|fill[_-]?me[_-]?in)$", re.I)
GENERIC = {"ai", "agent", "assistant", "llm", "bot", "model", "tool", "harness", "chatbot",
           "unknown", "n/a", "na", "none", "null", "nil", "other", "coding agent", "ai agent"}
# A value that is *only* a model id is invalid: the field names the harness, not the model.
# The negative lookahead keeps real harness names that merely start with a vendor prefix
# ("gemini-cli", "claude-code") out of this rule.
MODEL_ONLY = re.compile(
    r"^(?!.*(?:-cli|-code)$)(?:gpt[-0-9.a-z]*|o[1-9](?:-mini|-pro)?|claude[-0-9.a-z]+|"
    r"gemini[-0-9.a-z]*|deepseek[-0-9.a-z]*|llama[-0-9.a-z]*|mistral[-0-9.a-z]*|"
    r"qwen[-0-9.a-z]*|grok[-0-9.a-z]*|sonnet|opus|haiku)$", re.I)
LANGUAGE = re.compile(r"^(?:js|ts|py|custom:.+)$")
TIME_REMAINING = re.compile(r"^\d+d \d+h \d+m$|^not configured$", re.I)


def _parse_ts(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; ``Z`` is accepted as UTC."""
    v = value.strip()
    if not v:
        return None
    if v.endswith(("Z", "z")):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _blocks(text: str) -> Tuple[List[Tuple[int, str, List[str]]], List[str]]:
    """Split into ``(start_line, header, body_lines)``; also returns leading preamble lines."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("## [")]
    preamble = [ln for ln in lines[: starts[0]] if ln.strip()] if starts else [ln for ln in lines if ln.strip()]
    out = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        out.append((i + 1, lines[i], lines[i + 1: end]))
    return out, preamble


def _field(body: List[str], prefix: str) -> Optional[str]:
    for ln in body:
        if ln.strip().startswith(prefix):
            return ln.strip()[len(prefix):].strip()
    return None


def _field_present(body: List[str], prefix: str) -> bool:
    return any(ln.strip().startswith(prefix) for ln in body)


def check(path: str) -> dict:
    """Inspect a transcript log. Read-only. Returns a report dict with ``ok``."""
    errors: List[str] = []
    warnings: List[str] = []
    result = {"path": path, "blocks": 0, "session_starts": 0, "turns": 0, "tools": {},
              "errors": errors, "warnings": warnings}

    if not os.path.exists(path):
        errors.append(f"{path}: does not exist")
        return dict(result, ok=False)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        errors.append(f"{path}: cannot read ({exc})")
        return dict(result, ok=False)
    if not raw.strip():
        errors.append(f"{path}: file is empty")
        return dict(result, ok=False)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"{path}: not valid UTF-8 ({exc})")
        text = raw.decode("utf-8", errors="replace")
    if not raw.endswith(b"\n"):
        warnings.append("file does not end with a newline")

    blocks, preamble = _blocks(text)
    if preamble:
        warnings.append(f"{len(preamble)} non-blank line(s) before the first block "
                        f"(first: {preamble[0][:60]!r})")
    if not blocks:
        errors.append("no '## [...]' entries found; the log has no blocks")
        return dict(result, ok=False)
    result["blocks"] = len(blocks)

    for f in secret_scan.scan_text(text, path):
        errors.append(f"possible credential in transcript: {f.render()}")

    current_tool: Optional[str] = None
    current_tool_line = 0
    prev_ts: Optional[datetime] = None
    for lineno, header, body in blocks:
        m = BLOCK_HEADER.match(header)
        if not m:
            errors.append(f"line {lineno}: malformed block header {header[:70]!r}")
            continue
        ts_raw, title = m.group("ts"), m.group("title").strip()
        ts = _parse_ts(ts_raw)
        if ts is None:
            errors.append(f"line {lineno}: timestamp {ts_raw!r} is not valid ISO-8601")
        elif prev_ts is not None and ts < prev_ts:
            warnings.append(f"line {lineno}: timestamp {ts_raw} is earlier than the previous entry "
                            f"({prev_ts.isoformat()}); the log is out of chronological order")
        if ts is not None:
            prev_ts = ts
        if not title:
            errors.append(f"line {lineno}: block has no title")
        elif len(title) > MAX_TITLE:
            warnings.append(f"line {lineno}: title is {len(title)} chars (>{MAX_TITLE})")

        if title.strip().upper() == "SESSION START":
            result["session_starts"] += 1
            current_tool, current_tool_line = _check_session_start(lineno, body, errors, warnings, result)
        else:
            result["turns"] += 1
            _check_turn(lineno, body, errors, warnings, result, current_tool, current_tool_line)

    if result["session_starts"] == 0:
        warnings.append("no SESSION START entry found; every session must append one")
    return dict(result, ok=not errors)


def _validate_tool(value: Optional[str], lineno: int, errors: List[str], result: dict) -> Optional[str]:
    """Apply the AGENTS.md mandatory tool-name rule to one value."""
    if value is None:
        errors.append(f"line {lineno}: tool= is missing")
        return None
    if not value:
        errors.append(f"line {lineno}: tool= is blank")
        return None
    if PLACEHOLDER.match(value):
        errors.append(f"line {lineno}: tool={value!r} is still a template placeholder")
        return None
    if value.strip().lower() in GENERIC:
        errors.append(f"line {lineno}: tool={value!r} is a generic label, not a harness name")
        return None
    if MODEL_ONLY.match(value.strip()):
        errors.append(f"line {lineno}: tool={value!r} is a model name, not a harness name")
        return None
    if "\n" in value:
        errors.append(f"line {lineno}: tool= value contains a newline")
        return None
    result["tools"][value] = result["tools"].get(value, 0) + 1
    return value


def _check_session_start(lineno, body, errors, warnings, result) -> Tuple[Optional[str], int]:
    for field in SESSION_FIELDS:
        if not _field_present(body, field):
            errors.append(f"line {lineno}: SESSION START is missing {field}")
            continue
        if not _field(body, field):
            errors.append(f"line {lineno}: SESSION START field {field} is blank")
    lang = _field(body, "Language:")
    if lang and not LANGUAGE.match(lang):
        # Metadata hygiene, not a leak or a structural break: the history stays valid, the
        # enum in the AGENTS.md template is js|ts|py|custom:<name> and this drifted from it.
        warnings.append(f"line {lineno}: Language: {lang!r} is not one of js|ts|py|custom:<name>")
    remaining = _field(body, "Time Remaining:")
    if remaining and not TIME_REMAINING.match(remaining):
        warnings.append(f"line {lineno}: Time Remaining: {remaining!r} is not '<Xd Yh Zm>' "
                        f"or 'not configured'")
    repo_root = _field(body, "Repo Root:")
    if repo_root and not _looks_absolute(repo_root):
        warnings.append(f"line {lineno}: Repo Root: {repo_root!r} is not an absolute path")
    tool_val = _field(body, "tool=") if _field_present(body, "tool=") else None
    return _validate_tool(tool_val, lineno, errors, result), lineno


def _looks_absolute(p: str) -> bool:
    return p.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", p) is not None


def _check_turn(lineno, body, errors, warnings, result, current_tool, current_tool_line) -> None:
    text = "\n".join(body)
    for section in TURN_SECTIONS:
        if section not in text:
            errors.append(f"line {lineno}: turn entry is missing '{section}'")
    if "Context:" not in text:
        return
    context = text.split("Context:", 1)[1]
    ctx_lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    for field in CONTEXT_FIELDS:
        if not any(ln.startswith(field) for ln in ctx_lines):
            errors.append(f"line {lineno}: Context is missing {field}")
    tool = _validate_tool(_field(ctx_lines, "tool="), lineno, errors, result)
    if tool is None:
        return
    if current_tool is None:
        warnings.append(f"line {lineno}: turn entry appears before any SESSION START "
                        f"(tool={tool!r})")
    elif tool != current_tool:
        warnings.append(f"line {lineno}: tool={tool!r} does not match the session opened at "
                        f"line {current_tool_line} with tool={current_tool!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the chat transcript log for integrity")
    ap.add_argument("--log", default=os.path.join(ROOT, "log.txt"))
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--json", default=None, help="write the full report here")
    a = ap.parse_args(argv)
    res = check(a.log)
    ok = res["ok"] and (not a.strict or not res["warnings"])
    print(json.dumps({k: v for k, v in res.items() if k not in ("errors", "warnings")}, indent=1))
    for e in res["errors"]:
        print("  ERROR", e)
    for w in res["warnings"]:
        print("  warn ", w)
    print(f"{res['blocks']} blocks ({res['session_starts']} session starts, {res['turns']} turns), "
          f"tools={res['tools']}, errors={len(res['errors'])}, warnings={len(res['warnings'])} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(res, ok=ok), fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
