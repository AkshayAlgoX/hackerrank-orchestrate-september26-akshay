"""Tests for the credential scanner, the log-integrity checker and the packaging checker."""
import os
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "evaluation"))

import check_log  # noqa: E402
import check_package  # noqa: E402
import secret_scan  # noqa: E402

# Fixture strings are assembled from fragments on purpose: this file ships inside code.zip,
# and a credential-shaped literal in the source would (correctly) be flagged by
# secret_scan.py itself. Splitting them keeps the scanner's own fixture out of its findings.
FAKE_KEY = "sk-ant-" + "api03-" + ("A1b2C3d4E5f6G7h8" + "I9j0K1l2M3n4O5p6")
FAKE_DOTENV = "DEEPSEEK_API" + "_KEY=" + ("9f8e7d6c5b4a3928" + "1706f5e4d3c2b1a0")
FAKE_PEM = "-----BEGIN RSA PRIVATE " + "KEY-----"

SESSION_START = """## [2026-09-12T10:00:00Z] SESSION START

tool=Claude Code
Repo Root: /tmp/repo
Branch: main
Worktree: main
Parent Agent: none
Language: py
Time Remaining: 1d 2h 3m
"""

TURN = """## [2026-09-12T10:05:00Z] Add the validator

User Prompt (verbatim, secrets redacted):
add the validator

Agent Response Summary:
Added the validator and ran the suite.

Actions:
* created code/evaluation/validate_output.py

Context:
tool=Claude Code
branch=main
repo_root=/tmp/repo
worktree=main
parent_agent=none
"""


def _write(tmp_path, text, name="log.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------------------
# secret scanning
# ---------------------------------------------------------------------------------------

def test_scanner_finds_an_anthropic_key_and_redacts_it():
    findings = secret_scan.scan_text(f'key = "{FAKE_KEY}"', "x.py")
    assert findings
    assert findings[0].kind == "anthropic_api_key"
    rendered = findings[0].render()
    assert FAKE_KEY not in rendered, "the raw credential must never appear in a finding"
    assert "***" in rendered


@pytest.mark.parametrize("line", [
    "export BUYORWAIT_LLM_API_KEY=...",
    "export ANTHROPIC_API_KEY=<your-key>",
    "BUYORWAIT_LLM_API_KEY=REDACTED",
    "api_key: Optional[str]",
    "key={'set' if self.api_key else 'MISSING'}",
    "the api key comes from the environment",
])
def test_scanner_ignores_documentation_placeholders(line):
    assert secret_scan.scan_text(line, "README.md") == []


def test_scanner_finds_private_key_blocks():
    findings = secret_scan.scan_text(FAKE_PEM, "id_rsa")
    assert findings and findings[0].kind == "private_key_block"


def test_scanner_flags_a_dotenv_shaped_assignment():
    findings = secret_scan.scan_text(FAKE_DOTENV, ".env")
    assert findings
    assert all(FAKE_DOTENV.split("=", 1)[1] not in f.render() for f in findings)


def test_scanner_finds_a_credential_inside_a_zip(tmp_path):
    zpath = tmp_path / "code.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("code/config.py", f'API_KEY = "{FAKE_KEY}"\n')
        zf.writestr("code/notes.txt", "nothing to see\n")
    with zipfile.ZipFile(zpath) as zf:
        findings = secret_scan.scan_zipfile(zf)
    assert findings and all(FAKE_KEY not in f.render() for f in findings)


# ---------------------------------------------------------------------------------------
# log integrity
# ---------------------------------------------------------------------------------------

def test_a_well_formed_log_passes(tmp_path):
    res = check_log.check(_write(tmp_path, SESSION_START + "\n" + TURN))
    assert res["ok"], res["errors"]
    assert res["session_starts"] == 1
    assert res["turns"] == 1
    assert res["tools"] == {"Claude Code": 2}


def test_checker_never_modifies_the_log(tmp_path):
    """The transcript is frozen history: a checker must not be able to rewrite it."""
    path = _write(tmp_path, SESSION_START + "\n" + TURN)
    before = open(path, "rb").read()
    check_log.check(path)
    assert open(path, "rb").read() == before


def test_missing_file_is_an_error(tmp_path):
    res = check_log.check(str(tmp_path / "nope.txt"))
    assert not res["ok"]
    assert any("does not exist" in e for e in res["errors"])


def test_empty_file_is_an_error(tmp_path):
    res = check_log.check(_write(tmp_path, ""))
    assert not res["ok"]


def test_session_start_missing_tool_is_an_error(tmp_path):
    text = SESSION_START.replace("tool=Claude Code\n", "")
    res = check_log.check(_write(tmp_path, text))
    assert not res["ok"]
    assert any("tool= is missing" in e for e in res["errors"])


def test_blank_tool_is_an_error(tmp_path):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("tool=Claude Code", "tool=")))
    assert not res["ok"]
    assert any("blank" in e for e in res["errors"])


def test_placeholder_tool_is_an_error(tmp_path):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("tool=Claude Code", "tool=<harness>")))
    assert not res["ok"]
    assert any("placeholder" in e for e in res["errors"])


@pytest.mark.parametrize("bad", ["AI", "agent", "assistant", "llm", "unknown", "none", "n/a"])
def test_generic_tool_labels_are_errors(tmp_path, bad):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("tool=Claude Code", f"tool={bad}")))
    assert not res["ok"]
    assert any("generic label" in e for e in res["errors"])


@pytest.mark.parametrize("bad", ["claude-sonnet-5", "gpt-4o", "deepseek-chat", "gemini-2.5-pro"])
def test_model_name_only_tool_is_an_error(tmp_path, bad):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("tool=Claude Code", f"tool={bad}")))
    assert not res["ok"]
    assert any("model name, not a harness" in e for e in res["errors"])


def test_real_harness_names_are_accepted(tmp_path):
    for name in ("Claude Code", "antigravity-cli", "codex-cli", "gemini-cli"):
        res = check_log.check(_write(tmp_path, SESSION_START.replace("tool=Claude Code", f"tool={name}"),
                                       name=f"log_{name.replace(' ', '_')}.txt"))
        assert res["ok"], (name, res["errors"])


def test_session_start_missing_metadata_fields_is_an_error(tmp_path):
    text = SESSION_START.replace("Worktree: main\n", "").replace("Language: py\n", "")
    res = check_log.check(_write(tmp_path, text))
    assert not res["ok"]
    assert any("Worktree:" in e for e in res["errors"])
    assert any("Language:" in e for e in res["errors"])


def test_blank_metadata_field_is_an_error(tmp_path):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("Branch: main", "Branch:")))
    assert not res["ok"]
    assert any("field Branch: is blank" in e for e in res["errors"])


def test_unparseable_timestamp_is_an_error(tmp_path):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("2026-09-12T10:00:00Z", "yesterday")))
    assert not res["ok"]
    assert any("ISO-8601" in e for e in res["errors"])


def test_out_of_order_timestamps_are_warned(tmp_path):
    text = TURN + "\n" + SESSION_START  # a session start dated after the turn
    res = check_log.check(_write(tmp_path, text))
    assert any("out of chronological order" in w for w in res["warnings"])


def test_turn_missing_required_sections_is_an_error(tmp_path):
    text = SESSION_START + "\n" + TURN.replace("Actions:\n* created code/evaluation/validate_output.py\n", "")
    res = check_log.check(_write(tmp_path, text))
    assert not res["ok"]
    assert any("Actions:" in e for e in res["errors"])


def test_turn_missing_a_context_field_is_an_error(tmp_path):
    text = SESSION_START + "\n" + TURN.replace("worktree=main\n", "")
    res = check_log.check(_write(tmp_path, text))
    assert not res["ok"]
    assert any("Context is missing worktree=" in e for e in res["errors"])


def test_tool_mismatch_between_session_and_turn_is_warned(tmp_path):
    text = SESSION_START + "\n" + TURN.replace("tool=Claude Code\nbranch=main", "tool=codex-cli\nbranch=main")
    res = check_log.check(_write(tmp_path, text))
    assert any("does not match the session" in w for w in res["warnings"])


def test_overlong_title_is_warned(tmp_path):
    text = SESSION_START + "\n" + TURN.replace("Add the validator", "x" * 90)
    res = check_log.check(_write(tmp_path, text))
    assert any("title is" in w for w in res["warnings"])


def test_drifted_language_enum_is_warned_not_failed(tmp_path):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("Language: py", "Language: python")))
    assert res["ok"], res["errors"]
    assert any("Language:" in w for w in res["warnings"])


def test_drifted_time_remaining_is_warned(tmp_path):
    res = check_log.check(_write(tmp_path, SESSION_START.replace("1d 2h 3m", "23h 54m")))
    assert any("Time Remaining" in w for w in res["warnings"])


def test_a_credential_in_the_transcript_is_an_error(tmp_path):
    text = TURN.replace("add the validator", f"add the validator\n\nkey: {FAKE_KEY}")
    res = check_log.check(_write(tmp_path, SESSION_START + "\n" + text))
    assert not res["ok"]
    assert any("credential" in e for e in res["errors"])
    assert all(FAKE_KEY not in e for e in res["errors"]), "errors must stay redacted"


def test_log_without_any_session_start_is_warned(tmp_path):
    res = check_log.check(_write(tmp_path, TURN))
    assert any("SESSION START" in w for w in res["warnings"])


# ---------------------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------------------

def _mini_repo(tmp_path, with_usage_report=True, readme=True):
    code = tmp_path / "code"
    (code / "evaluation").mkdir(parents=True)
    (code / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (code / "requirements.txt").write_text("# stdlib only\n", encoding="utf-8")
    if with_usage_report:
        (code / "evaluation" / "usage_report.md").write_text("# Token usage\n", encoding="utf-8")
    if readme:
        (tmp_path / "README.md").write_text("# Solution\n", encoding="utf-8")
    return tmp_path


def test_a_complete_package_passes(tmp_path):
    root = _mini_repo(tmp_path)
    res = check_package.check(str(root), zip_path=str(root / "code.zip"))
    assert res["ok"], res["errors"]
    assert "code/main.py" in [m["path"] for m in res["manifest"]]


def test_missing_usage_report_fails(tmp_path):
    root = _mini_repo(tmp_path, with_usage_report=False)
    res = check_package.check(str(root))
    assert not res["ok"]
    assert any("usage_report.md" in e for e in res["errors"])


def test_missing_readme_fails(tmp_path):
    root = _mini_repo(tmp_path, readme=False)
    res = check_package.check(str(root))
    assert not res["ok"]
    assert any("README" in e for e in res["errors"])


def test_missing_main_is_an_error(tmp_path):
    root = _mini_repo(tmp_path)
    os.remove(root / "code" / "main.py")
    res = check_package.check(str(root))
    assert not res["ok"]
    assert any("code/main.py" in e for e in res["errors"])


@pytest.mark.parametrize("junk", ["code/.venv/bin/python", "code/__pycache__/x.pyc",
                                  "code/.pytest_cache/CACHEDIR.TAG", "code/module.pyc",
                                  "code/build/out.o", "code/node_modules/pkg/index.js"])
def test_caches_and_build_artifacts_are_excluded(tmp_path, junk):
    root = _mini_repo(tmp_path)
    p = root / junk
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("junk\n", encoding="utf-8")
    res = check_package.check(str(root))
    assert res["ok"], res["errors"]
    # A pruned directory is reported by name (trailing slash); a pruned file by its path.
    assert any(junk == e or junk.startswith(e) for e in res["excluded"]), res["excluded"]
    assert junk not in [m["path"] for m in res["manifest"]]


@pytest.mark.parametrize("bad", [".env", "credentials.json", "id_rsa", "secrets.json"])
def test_forbidden_files_inside_the_solution_are_reported(tmp_path, bad):
    root = _mini_repo(tmp_path)
    (root / "code" / bad).write_text("x\n", encoding="utf-8")
    res = check_package.check(str(root))
    assert not res["ok"]
    assert any(bad in e for e in res["errors"])


def test_root_log_is_reported_as_not_submitted(tmp_path):
    """log.txt is the transcript and ships separately, never inside code.zip."""
    root = _mini_repo(tmp_path)
    (root / "log.txt").write_text("## [2026-09-12T10:00:00Z] SESSION START\n", encoding="utf-8")
    res = check_package.check(str(root))
    assert res["ok"], res["errors"]
    assert any(entry.startswith("log.txt") for entry in res["root_skipped"])
    assert "log.txt" not in [m["path"] for m in res["manifest"]]


def test_a_dotenv_inside_code_fails_the_check(tmp_path):
    root = _mini_repo(tmp_path)
    (root / "code" / ".env").write_text("DEEPSEEK_API_KEY=abc123\n", encoding="utf-8")
    res = check_package.check(str(root))
    assert not res["ok"]
    assert any(".env" in e for e in res["errors"])


def test_a_credential_in_the_source_fails_the_check(tmp_path):
    root = _mini_repo(tmp_path)
    (root / "code" / "config.py").write_text(f'API_KEY = "{FAKE_KEY}"\n', encoding="utf-8")
    res = check_package.check(str(root))
    assert not res["ok"]
    assert any("credential" in e for e in res["errors"])
    assert all(FAKE_KEY not in e for e in res["errors"])


def test_build_writes_a_verifiable_archive(tmp_path):
    root = _mini_repo(tmp_path)
    res = check_package.build(str(root), str(root / "code.zip"))
    assert res["built"] and res["ok"], res["errors"]
    assert res["zip_check"]["ok"]
    with zipfile.ZipFile(root / "code.zip") as zf:
        names = zf.namelist()
    assert "code/main.py" in names
    assert "code/evaluation/usage_report.md" in names
    assert "README.md" in names
    assert not any(n.endswith(".pyc") for n in names)


def test_build_refuses_when_a_credential_is_present(tmp_path):
    root = _mini_repo(tmp_path)
    (root / "code" / "leak.py").write_text(f'KEY = "{FAKE_KEY}"\n', encoding="utf-8")
    res = check_package.build(str(root), str(root / "code.zip"))
    assert not res["built"]
    assert not os.path.exists(root / "code.zip")


def test_tampered_archive_is_rejected(tmp_path):
    """The zip is re-opened, so a forbidden member added after the fact is caught."""
    root = _mini_repo(tmp_path)
    check_package.build(str(root), str(root / "code.zip"))
    with zipfile.ZipFile(root / "code.zip", "a") as zf:
        zf.writestr(".venv/pyvenv.cfg", "home = /usr\n")
    res = check_package.verify_zip(str(root / "code.zip"))
    assert not res["ok"]
    assert any("excluded artifact" in p for p in res["problems"])


def test_archive_missing_the_usage_report_is_rejected(tmp_path):
    """The mandatory usage report is checked against the archive, not just the tree."""
    root = _mini_repo(tmp_path)
    with zipfile.ZipFile(root / "code.zip", "w") as zf:
        zf.writestr("code/main.py", "print('hi')\n")
        zf.writestr("README.md", "# Solution\n")
    res = check_package.check(str(root), zip_path=str(root / "code.zip"))
    assert not res["zip_check"]["ok"]
    assert any("usage_report.md" in p for p in res["zip_check"]["problems"])


def test_archive_with_a_credential_inside_is_rejected(tmp_path):
    root = _mini_repo(tmp_path)
    check_package.build(str(root), str(root / "code.zip"))
    with zipfile.ZipFile(root / "code.zip", "a") as zf:
        zf.writestr("code/leak.py", f'KEY = "{FAKE_KEY}"\n')
    res = check_package.verify_zip(str(root / "code.zip"))
    assert not res["ok"]
    assert any("credential" in p for p in res["problems"])
    assert all(FAKE_KEY not in p for p in res["problems"])


def test_real_repository_package_passes():
    """The actual repo must be submittable: no secrets, no stray build artifacts."""
    res = check_package.check(ROOT, zip_path=os.path.join(ROOT, "code.zip"))
    assert res["ok"], res["errors"]
    paths = [m["path"] for m in res["manifest"]]
    assert "code/main.py" in paths
    assert "code/evaluation/usage_report.md" in paths
    assert not any(p.endswith((".pyc", ".pyo")) for p in paths)
