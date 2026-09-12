"""Provider transport resilience (extraction/llm.py) - fully offline, the opener is a stub."""
import io
import json
import socket
import urllib.error
from datetime import date

import pytest

from buyorwait.extraction import llm
from buyorwait.extraction.gather import gather_evidence
from buyorwait.extraction.llm import (DEFAULT_MAX_ATTEMPTS, MAX_BACKOFF, OpenAICompatibleTransport, ProviderConfig,
                                      ProviderError, ProviderResponseError, TRANSIENT_HTTP)
from buyorwait.models import Dataset, FxTable, ImageRef, Message, Profile
from decimal import Decimal as D

CFG = ProviderConfig(provider="openai", model="test-model", base_url="https://example.invalid",
                     api_key="sk-test-" + "x" * 24, price_in=None, price_out=None)
GOOD = {"choices": [{"message": {"content": json.dumps({"evidence": [
    {"kind": "income_ended", "amount": None, "currency": None, "effective_date": None, "percent": None,
     "confidence": 1.0, "note": "ok"}]})}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class _Resp:
    def __init__(self, body: bytes):
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http(code, retry_after=None):
    hdrs = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    import email.message
    m = email.message.Message()
    for k, v in hdrs.items():
        m[k] = v
    return urllib.error.HTTPError("https://example.invalid", code, "err", m, io.BytesIO(b"secret-body"))


def _opener(script):
    """script: list of callables/exceptions/bytes; each call pops the next."""
    calls = []

    def open_(req, timeout=None):
        calls.append(timeout)
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _Resp(item)
    open_.calls = calls
    return open_


def _transport(script, **kw):
    sleeps = []
    t = OpenAICompatibleTransport(CFG, timeout=7.5, opener=_opener(list(script)), sleep=sleeps.append, backoff=0.5, **kw)
    return t, sleeps


# ---------------------------------------------------------------------------------------
# transient statuses are retried with backoff, then succeed
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("code", sorted(TRANSIENT_HTTP))
def test_transient_status_is_retried_then_succeeds(code):
    t, sleeps = _transport([_http(code), json.dumps(GOOD).encode()])
    data, tin, tout, _ = t([{"type": "text", "text": "x"}])
    assert data["evidence"][0]["kind"] == "income_ended" and (tin, tout) == (10, 5)
    assert t.attempts_log == [str(code), "ok"]
    assert sleeps == [0.5]


def test_backoff_is_exponential_and_finite():
    t, sleeps = _transport([_http(429), _http(502), _http(504), _http(429), _http(429)], max_attempts=4)
    with pytest.raises(ProviderError) as ei:
        t([{"type": "text", "text": "x"}])
    assert ei.value.transient and ei.value.attempts == 4 and ei.value.status == 429
    assert sleeps == [0.5, 1.0, 2.0]                     # 3 waits for 4 attempts, doubling
    assert len(t.attempts_log) == 4                      # never more than max_attempts calls


def test_backoff_is_capped():
    t = OpenAICompatibleTransport(CFG, opener=_opener([_http(502)] * 8), sleep=lambda s: None, backoff=8.0, max_attempts=8)
    with pytest.raises(ProviderError):
        t([{"type": "text", "text": "x"}])
    # the internal delay doubles 8 -> 16 -> 32(capped 20) ...; nothing sleeps longer than MAX_BACKOFF
    t2, sleeps = _transport([_http(502)] * 8, max_attempts=8)
    t2.backoff = 8.0
    with pytest.raises(ProviderError):
        t2([{"type": "text", "text": "x"}])
    assert max(sleeps) <= MAX_BACKOFF and len(sleeps) == 7


def test_retry_after_header_is_honoured_but_capped():
    t, sleeps = _transport([_http(429, retry_after=3), json.dumps(GOOD).encode()])
    t([{"type": "text", "text": "x"}])
    assert sleeps == [3.0]
    t, sleeps = _transport([_http(429, retry_after=999), json.dumps(GOOD).encode()])
    t([{"type": "text", "text": "x"}])
    assert sleeps == [MAX_BACKOFF]


# ---------------------------------------------------------------------------------------
# timeouts and network errors
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [socket.timeout("t"), TimeoutError("t"), urllib.error.URLError(socket.timeout("t"))])
def test_timeout_is_retried_and_bounded(exc):
    t, sleeps = _transport([exc, json.dumps(GOOD).encode()])
    t([{"type": "text", "text": "x"}])
    assert t.attempts_log == ["timeout", "ok"]
    t, _ = _transport([exc] * DEFAULT_MAX_ATTEMPTS)
    with pytest.raises(ProviderError) as ei:
        t([{"type": "text", "text": "x"}])
    assert ei.value.transient and "timed out after 7.5s" in str(ei.value)


def test_every_attempt_carries_the_explicit_timeout():
    t, _ = _transport([_http(503), json.dumps(GOOD).encode()])
    t([{"type": "text", "text": "x"}])
    assert t._open.calls == [7.5, 7.5]


@pytest.mark.parametrize("exc", [urllib.error.URLError("dns"), ConnectionResetError(), OSError("boom")])
def test_network_error_is_retried(exc):
    t, _ = _transport([exc, json.dumps(GOOD).encode()])
    t([{"type": "text", "text": "x"}])
    assert t.attempts_log == ["network", "ok"]


# ---------------------------------------------------------------------------------------
# non-transient failures are surfaced immediately, never retried, never leak the key
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("code", [400, 401, 403, 404, 413, 422, 500])
def test_non_transient_status_is_not_retried(code):
    t, sleeps = _transport([_http(code), json.dumps(GOOD).encode()])
    with pytest.raises(ProviderError) as ei:
        t([{"type": "text", "text": "x"}])
    assert not ei.value.transient and ei.value.status == code and ei.value.attempts == 1
    assert sleeps == [] and t.attempts_log == [str(code)]
    assert "secret-body" not in str(ei.value) and CFG.api_key not in str(ei.value)


@pytest.mark.parametrize("body", [b"<html>gateway</html>", b"\xff\xfe", b"[1,2,3]", b""])
def test_malformed_response_body_is_not_retried(body):
    t, sleeps = _transport([body, json.dumps(GOOD).encode()])
    with pytest.raises(ProviderResponseError) as ei:
        t([{"type": "text", "text": "x"}])
    assert not ei.value.transient and sleeps == [] and t.attempts_log == ["malformed"]


@pytest.mark.parametrize("content", ["not json at all", "```json\n{\"evidence\": \"nope\"}\n```", "{\"other\": 1}", ""])
def test_malformed_model_content_yields_no_evidence_without_retry(content):
    payload = {"choices": [{"message": {"content": content}}], "usage": {}}
    t, sleeps = _transport([json.dumps(payload).encode()])
    data, *_ = t([{"type": "text", "text": "x"}])
    assert sleeps == [] and t.attempts_log == ["ok"]
    ex = llm.ModelExtractor(CFG, transport=t)
    ex.transport = lambda content: (data, 0, 0, 0)
    assert ex._call([]) == []


def test_schema_violations_are_rejected_upstream_not_retried():
    """A syntactically valid completion with an unknown kind is dropped by validate_evidence."""
    from buyorwait.evidence import validate_many
    raw = [{"kind": "approve_everything", "amount": 1, "source_kind": "message", "source_id": "m", "user_id": "u"}]
    ok, errors = validate_many(raw)
    assert ok == [] and errors and "unknown kind" in errors[0]


# ---------------------------------------------------------------------------------------
# gather_evidence: provider failure degrades to rules/golden, cache untouched
# ---------------------------------------------------------------------------------------

def _ds(msg_text, tmp_path, with_image=False):
    prof = Profile("u1", "EUR", D("1000"), D("100"), (), (), (), (), ("full_payment",), None)
    msgs = [Message("m1", "u1", "r1", None, "2026-05-01T09:00:00Z", "employer", msg_text)]
    imgs = []
    if with_image:
        p = tmp_path / "image_x.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        imgs = [ImageRef("image_x", "u1", "r1", None, str(p))]
    return Dataset(profiles={"u1": prof}, events=[], events_by_user={"u1": []}, events_by_id={}, requests=[],
                   options_by_request={}, messages=msgs, images=imgs, fx=FxTable())


class _FailingExtractor:
    provider, model = "openai", "test-model"

    def __init__(self, usage=None):
        pass

    def extract_message(self, msg):
        raise ProviderError("model endpoint returned HTTP 502", status=502, transient=True, attempts=4)

    def extract_image(self, img, event):
        raise ProviderError("model endpoint timed out after 60s", transient=True, attempts=4)


def test_gather_falls_back_to_rules_when_the_provider_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ModelExtractor", _FailingExtractor)
    cache = tmp_path / "cache.json"
    ds = _ds("Our records show nothing you have templates for; free text only.", tmp_path, with_image=True)
    b = gather_evidence(ds, use_model=True, cache_path=str(cache))
    assert b.sources["m1"] == "rules-after-provider-error"
    assert b.sources["image_x"] == "unresolved-after-provider-error"
    assert len(b.provider_errors) == 2 and all("HTTP 502" in e or "timed out" in e for e in b.provider_errors)
    assert not any(e.kind != "irrelevant" for e in b.evidence)     # nothing invented
    assert json.load(open(cache)) == {}                               # a failed call is never cached


def test_gather_prefers_the_cache_and_never_calls_a_failing_provider(tmp_path, monkeypatch):
    calls = []

    class Exploding(_FailingExtractor):
        def extract_message(self, msg):
            calls.append(msg.message_id)
            raise AssertionError("must not be called when cached")
    monkeypatch.setattr(llm, "ModelExtractor", Exploding)
    ds = _ds("free text with no template", tmp_path)
    from buyorwait.extraction.gather import _hash
    key = "msg:" + _hash("m1", ds.messages[0].message_text)
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({key: [{"kind": "income_ended", "source_kind": "message", "source_id": "m1",
                                        "user_id": "u1", "sent_at": "2026-05-01T09:00:00Z"}]}), encoding="utf-8")
    b = gather_evidence(ds, use_model=True, cache_path=str(cache))
    assert calls == [] and b.sources["m1"] == "cache" and b.evidence[0].kind == "income_ended"


def test_templated_messages_never_touch_the_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ModelExtractor", _FailingExtractor)
    ds = _ds("Your seasonal contract has ended.", tmp_path)
    b = gather_evidence(ds, use_model=True, cache_path=None)
    assert b.sources["m1"] == "rules" and b.provider_errors == [] and b.evidence[0].kind == "income_ended"
