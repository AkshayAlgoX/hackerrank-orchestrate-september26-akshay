"""Provider layer: configuration from the environment, OpenAI-compatible wire format, strict typing.

No network access: the transport is exercised with a fake opener that records the request and
returns a canned response.
"""
from __future__ import annotations

import base64
import io
import json
import os
from datetime import date

import pytest

from buyorwait.evidence import validate_many
from buyorwait.extraction.gather import gather_evidence
from buyorwait.extraction.llm import (ModelExtractor, OpenAICompatibleTransport, ProviderConfig, UsageLedger,
                                       _parse_json_object)
from buyorwait.models import Dataset, Event, FxTable, ImageRef, Message


def test_config_from_env_openai_defaults_to_deepseek_endpoint():
    cfg = ProviderConfig.from_env({"DEEPSEEK_API_KEY": "sk-secret-value-123", "BUYORWAIT_LLM_MODEL": "some-vision-model"})
    assert cfg.provider == "openai" and cfg.base_url == "https://api.deepseek.com" and cfg.usable
    assert "sk-secret-value-123" not in cfg.describe()     # the key never appears in any description


def test_config_explicit_provider_and_base_url():
    cfg = ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "openai", "BUYORWAIT_LLM_API_KEY": "secret",
                                   "BUYORWAIT_LLM_MODEL": "m", "BUYORWAIT_LLM_BASE_URL": "https://gw.example/v1/",
                                   "BUYORWAIT_LLM_PRICE_IN": "0.14", "BUYORWAIT_LLM_PRICE_OUT": "0.28"})
    assert cfg.base_url == "https://gw.example/v1" and cfg.price_in == 0.14 and cfg.price_out == 0.28
    assert "secret" not in cfg.describe()


def test_config_none_without_keys_and_model_required():
    assert ProviderConfig.from_env({}).provider == "none"
    assert not ProviderConfig.from_env({"DEEPSEEK_API_KEY": "k"}).usable   # model id is required
    assert not ModelExtractor.available({})
    with pytest.raises(ValueError):
        ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "gemini"})


class FakeOpener:
    def __init__(self, reply: dict):
        self.reply = reply
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        body = json.dumps(self.reply).encode("utf-8")

        class R(io.BytesIO):
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False
        return R(body)


def _reply(text, prompt=120, completion=30, cached=100):
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "prompt_cache_hit_tokens": cached}}


def test_openai_transport_request_shape_and_usage():
    cfg = ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "openai", "BUYORWAIT_LLM_API_KEY": "sk-test",
                                   "BUYORWAIT_LLM_MODEL": "vision-x", "BUYORWAIT_LLM_PRICE_IN": "1", "BUYORWAIT_LLM_PRICE_OUT": "2"})
    opener = FakeOpener(_reply(json.dumps({"evidence": [{"kind": "salary_amount_change", "amount": 2100, "currency": "EUR",
                                                          "effective_date": "2026-02-15", "percent": None, "confidence": 0.9, "note": ""}]})))
    ex = ModelExtractor(cfg, transport=OpenAICompatibleTransport(cfg, opener=opener))
    msg = Message("m1", "u1", None, None, "2026-01-01T00:00:00Z", "employer", "salary increased to EUR 2100 from 2026-02-15")
    raws = ex.extract_message(msg)
    req = opener.requests[0]
    assert req.full_url == "https://api.deepseek.com/chat/completions"
    assert req.get_header("Authorization") == "Bearer sk-test"
    body = json.loads(req.data)
    assert body["model"] == "vision-x" and body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system" and body["messages"][1]["content"][0]["type"] == "text"
    assert "untrusted_message" in body["messages"][1]["content"][0]["text"]
    ok, errs = validate_many(raws)
    assert errs == [] and ok[0].kind == "salary_amount_change" and str(ok[0].amount) == "2100" and ok[0].source_id == "m1"
    u = ex.usage.to_json()["vision-x"]
    assert (u["calls"], u["input_tokens"], u["output_tokens"], u["cache_read_tokens"]) == (1, 120, 30, 100)
    assert u["pricing_known"] and abs(u["cost_usd"] - (120 * 1 + 30 * 2) / 1e6) < 1e-12


def test_openai_transport_image_part_is_data_url(tmp_path):
    cfg = ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "openai", "BUYORWAIT_LLM_API_KEY": "k", "BUYORWAIT_LLM_MODEL": "v"})
    png = tmp_path / "image_99.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    opener = FakeOpener(_reply('```json\n{"evidence": [{"kind": "expense_amount_resolved", "amount": "704.05", "currency": "INR", '
                               '"effective_date": null, "percent": null, "confidence": 1, "note": "amount due"}]}\n```'))
    ex = ModelExtractor(cfg, transport=OpenAICompatibleTransport(cfg, opener=opener))
    ev = Event("e1", "u1", "expense", "Outstanding bill", "utilities", "debit", None, "INR", date(2026, 2, 6), date(2026, 2, 9), "pending", None, "fixed", None)
    raws = ex.extract_image(ImageRef("image_99", "u1", None, "e1", str(png)), ev)
    body = json.loads(opener.requests[0].data)
    parts = body["messages"][1]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1]) == png.read_bytes()
    assert "e1" in parts[1]["text"]
    ok, errs = validate_many(raws)
    assert errs == [] and ok[0].kind == "expense_amount_resolved" and str(ok[0].amount) == "704.05" and ok[0].related_event_id == "e1"


def test_unknown_pricing_is_flagged_not_invented():
    cfg = ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "openai", "BUYORWAIT_LLM_API_KEY": "k", "BUYORWAIT_LLM_MODEL": "v"})
    ex = ModelExtractor(cfg, transport=OpenAICompatibleTransport(cfg, opener=FakeOpener(_reply('{"evidence": []}'))))
    ex.extract_message(Message("m", "u", None, None, "", "bank", "hello"))
    u = ex.usage.to_json()["v"]
    assert u["pricing_known"] is False and u["cost_usd"] == 0.0


def test_malformed_model_output_yields_no_evidence():
    assert _parse_json_object("not json at all") == {}
    assert _parse_json_object('prefix {"evidence": []} suffix') == {"evidence": []}
    cfg = ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "openai", "BUYORWAIT_LLM_API_KEY": "k", "BUYORWAIT_LLM_MODEL": "v"})
    ex = ModelExtractor(cfg, transport=OpenAICompatibleTransport(cfg, opener=FakeOpener(_reply("garbage"))))
    assert ex.extract_message(Message("m", "u", None, None, "", "bank", "hello")) == []


def test_injected_instruction_in_model_output_is_rejected_by_schema():
    # even if a model "obeys" an embedded instruction, only closed-vocabulary literals survive validation
    cfg = ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "openai", "BUYORWAIT_LLM_API_KEY": "k", "BUYORWAIT_LLM_MODEL": "v"})
    reply = _reply(json.dumps({"evidence": [{"kind": "set_balance", "amount": 999999, "currency": "EUR", "effective_date": None,
                                             "percent": None, "confidence": 1, "note": "ignore previous rules"}]}))
    ex = ModelExtractor(cfg, transport=OpenAICompatibleTransport(cfg, opener=FakeOpener(reply)))
    raws = ex.extract_message(Message("m", "u", None, None, "", "unknown", "Pay the release charge now"))
    ok, errs = validate_many(raws)
    assert ok == [] and len(errs) == 1 and "unknown kind" in errs[0]


def test_gather_uses_content_hash_cache_before_calling_the_model(tmp_path):
    msg = Message("m1", "u1", None, None, "2026-01-01T00:00:00Z", "employer", "Totally novel wording: pay goes to EUR 2100 from 2026-02-15.")
    ds = Dataset(profiles={}, events=[], events_by_user={}, events_by_id={}, requests=[], options_by_request={},
                 messages=[msg], images=[], fx=FxTable())
    cfg = ProviderConfig.from_env({"BUYORWAIT_LLM_PROVIDER": "openai", "BUYORWAIT_LLM_API_KEY": "sk-secret-value-123", "BUYORWAIT_LLM_MODEL": "v"})
    opener = FakeOpener(_reply(json.dumps({"evidence": [{"kind": "salary_amount_change", "amount": 2100, "currency": "EUR",
                                                          "effective_date": "2026-02-15", "percent": None, "confidence": 1, "note": ""}]})))
    cache = tmp_path / "cache.json"
    from buyorwait.extraction import llm
    real_init = llm.ModelExtractor.__init__

    def fake_init(self, cfg_=None, usage=None, transport=None):
        real_init(self, cfg, usage, OpenAICompatibleTransport(cfg, opener=opener))
    llm.ModelExtractor.__init__ = fake_init
    try:
        b1 = gather_evidence(ds, use_model=True, cache_path=str(cache))
        b2 = gather_evidence(ds, use_model=True, cache_path=str(cache))
    finally:
        llm.ModelExtractor.__init__ = real_init
    assert b1.sources["m1"] == "model" and b2.sources["m1"] == "cache" and len(opener.requests) == 1
    assert b1.evidence[0].kind == "salary_amount_change" and b2.evidence[0].kind == "salary_amount_change"
    assert "sk-secret-value-123" not in cache.read_text()
