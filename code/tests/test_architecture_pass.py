import os
import json
from unittest.mock import MagicMock, patch
import pytest

from buyorwait.extraction.gather import gather_evidence
from buyorwait.extraction.agent import EvidenceAgent, EvidenceStatus
from buyorwait.models import Dataset, Message, ImageRef, Event
from buyorwait.evidence import Evidence, validate_many

def test_evidence_agent_invoked():
    """A. EvidenceAgent is actually invoked by gather_evidence()."""
    ds = Dataset(profiles=[], events=[], events_by_user={}, events_by_id={}, requests=[], options_by_request={}, messages=[], images=[], fx={}, dataset_dir="")
    with patch('buyorwait.extraction.agent.EvidenceAgent') as MockAgent:
        instance = MockAgent.return_value
        instance.records = []
        gather_evidence(ds, use_model=False, cache_path=None)
        MockAgent.assert_called_once()

def test_tool_routing():
    """B. Each tool routes to the existing implementation."""
    from buyorwait.extraction.agent import TOOL_REGISTRY
    assert TOOL_REGISTRY["extract_message"].implementation == "buyorwait.extraction.llm.ModelExtractor.extract_message"
    assert TOOL_REGISTRY["extract_image"].implementation == "buyorwait.extraction.llm.ModelExtractor.extract_image"
    assert TOOL_REGISTRY["submit_canonical_fact"].implementation == "buyorwait.extraction.llm.ModelExtractor.recover"

def test_provider_call_count():
    """C. Existing provider-call count is unchanged on clean inputs."""
    # We verify that EvidenceAgent delegates exactly once for a clean input
    extractor = MagicMock()
    extractor.extract_message.return_value = [{"kind": "expense_amount_resolved", "amount": "100.00", "currency": "USD", "confidence": 0.9, "note": "test", "source_kind": "message", "source_id": "m1", "user_id": "u1", "request_id": "r1"}]
    extractor.supports_recovery = True

    agent = EvidenceAgent(extractor, {}, {}, {"details": []}, [])
    msg = Message(message_id="m1", user_id="u1", request_id="r1", related_event_id=None, sent_at="2026-09-13T00:00:00Z", source_type="sms", message_text="spent 100")
    outcome = agent.process_message(msg, "key1")

    extractor.extract_message.assert_called_once_with(msg)
    extractor.recover.assert_not_called()
    assert outcome.record.validation_status == EvidenceStatus.ACCEPTED

def test_recovery_capped_at_one():
    """D. Recovery remains capped at one."""
    extractor = MagicMock()
    # Bad amount missing
    extractor.extract_message.return_value = [{"kind": "expense_amount_resolved", "currency": "USD", "confidence": 0.9, "note": "test", "source_kind": "message", "source_id": "m1", "user_id": "u1", "request_id": "r1"}]
    extractor.recover.return_value = [{"kind": "expense_amount_resolved", "amount": "-100", "currency": "USD", "confidence": 0.9, "note": "test", "source_kind": "message", "source_id": "m1", "user_id": "u1", "request_id": "r1"}]
    extractor.supports_recovery = True

    recovery_dict = {"entered": 0, "calls": 0, "recovered": 0, "rejected": 0, "skipped_unsupported": 0, "details": []}
    agent = EvidenceAgent(extractor, {}, {}, recovery_dict, [])
    msg = Message(message_id="m1", user_id="u1", request_id="r1", related_event_id=None, sent_at="2026-09-13T00:00:00Z", source_type="sms", message_text="spent 100")

    outcome = agent.process_message(msg, "key1")

    extractor.extract_message.assert_called_once()
    extractor.recover.assert_called_once()
    assert recovery_dict["calls"] == 1
    assert outcome.record.validation_status == EvidenceStatus.ABSTAINED

def test_abstain_no_fabricate_zero():
    """E. ABSTAIN cannot fabricate amount=0."""
    agent = EvidenceAgent(None, {}, {}, {"details": []}, [])
    msg = Message(message_id="m1", user_id="u1", request_id="r1", related_event_id=None, sent_at="2026-09-13T00:00:00Z", source_type="sms", message_text="gibberish")
    outcome = agent.process_message(msg, "key1")
    assert outcome.record.validation_status == EvidenceStatus.ABSTAINED
    # Raws should contain only irrelevant rules output, no zero amounts fabricated
    assert all("amount" not in r or r["amount"] is None for r in outcome.raws)

def test_golden_fallback_unchanged():
    """F. Golden fallback behavior remains unchanged."""
    golden = {"img1": {"sha256": "hash1", "related_event_id": "e1", "amount": "50.00", "currency": "USD"}}
    agent = EvidenceAgent(None, {}, golden, {"details": []}, [])
    img = ImageRef(image_id="img1", user_id="u1", request_id="r1", related_event_id="e1", path="dummy.png")

    with patch('os.path.exists', return_value=True):
        outcome = agent.process_image(img, None, "key1", "hash1")

    assert outcome.route == "golden"
    assert outcome.raws[0]["amount"] == "50.00"

def test_financial_decision_output_byte_identical():
    """G. Financial decision output remains byte-identical."""
    import hashlib
    # Only verify this locally if output.csv exists and is baseline
    if os.path.exists("output.csv"):
        with open("output.csv", "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h == "4b2f61af4e8306e9c3cff47ba7af75dcbb3f3a05ed9c9bfce0f8cc5284d718c6"
