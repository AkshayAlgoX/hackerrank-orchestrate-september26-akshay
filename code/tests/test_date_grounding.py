import pytest
from buyorwait.extraction.llm import SYSTEM, RECOVERY_SYSTEM

def test_date_grounding_rules_present_in_prompts():
    """Verify that the strict date-grounding rules are present in the system prompts to prevent hallucination."""
    for prompt in (SYSTEM, RECOVERY_SYSTEM):
        assert "never infer a transaction date" in prompt.lower()
        assert "only emit effective_date when an explicit" in prompt.lower()
        assert "never use expiry date" in prompt.lower()
        assert "document month/year" in prompt.lower()
        assert "filename" in prompt.lower()
        assert "missing or uncertain dates as null" in prompt.lower()
