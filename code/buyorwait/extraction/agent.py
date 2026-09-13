"""EvidenceAgent: the bounded perception orchestrator, its tool registry and its records.

Model describes; deterministic code decides. The agent's whole job is to turn one untrusted
source (a message or an image) into canonical `Evidence` records or to ABSTAIN. It chooses a
route, invokes exactly one registered evidence tool, validates the result with the canonical
validator, may invoke the existing bounded recovery once, and returns. It never touches a
ledger, a balance, an exchange rate, a deadline, a plan ranking or an output file - the
registry below states that for every tool, and a test enforces it.

There is no loop: one observation, one route, at most one tool call plus at most one recovery
call (MAX_RECOVERY_CALLS == 1), one result.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..evidence import Evidence, KINDS, validate_many
from ..models import Event, ImageRef, Message
from .rules import classify_message

# ---------------------------------------------------------------------------------------
# 2. tool registry - every capability the perception layer has, and what it may NOT do
# ---------------------------------------------------------------------------------------

EVIDENCE_ITEM_FIELDS = ("kind", "amount", "currency", "effective_date", "percent", "confidence", "note")
FORBIDDEN_AUTHORITY = ("ledger", "balance", "fx", "exchange rate", "deadline", "ranking", "plan", "output")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    purpose: str
    input_contract: str
    output_contract: str
    implementation: str                 # dotted path of the existing code that backs the tool
    side_effects: str = "none"
    financial_authority: str = "none"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "extract_message": ToolSpec(
        name="extract_message",
        purpose="Read one untrusted message and return literal financial facts it states.",
        input_contract="Message(message_id, user_id, request_id, related_event_id, sent_at, source_type, "
                       "message_text); the text is fenced as data (<untrusted_message>).",
        output_contract=f"list of raw evidence dicts with fields {EVIDENCE_ITEM_FIELDS} plus provenance stamps "
                        "(source_kind, source_id, user_id, request_id, related_event_id, sent_at); "
                        "kind must be one of the closed vocabulary; every item still goes through "
                        "evidence.validate_evidence before it exists.",
        implementation="buyorwait.extraction.llm.ModelExtractor.extract_message",
    ),
    "extract_image": ToolSpec(
        name="extract_image",
        purpose="Read one PNG (payslip, bill, receipt, statement) and return the single amount the linked "
                "event refers to, or other literal facts it prints.",
        input_contract="ImageRef(image_id, user_id, request_id, related_event_id, path) and the related Event "
                       "(its description is fenced as data); the PNG bytes are sent as an image part.",
        output_contract="as extract_message; expense_amount_resolved carries amount and currency; dates only "
                        "when a transaction/charge/bill date is visibly printed.",
        implementation="buyorwait.extraction.llm.ModelExtractor.extract_image",
    ),
    "submit_canonical_fact": ToolSpec(
        name="submit_canonical_fact",
        purpose="Bounded recovery: re-express facts of the SAME source in canonical form after the first "
                "extraction failed validation. At most one call per source; never a loop.",
        input_contract="the fenced source, the invalid extraction and the exact validation errors "
                       "(llm.recovery_context); the model answers through native function calling.",
        output_contract="{facts: [evidence item]} with exactly the fields " + str(EVIDENCE_ITEM_FIELDS)
                        + " (additionalProperties false); re-stamped with the source's provenance; validated "
                        "by the same validator as the first extraction.",
        implementation="buyorwait.extraction.llm.ModelExtractor.recover",
    ),
}


def registry_manifest() -> List[Dict[str, Any]]:
    """Inspectable listing of every perception tool (used by tests and audits)."""
    return [spec.to_json() for spec in TOOL_REGISTRY.values()]


# ---------------------------------------------------------------------------------------
# 3. first-class ABSTAIN + 4. the immutable evidence record
# ---------------------------------------------------------------------------------------

class EvidenceStatus(str, Enum):
    ACCEPTED = "ACCEPTED"     # a validated fact came straight from the chosen route
    RECOVERED = "RECOVERED"   # the first extraction failed validation; one recovery call repaired it
    ABSTAINED = "ABSTAINED"   # no validated fact for this source: nothing is invented, unknown is not zero


@dataclass(frozen=True)
class EvidenceRecord:
    """Compact, immutable account of how one source was handled. Audit only.

    Nothing in the financial kernel reads these records; they travel with the run metadata
    and the decision proofs so a reviewer can see, per source, which route produced which
    fact, how confident the perception layer was, and why it abstained when it did.
    """
    source_id: str
    source_kind: str                   # message | image
    user_id: str
    extraction_method: str             # rules | cache | model | golden | none
    provider: str
    model: str
    confidence: Optional[float]
    validation_status: str             # EvidenceStatus value
    recovery_calls: int
    abstention_reason: Optional[str]
    provenance: Tuple[Dict[str, Any], ...]   # the validated facts (kind, amount, currency, effective_date, related_event_id)
    route: str                         # the label kept in EvidenceBundle.sources

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance"] = list(self.provenance)
        return d


@dataclass
class AgentOutcome:
    raws: List[Dict[str, Any]]         # what goes into the bundle (validated again there)
    route: str                         # sources[...] label, unchanged from the pre-agent code
    record: EvidenceRecord


def _facts(raws: List[Dict[str, Any]]) -> Tuple[List[Evidence], List[str]]:
    return validate_many(raws)


def _summ(ev: Evidence) -> Dict[str, Any]:
    return {"kind": ev.kind, "amount": None if ev.amount is None else str(ev.amount), "currency": ev.currency,
            "effective_date": ev.effective_date.isoformat() if ev.effective_date else None,
            "related_event_id": ev.related_event_id}


# ---------------------------------------------------------------------------------------
# 1. the agent
# ---------------------------------------------------------------------------------------

class EvidenceAgent:
    """Observe -> route -> one tool -> validate -> (one recovery) -> canonical evidence or ABSTAIN.

    All state it mutates (`cache`, `recovery`, `provider_errors`) is handed in by gather_evidence
    so the counters and cache semantics are exactly the pre-existing ones.
    """

    def __init__(self, extractor, cache: Dict[str, Any], golden: Dict[str, Any], recovery: Dict[str, Any],
                 provider_errors: List[str], escalate_unmatched_messages: bool = True):
        self.extractor = extractor
        self.cache = cache
        self.golden = golden
        self.recovery = recovery
        self.provider_errors = provider_errors
        self.escalate = escalate_unmatched_messages
        self.provider = getattr(extractor, "provider", "none") if extractor is not None else "none"
        self.model = getattr(extractor, "model", "none") if extractor is not None else "none"
        self.records: List[EvidenceRecord] = []

    # ---- helpers -----------------------------------------------------------------------
    def _recovery_entry(self, source_id: str) -> Optional[Dict[str, Any]]:
        details = self.recovery.get("details") or []
        if details and details[-1].get("source_id") == source_id:
            return details[-1]
        return None

    def _record(self, source_kind: str, source_id: str, user_id: str, method: str, route: str,
                raws: List[Dict[str, Any]], abstain: Optional[str] = None) -> EvidenceRecord:
        ok, errors = _facts(raws)
        usable = [e for e in ok if e.kind not in ("irrelevant",)]
        entry = self._recovery_entry(source_id)
        calls = int(entry.get("calls", 0)) if entry else 0
        if abstain is not None or not usable:
            status = EvidenceStatus.ABSTAINED
            reason = abstain or (entry["outcome"] if entry else ("no fact stated" if ok else "; ".join(errors) or "no evidence"))
        elif entry and str(entry.get("outcome", "")).startswith("recovered"):
            status, reason = EvidenceStatus.RECOVERED, None
        else:
            status, reason = EvidenceStatus.ACCEPTED, None
        conf = max((float(e.confidence) for e in usable), default=None)
        rec = EvidenceRecord(source_id=source_id, source_kind=source_kind, user_id=user_id, extraction_method=method,
                             provider=self.provider if method == "model" else "none",
                             model=self.model if method == "model" else "none",
                             confidence=conf, validation_status=status.value, recovery_calls=calls,
                             abstention_reason=reason, provenance=tuple(_summ(e) for e in usable), route=route)
        self.records.append(rec)
        return rec

    # ---- messages ----------------------------------------------------------------------
    def process_message(self, msg: Message, key: str) -> AgentOutcome:
        from .gather import _validated_or_recovered
        from .llm import ProviderError
        rule = classify_message(msg)
        matched = not (len(rule) == 1 and rule[0]["kind"] == "irrelevant" and rule[0].get("confidence", 1) < 0.5)
        if matched or not self.escalate:
            rec = self._record("message", msg.message_id, msg.user_id, "rules", "rules", rule,
                               abstain=None if matched else "no template matched; escalation disabled")
            return AgentOutcome(rule, "rules", rec)
        if key in self.cache:
            out = self.cache[key]
            return AgentOutcome(out, "cache", self._record("message", msg.message_id, msg.user_id, "cache", "cache", out))
        if self.extractor is not None:
            try:
                out = self.extractor.extract_message(msg)          # tool: extract_message
            except ProviderError as exc:
                # transport gave up (after its bounded retries) or answered garbage: the message is
                # not lost, the deterministic rules result stands and the failure is recorded
                self.provider_errors.append(f"{msg.message_id}: {exc}")
                rec = self._record("message", msg.message_id, msg.user_id, "rules", "rules-after-provider-error", rule,
                                   abstain=f"provider error: {exc}; no template matched")
                return AgentOutcome(rule, "rules-after-provider-error", rec)
            out = _validated_or_recovered(self.extractor, msg.message_id, out, self.recovery, self.provider_errors)
            self.cache[key] = out
            route = "model+recovery" if self._recovery_entry(msg.message_id) else "model"
            return AgentOutcome(out, route, self._record("message", msg.message_id, msg.user_id, "model", route, out))
        rec = self._record("message", msg.message_id, msg.user_id, "rules", "rules", rule,
                           abstain="no provider configured; no template matched")
        return AgentOutcome(rule, "rules", rec)

    # ---- images ------------------------------------------------------------------------
    def process_image(self, img: ImageRef, event: Optional[Event], key: Optional[str], file_hash: Optional[str]) -> AgentOutcome:
        from .gather import _validated_or_recovered
        from .llm import ProviderError
        if not os.path.exists(img.path):
            rec = self._record("image", img.image_id, img.user_id, "none", "missing-file", [],
                               abstain="image file is absent; no evidence invented")
            return AgentOutcome([], "missing-file", rec)
        if key in self.cache:
            out = self.cache[key]
            return AgentOutcome(out, "cache", self._record("image", img.image_id, img.user_id, "cache", "cache", out))
        model_failed = False
        if self.extractor is not None:
            try:
                out = self.extractor.extract_image(img, event)     # tool: extract_image
            except ProviderError as exc:
                self.provider_errors.append(f"{img.image_id}: {exc}")
                model_failed = True           # fall through to the hand-verified golden, if any
            else:
                out = _validated_or_recovered(self.extractor, img.image_id, out, self.recovery, self.provider_errors)
                self.cache[key] = out
                route = "model+recovery" if self._recovery_entry(img.image_id) else "model"
                return AgentOutcome(out, route, self._record("image", img.image_id, img.user_id, "model", route, out))
        g = self.golden.get(img.image_id)
        if g and g.get("sha256") == file_hash and g.get("related_event_id") == img.related_event_id:
            raw = dict(source_kind="image", source_id=img.image_id, user_id=img.user_id, request_id=img.request_id,
                       related_event_id=img.related_event_id, kind="expense_amount_resolved",
                       amount=g["amount"], currency=g["currency"], confidence=1.0,
                       note=f"hand-verified golden: {g.get('field', '')}")
            route = "golden-after-provider-error" if model_failed else "golden"
            return AgentOutcome([raw], route, self._record("image", img.image_id, img.user_id, "golden", route, [raw]))
        route = "unresolved-after-provider-error" if model_failed else "unresolved"
        rec = self._record("image", img.image_id, img.user_id, "none", route, [],
                           abstain="provider unavailable and no verified reading" if model_failed
                           else "no provider configured and no verified reading")
        return AgentOutcome([], route, rec)
