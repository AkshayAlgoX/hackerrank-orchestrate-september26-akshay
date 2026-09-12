"""Collect, cache, and validate evidence for the whole dataset."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..atomic import atomic_write
from ..evidence import Evidence, validate_many
from ..models import Dataset, ImageRef, Message
from .rules import classify_message

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CACHE = os.path.join(CODE_DIR, "evidence_cache.json")
IMAGE_GOLDEN = os.path.join(CODE_DIR, "evaluation", "golden", "image_extraction_golden.json")


@dataclass
class EvidenceBundle:
    evidence: List[Evidence]
    rejected: List[str]
    sources: Dict[str, str] = field(default_factory=dict)   # source_id -> rules|cache|model|golden
    usage: Dict[str, Any] = field(default_factory=dict)
    provider: str = "none"
    model: str = "none"
    provider_errors: List[str] = field(default_factory=list)   # "<source_id>: <error>" per failed call
    cache_notes: List[str] = field(default_factory=list)       # e.g. an unreadable cache that was ignored
    # Bounded evidence recovery counters (see _validated_or_recovered). "entered" counts sources
    # whose fresh model extraction failed validation; "calls" is the number of recovery calls
    # actually made (at most one per such source); a normal offline run keeps all of these at 0.
    recovery: Dict[str, Any] = field(default_factory=lambda: {"entered": 0, "calls": 0, "recovered": 0,
                                                              "rejected": 0, "skipped_unsupported": 0, "details": []})

    def for_user(self, user_id: str) -> List[Evidence]:
        return [e for e in self.evidence if e.user_id == user_id]


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:24]


def _load_cache(path: str, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load a JSON cache; a missing file is an empty cache, and so is an unreadable one.

    A truncated or corrupted file (an interrupted writer from before atomic saves, a disk
    error, a stray edit) must not take the whole run down: it is ignored, recorded in
    ``notes`` when given, and the run proceeds from an empty cache - exactly what a first
    run does. The next successful save replaces it atomically.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if notes is not None:
            notes.append(f"{path}: unreadable cache ignored ({type(exc).__name__}); starting from an empty cache")
        return {}
    if not isinstance(data, dict):
        if notes is not None:
            notes.append(f"{path}: cache is not a JSON object; starting from an empty cache")
        return {}
    return data


def _save_cache(path: str, cache: Dict[str, Any]) -> None:
    """Replace the cache atomically: a failure leaves the previous valid file untouched."""
    if not path:
        return
    atomic_write(path, lambda fh: json.dump(cache, fh, indent=1, sort_keys=True), mode="w", encoding="utf-8")


def _validated_or_recovered(extractor, source_id: str, raws: List[Dict[str, Any]], recovery: Dict[str, Any],
                            provider_errors: List[str]) -> List[Dict[str, Any]]:
    """Deterministic firewall around a fresh model extraction.

    Every raw item goes through validate_many exactly as before. If all pass, nothing else
    happens (the normal path is unchanged). If some fail and the transport supports native
    tool calling, ONE recovery call may re-express the failed items; its output goes through
    the very same validate_many. Whatever still fails is rejected exactly as it would have been
    without recovery. The returned list contains raw dicts only; validation is repeated on the
    whole bundle later, so nothing bypasses the normal pipeline.
    """
    from ..evidence import validate_many as _validate
    from .llm import MAX_RECOVERY_CALLS, ProviderError
    ok, errors = _validate(raws)
    if not errors:
        return raws
    # split raws into the ones that validated and the ones that did not (validate_many keeps order)
    kept, invalid = [], []
    for r in raws:
        good, _ = _validate([r])
        (kept if good else invalid).append(r)
    recovery["entered"] += 1
    entry = {"source_id": source_id, "errors": list(errors), "outcome": "", "calls": 0}
    recovery["details"].append(entry)
    # Without recovery (or when it fails) the invalid items stay in the list: the final
    # validate_many rejects and records them exactly as it always has.
    if extractor is None or not extractor.supports_recovery:
        recovery["skipped_unsupported"] += 1
        entry["outcome"] = "skipped: transport has no tool calling; invalid items rejected"
        return kept + invalid
    calls = 0
    try:
        calls += 1
        assert calls <= MAX_RECOVERY_CALLS
        recovery["calls"] += 1
        entry["calls"] = calls
        candidates = extractor.recover(invalid, errors)
    except ProviderError as exc:
        provider_errors.append(f"{source_id} (recovery): {exc}")
        recovery["rejected"] += 1
        entry["outcome"] = f"provider error during recovery: {type(exc).__name__}; invalid items rejected"
        return kept + invalid
    good, again = _validate(candidates)
    accepted = [c for c in candidates if not _validate([c])[1]]
    if accepted:
        recovery["recovered"] += 1
        entry["outcome"] = f"recovered {len(accepted)} fact(s)" + (f"; {len(again)} still invalid" if again else "")
    else:
        recovery["rejected"] += 1
        entry["outcome"] = "recovery produced no valid fact; invalid items rejected"
    entry["recovery_errors"] = again
    # a successful repair supersedes the invalid originals; otherwise they are rejected as before
    return kept + accepted if accepted else kept + invalid


def gather_evidence(ds: Dataset, use_model: Optional[bool] = None, cache_path: Optional[str] = DEFAULT_CACHE,
                    escalate_unmatched_messages: bool = True) -> EvidenceBundle:
    from .llm import ModelExtractor, ProviderError, UsageLedger

    cache_notes: List[str] = []
    cache = _load_cache(cache_path, cache_notes) if cache_path else {}
    raws: List[Dict[str, Any]] = []
    sources: Dict[str, str] = {}
    provider_errors: List[str] = []
    recovery: Dict[str, Any] = {"entered": 0, "calls": 0, "recovered": 0, "rejected": 0, "skipped_unsupported": 0, "details": []}
    usage = UsageLedger()
    extractor = None
    if use_model is None:
        use_model = ModelExtractor.available()
    if use_model:
        extractor = ModelExtractor(usage=usage)   # provider/model/key come from the environment only

    # ---- messages: rules first, model only for unmatched templates ------------------------
    for msg in ds.messages:
        rule = classify_message(msg)
        matched = not (len(rule) == 1 and rule[0]["kind"] == "irrelevant" and rule[0].get("confidence", 1) < 0.5)
        if matched or not escalate_unmatched_messages:
            raws.extend(rule)
            sources[msg.message_id] = "rules"
            continue
        key = "msg:" + _hash(msg.message_id, msg.message_text)
        if key in cache:
            raws.extend(cache[key]); sources[msg.message_id] = "cache"
        elif extractor is not None:
            try:
                out = extractor.extract_message(msg)
            except ProviderError as exc:
                # transport gave up (after its bounded retries) or answered garbage: the message is
                # not lost, the deterministic rules result stands and the failure is recorded
                provider_errors.append(f"{msg.message_id}: {exc}")
                raws.extend(rule); sources[msg.message_id] = "rules-after-provider-error"
            else:
                out = _validated_or_recovered(extractor, msg.message_id, out, recovery, provider_errors)
                cache[key] = out; raws.extend(out)
                sources[msg.message_id] = "model+recovery" if recovery["details"] and recovery["details"][-1]["source_id"] == msg.message_id else "model"
        else:
            raws.extend(rule); sources[msg.message_id] = "rules"

    # ---- images: cache -> model -> hand-verified golden -----------------------------------
    golden = _load_cache(IMAGE_GOLDEN, cache_notes)
    for img in ds.images:
        if not os.path.exists(img.path):
            sources[img.image_id] = "missing-file"
            continue
        key = "img:" + _hash(img.image_id, _file_hash(img.path), img.related_event_id or "")
        event = ds.events_by_id.get(img.related_event_id) if img.related_event_id else None
        model_failed = False
        if key in cache:
            raws.extend(cache[key]); sources[img.image_id] = "cache"
            continue
        if extractor is not None:
            try:
                out = extractor.extract_image(img, event)
            except ProviderError as exc:
                provider_errors.append(f"{img.image_id}: {exc}")
                model_failed = True           # fall through to the hand-verified golden, if any
            else:
                out = _validated_or_recovered(extractor, img.image_id, out, recovery, provider_errors)
                cache[key] = out; raws.extend(out)
                sources[img.image_id] = "model+recovery" if recovery["details"] and recovery["details"][-1]["source_id"] == img.image_id else "model"
                continue
        if img.image_id in golden and golden[img.image_id].get("sha256") == _file_hash(img.path) \
                and golden[img.image_id].get("related_event_id") == img.related_event_id:
            g = golden[img.image_id]
            raws.append(dict(source_kind="image", source_id=img.image_id, user_id=img.user_id, request_id=img.request_id,
                             related_event_id=img.related_event_id, kind="expense_amount_resolved",
                             amount=g["amount"], currency=g["currency"], confidence=1.0,
                             note=f"hand-verified golden: {g.get('field', '')}"))
            sources[img.image_id] = "golden-after-provider-error" if model_failed else "golden"
        else:
            sources[img.image_id] = "unresolved-after-provider-error" if model_failed else "unresolved"

    if cache_path:
        _save_cache(cache_path, cache)
    evidence, rejected = validate_many(raws)
    # keep only evidence whose related event (when given) belongs to the same user
    keep = []
    for e in evidence:
        if e.related_event_id and e.related_event_id in ds.events_by_id and ds.events_by_id[e.related_event_id].user_id != e.user_id:
            rejected.append(f"{e.source_id}: related event belongs to another user")
            continue
        keep.append(e)
    return EvidenceBundle(keep, rejected, sources, usage.to_json(),
                          provider=extractor.provider if extractor else "none",
                          model=extractor.model if extractor else "none",
                          provider_errors=provider_errors, cache_notes=cache_notes, recovery=recovery)
