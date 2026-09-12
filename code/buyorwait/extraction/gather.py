"""Collect, cache, and validate evidence for the whole dataset."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    def for_user(self, user_id: str) -> List[Evidence]:
        return [e for e in self.evidence if e.user_id == user_id]


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:24]


def _load_cache(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_cache(path: str, cache: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)


def gather_evidence(ds: Dataset, use_model: Optional[bool] = None, cache_path: Optional[str] = DEFAULT_CACHE,
                    escalate_unmatched_messages: bool = True) -> EvidenceBundle:
    from .llm import ModelExtractor, UsageLedger

    cache = _load_cache(cache_path) if cache_path else {}
    raws: List[Dict[str, Any]] = []
    sources: Dict[str, str] = {}
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
            out = extractor.extract_message(msg)
            cache[key] = out; raws.extend(out); sources[msg.message_id] = "model"
        else:
            raws.extend(rule); sources[msg.message_id] = "rules"

    # ---- images: cache -> model -> hand-verified golden -----------------------------------
    golden = _load_cache(IMAGE_GOLDEN)
    for img in ds.images:
        if not os.path.exists(img.path):
            sources[img.image_id] = "missing-file"
            continue
        key = "img:" + _hash(img.image_id, _file_hash(img.path), img.related_event_id or "")
        event = ds.events_by_id.get(img.related_event_id) if img.related_event_id else None
        if key in cache:
            raws.extend(cache[key]); sources[img.image_id] = "cache"
        elif extractor is not None:
            out = extractor.extract_image(img, event)
            cache[key] = out; raws.extend(out); sources[img.image_id] = "model"
        elif img.image_id in golden and golden[img.image_id].get("sha256") == _file_hash(img.path) \
                and golden[img.image_id].get("related_event_id") == img.related_event_id:
            g = golden[img.image_id]
            raws.append(dict(source_kind="image", source_id=img.image_id, user_id=img.user_id, request_id=img.request_id,
                             related_event_id=img.related_event_id, kind="expense_amount_resolved",
                             amount=g["amount"], currency=g["currency"], confidence=1.0,
                             note=f"hand-verified golden: {g.get('field', '')}"))
            sources[img.image_id] = "golden"
        else:
            sources[img.image_id] = "unresolved"

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
                          model=extractor.model if extractor else "none")
