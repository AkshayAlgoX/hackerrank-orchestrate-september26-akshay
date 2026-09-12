"""Perception boundary: untrusted text/images -> validated Evidence.

Order of authority for a given source:
  1. deterministic rules (messages; closed template vocabulary, EN + ID)
  2. cached model output (content-hash keyed)
  3. live model call (Anthropic) when credentials are configured
  4. hand-verified golden extraction (images only) as the offline fallback
Everything passes through `validate_evidence` before it can influence a decision.
"""
from .gather import gather_evidence, EvidenceBundle  # noqa: F401
