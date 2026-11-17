"""Custom deterministic NeMo Guardrails actions for PII, data leakage, and rate limiting."""

from collections import deque
import re
import sys
import time
import types
from typing import Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome

# Share rate-limit timestamps across NeMo's importlib-loaded actions module and Streamlit app.
_RATE_STATE_KEY = "_guardrail_rate_state"
if _RATE_STATE_KEY not in sys.modules:
    _rate_mod = types.ModuleType(_RATE_STATE_KEY)
    _rate_mod.timestamps = deque()
    sys.modules[_RATE_STATE_KEY] = _rate_mod

RATE_TIMESTAMPS: deque = sys.modules[_RATE_STATE_KEY].timestamps

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d{4}[- ]?){3}\d{1,7}\b"),  # Credit card (13-19 digits)
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # Email
    re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b|\b\+\d{1,3}[-.\s]?\d{6,14}\b"),  # Phone (US/Intl)
    re.compile(r"\b[6-9]\d{4}[-\s]?\d{5}\b"),  # Phone (10-digit Indian mobile, e.g. 98765-43210)
    re.compile(r"\bCX-\d{3,6}\b"),  # Student ID (e.g. CX-4471)
]

DATA_LEAKAGE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),  # OpenAI key
    re.compile(r"\blsv2_[A-Za-z0-9_-]{12,}\b"),  # LangSmith key
    re.compile(r"\bpcsk_[A-Za-z0-9_-]{12,}\b"),  # Pinecone key
    re.compile(r"\bgsk_[A-Za-z0-9_-]{12,}\b"),  # Groq key
    re.compile(r"\bhf_[A-Za-z0-9_-]{12,}\b"),  # HuggingFace token
    re.compile(r"\bghp_[A-Za-z0-9]{16,}\b"),  # GitHub token
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),  # Google API key
    re.compile(r"Answer the student's question using ONLY the course transcript excerpts provided below", re.IGNORECASE),
]


def _target_text(context: Optional[dict], source: str) -> str:
    """Pull either the user prompt or bot reply out of NeMo's runtime context."""
    ctx = context or {}
    key = "bot_message" if source == "output" else "user_message"
    return str(ctx.get(key) or "")


@action(is_system_action=True)
async def check_rate_limit(
    max_requests: int = 5,
    window_seconds: int = 60,
    **kwargs,
) -> RailOutcome:
    """Block requests exceeding max_requests within a sliding window_seconds window."""
    now = time.monotonic()
    while RATE_TIMESTAMPS and (now - RATE_TIMESTAMPS[0]) >= window_seconds:
        RATE_TIMESTAMPS.popleft()
    if len(RATE_TIMESTAMPS) >= max_requests:
        return RailOutcome.block()
    RATE_TIMESTAMPS.append(now)
    return RailOutcome.allow()


@action(is_system_action=True)
async def check_pii_regex(
    context: Optional[dict] = None,
    source: str = "input",
    **kwargs,
) -> RailOutcome:
    """Detect common PII patterns (SSN, credit card, email, phone) via regex."""
    text = _target_text(context, source)
    if any(pattern.search(text) for pattern in PII_PATTERNS):
        return RailOutcome.block()
    return RailOutcome.allow()


@action(is_system_action=True)
async def check_data_leakage(
    context: Optional[dict] = None,
    source: str = "output",
    **kwargs,
) -> RailOutcome:
    """Detect leaked API keys, system prompt secrets, or internal markers."""
    text = _target_text(context, source)
    if any(pattern.search(text) for pattern in DATA_LEAKAGE_PATTERNS):
        return RailOutcome.block()
    return RailOutcome.allow()
