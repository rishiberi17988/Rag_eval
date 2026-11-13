# src/tracing.py
"""
LangSmith observability configuration.

We disable automatic per-call tracing (OpenAI calls, LangChain chains, Pinecone
retrievers, and passing test cases) by setting LANGSMITH_TRACING=false in the
process environment, and only enable tracing inside `trace_failed_case` (in
evals/harness.py) via `langsmith.tracing_context(enabled=True)`.
"""

import os

from dotenv import load_dotenv
from langsmith import traceable as _ls_traceable, tracing_context

load_dotenv()

os.environ.setdefault("LANGSMITH_PROJECT", "rag-eval")

# Remember if LangSmith was enabled in .env, then turn off global auto-tracing
# so LangChain (ChatOpenAI, PineconeVectorStore) and normal pipeline steps do
# NOT send every call to LangSmith.
_FLAG = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "true").lower()
_HAS_KEY = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
_FAILURES_TRACING_ENABLED = (_FLAG == "true") and _HAS_KEY

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


def tracing_enabled() -> bool:
    return _FAILURES_TRACING_ENABLED


def traceable(*args, **kwargs):
    """No-op decorator for normal pipeline/eval functions so individual calls
    aren't sent to LangSmith."""
    def _decorator(fn):
        return fn
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return _decorator


def flush_traces():
    """Block until all pending failure spans are sent to LangSmith."""
    if not _FAILURES_TRACING_ENABLED:
        return
    try:
        from langsmith.run_trees import get_cached_client
        get_cached_client().flush()
    except Exception:
        pass


__all__ = ["traceable", "_ls_traceable", "tracing_context", "tracing_enabled", "flush_traces"]
