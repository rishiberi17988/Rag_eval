# src/tracing.py
"""
Single place that wires up LangSmith observability for the whole project.

Every module imports `traceable` from here (instead of straight from langsmith)
so the env is always loaded BEFORE the first span is created.

Enable tracing by adding to .env:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=rag-eval            # optional, defaults to "rag-eval"
    # LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # only for EU region

If LANGSMITH_TRACING is not "true" (or no key is set), every @traceable is a
no-op, so the code runs exactly the same with tracing off.

Because LANGSMITH_TRACING also turns on LangChain's built-in callbacks, the
LangChain pieces (Pinecone retriever, embeddings) show up as child runs nested
under our @traceable spans automatically.
"""

import os

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

os.environ.setdefault("LANGSMITH_PROJECT", "rag-eval")


def tracing_enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    return flag.lower() == "true" and has_key


def flush_traces():
    """Block until all pending spans are sent. Call at the end of short scripts
    so the process doesn't exit before the background uploader finishes."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass
    try:
        from langsmith.run_trees import get_cached_client
        get_cached_client().flush()
    except Exception:
        pass


__all__ = ["traceable", "tracing_enabled", "flush_traces"]
