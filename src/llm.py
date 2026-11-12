# src/llm.py
"""
One place for every OpenAI call in the project.

  EMBEDDING_MODEL  text-embedding-3-small  -> chunk/query embeddings (Pinecone)
  LLM_MODEL        gpt-6-luna              -> reranking, answer generation, eval judge

Both can be overridden from .env (EMBEDDING_MODEL=..., LLM_MODEL=...).

The OpenAI client is wrapped with LangSmith's `wrap_openai`, so every chat call
is traced as an LLM run (prompt, output, token usage, latency) and nests under
whatever @traceable span called it.
"""

import os

from openai import OpenAI, AsyncOpenAI
from langsmith.wrappers import wrap_openai

from src.tracing import traceable  # noqa: F401  (also runs load_dotenv())

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536   # text-embedding-3-small output size -- must match the Pinecone index
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-6-luna")

_client = None
_async_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        _client = wrap_openai(OpenAI())
    return _client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        _async_client = wrap_openai(AsyncOpenAI())
    return _async_client


def chat(messages, model=LLM_MODEL, **kwargs) -> str:
    res = get_client().chat.completions.create(model=model, messages=messages, **kwargs)
    return res.choices[0].message.content


async def achat(messages, model=LLM_MODEL, **kwargs) -> str:
    res = await get_async_client().chat.completions.create(model=model, messages=messages, **kwargs)
    return res.choices[0].message.content
