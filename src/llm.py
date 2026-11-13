# src/llm.py
"""
One place for every OpenAI call in the project.

  EMBEDDING_MODEL  text-embedding-3-small  -> chunk/query embeddings (Pinecone)
  LLM_MODEL        gpt-6-luna              -> reranking, answer generation

Both can be overridden from .env (EMBEDDING_MODEL=..., LLM_MODEL=...).
Individual OpenAI calls are NOT traced in LangSmith (only failed eval test
cases are logged by evals/harness.py).
"""

import os

from openai import OpenAI, AsyncOpenAI

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
        _client = OpenAI()
    return _client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        _async_client = AsyncOpenAI()
    return _async_client


def chat(messages, model=LLM_MODEL, **kwargs) -> str:
    res = get_client().chat.completions.create(model=model, messages=messages, **kwargs)
    return res.choices[0].message.content


async def achat(messages, model=LLM_MODEL, **kwargs) -> str:
    res = await get_async_client().chat.completions.create(model=model, messages=messages, **kwargs)
    return res.choices[0].message.content
