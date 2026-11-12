# src/generator.py
"""
The main LLM: answers a question from retrieved transcript chunks.

    python -m src.generator "what is regression testing?"
"""

import sys

from src.llm import chat, LLM_MODEL
from src.tracing import traceable, flush_traces

ANSWER_SYSTEM = (
    "You answer questions about a series of LLM-evaluation course sessions. "
    "Use ONLY the provided transcript excerpts. If they don't contain the answer, "
    "say you don't know. If an excerpt is labelled with a session, cite it like "
    "(Session 3); never invent session numbers."
)


def format_context(context):
    """Accepts a list of LangChain Documents (retriever/reranker), a list of
    plain chunk strings (eval_generator / rag_pipeline), or a single string."""
    if isinstance(context, str):
        return context
    parts = []
    for item in context:
        if isinstance(item, str):
            parts.append(item)
        else:
            session = getattr(item, "metadata", {}).get("session")
            prefix = f"(Session {session}) " if session is not None else ""
            parts.append(f"{prefix}{item.page_content}")
    return "\n\n".join(parts)


@traceable(run_type="chain", name="generate_answer", metadata={"llm": LLM_MODEL})
def generate(query, context, model=LLM_MODEL):
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Transcript excerpts:\n{format_context(context)}\n\nQuestion: {query}"},
    ]
    return chat(messages, model=model)


@traceable(run_type="chain", name="rag_pipeline")
def answer(query, retriever):
    docs = retriever.invoke(query)
    return {
        "answer": generate(query, docs),
        "contexts": [d.page_content for d in docs],
    }


if __name__ == "__main__":
    from src.reranker import RerankingRetriever

    q = " ".join(sys.argv[1:]) or "what is regression testing?"
    out = answer(q, RerankingRetriever())
    print(out["answer"])
    flush_traces()
