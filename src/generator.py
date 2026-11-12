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
    "say you don't know. Cite sessions like (Session 3)."
)


def format_context(docs):
    return "\n\n".join(f"(Session {d.metadata.get('session')}) {d.page_content}" for d in docs)


@traceable(run_type="chain", name="generate_answer", metadata={"llm": LLM_MODEL})
def generate(query, docs, model=LLM_MODEL):
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Transcript excerpts:\n{format_context(docs)}\n\nQuestion: {query}"},
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
