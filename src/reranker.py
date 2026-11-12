import json

from src.retriever import load_store
from src.tracing import traceable
from src.llm import chat, LLM_MODEL

# LLM-as-reranker: the bi-encoder (Pinecone) over-retrieves, then the LLM reads
# every (query, chunk) pair together and scores relevance in ONE call.

RERANK_SYSTEM = (
    "You are a relevance ranker for a retrieval system. Given a user query and a "
    "numbered list of transcript chunks, score how useful each chunk is for "
    "answering the query, from 0 (irrelevant) to 10 (directly answers it). "
    "Score every chunk exactly once."
)

RERANK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rerank_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "score": {"type": "number"},
                        },
                        "required": ["index", "score"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    },
}


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5, model=LLM_MODEL):
        self.store = load_store()
        self.model = model
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking

    @traceable(run_type="retriever", name="bi_encoder_search")
    def _over_retrieve(self, query):
        # 1. OVER-RETRIEVE: fast vector search in Pinecone, deliberately more than we need
        return self.store.similarity_search(query, k=self.fetch_k)

    @traceable(run_type="chain", name="llm_rerank", metadata={"reranker": LLM_MODEL})
    def _rerank(self, query, candidates):
        # 2. RERANK: the LLM scores each chunk against the query
        numbered = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(candidates))
        messages = [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": f"Query: {query}\n\nChunks:\n{numbered}"},
        ]
        try:
            raw = chat(messages, model=self.model, response_format=RERANK_SCHEMA)
            scores = {int(s["index"]): float(s["score"]) for s in json.loads(raw)["scores"]}
        except Exception:
            # if the LLM output can't be parsed, fall back to the vector-search order
            scores = {i: float(len(candidates) - i) for i in range(len(candidates))}

        # 3. SORT by score, keep the best top_k (scores returned so they show in the trace)
        order = sorted(range(len(candidates)), key=lambda i: scores.get(i, 0.0), reverse=True)
        return [
            {"session": candidates[i].metadata.get("session"), "score": scores.get(i, 0.0), "doc": candidates[i]}
            for i in order[: self.top_k]
        ]

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        candidates = self._over_retrieve(query)
        if not candidates:
            return []

        ranked = self._rerank(query, candidates)
        return [r["doc"] for r in ranked]
