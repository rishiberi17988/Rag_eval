# evals/eval_retriever_with_reranker.py
"""
Evaluates the reranking Pinecone retriever (src/reranker.py) on its own.
No answer is generated -- only the over-retrieved + LLM-reranked chunks are judged.

    python -m evals.eval_retriever_with_reranker

For each golden in golden/retriever_goldens.json, RerankingRetriever fetches
fetch_k candidates from Pinecone, reranks them with the LLM (gpt-6-luna by
default) down to top_k, and the judge LLM (gpt-6.1-terra by default) scores the
surviving chunks against the golden's ideal answer with two DeepEval metrics:

    Contextual Recall     -- did we retrieve everything needed for the ideal answer?
    Contextual Precision  -- are the relevant chunks ranked ABOVE the irrelevant ones?

Goldens whose source session has no transcript in data/ are skipped (with a
warning), since no retriever could find them.
"""

import glob
import inspect
import json
import os
import re

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever
from src.retriever import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from src.llm import chat, achat, LLM_MODEL, EMBEDDING_MODEL
from src.tracing import traceable, flush_traces   # also runs load_dotenv()
from evals.harness import load_goldens, summarize_by_metric, print_summary

GOLDEN_PATH = "golden/retriever_goldens.json"
# LLM that scores the metrics. Override with JUDGE_MODEL=... in .env.
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL", "gpt-6.1-terra")
THRESHOLD = 0.7
MAX_CONCURRENT = 5   # parallel test cases DeepEval runs at once


# ---------------------------------------------------------------- judge
class OpenAIJudge(DeepEvalBaseLLM):
    """DeepEval judge that calls the OpenAI chat API through src/llm.py
    (so every call is traced in LangSmith).

    When DeepEval passes a `schema` (a pydantic model), the judge requests
    JSON output and parses it into that schema. Without a schema it returns
    the raw text."""

    def __init__(self, model_name: str = JUDGE_MODEL_NAME):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    @staticmethod
    def _messages(prompt: str):
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _to_schema(raw: str, schema):
        if hasattr(schema, "model_validate_json"):
            try:
                return schema.model_validate_json(raw)
            except Exception:
                pass
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start : end + 1] if start != -1 and end != -1 else raw)
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        return schema(**data)

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return chat(self._messages(prompt), model=self.model_name)
        try:
            raw = chat(self._messages(prompt), model=self.model_name,
                       response_format={"type": "json_object"})
        except Exception:
            raw = chat(self._messages(prompt), model=self.model_name)
        return self._to_schema(raw, schema)

    async def a_generate(self, prompt: str, schema=None):
        if schema is None:
            return await achat(self._messages(prompt), model=self.model_name)
        try:
            raw = await achat(self._messages(prompt), model=self.model_name,
                              response_format={"type": "json_object"})
        except Exception:
            raw = await achat(self._messages(prompt), model=self.model_name)
        return self._to_schema(raw, schema)

    def get_model_name(self) -> str:
        return self.model_name


JUDGE_MODEL = OpenAIJudge(JUDGE_MODEL_NAME)


# ---------------------------------------------------------------- helpers
def indexed_sessions():
    """Session numbers that actually exist in the corpus (data/*.vtt)."""
    found = set()
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        m = re.search(r"Session[ _]*(\d+)", path)
        if m:
            found.add(m.group(1))
    return found


def golden_session(g):
    m = re.search(r"(\d+)", g.get("source", ""))
    return m.group(1) if m else None


@traceable(run_type="chain", name="retrieve_for_golden")
def retrieve_for_golden(retriever, query):
    retrieved = retriever.invoke(query)
    return [doc.page_content for doc in retrieved]


@traceable(run_type="chain", name="eval_retriever_with_reranker", tags=["eval", "retriever", "reranker"])
def run(retriever, include_missing=False):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # Goldens whose source session isn't in data/ can NEVER be retrieved, so
    # they would measure missing data, not retriever quality. Skip by default.
    available = indexed_sessions()
    missing = [g for g in goldens if golden_session(g) not in available]
    if missing and not include_missing:
        print(f"[warn] skipping {len(missing)} golden(s) whose source session is not "
              f"in {DATA_DIR}/: {', '.join(g['id'] + ' (' + g['source'] + ')' for g in missing)}")
        goldens = [g for g in goldens if golden_session(g) in available]

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieval_context = retrieve_for_golden(retriever, g["query"])

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    eval_kwargs = {}
    if "max_concurrent" in inspect.signature(evaluate).parameters:
        eval_kwargs["max_concurrent"] = MAX_CONCURRENT

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        **eval_kwargs,
        hyperparameters={
            "model": getattr(retriever, "model", LLM_MODEL),
            "prompt template": "retriever_only (no generator)",
            "retriever": "reranker",
            "embedding_model": EMBEDDING_MODEL,
            "vector_store": "pinecone",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "fetch_k": getattr(retriever, "fetch_k", 10),
            "top_k": getattr(retriever, "top_k", TOP_K),
            "judge_model": JUDGE_MODEL_NAME,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the reranking retriever, then run."""
    return run(RerankingRetriever())


if __name__ == "__main__":
    print_summary("retriever_with_reranker", run_local())
    flush_traces()
