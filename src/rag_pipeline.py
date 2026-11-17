import asyncio
from functools import lru_cache
import os
from pathlib import Path

import nest_asyncio
from nemoguardrails import LLMRails, RailsConfig

from src.retriever import build_retriever
from src.generator import generate
from src.tracing import traceable, _ls_traceable

nest_asyncio.apply()

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = str(ROOT / "guardrails" / "config")

DEFAULT_INPUT_FLOWS = (
    "jailbreak check input",
    "prompt injection check input",
    "pii check input",
    "data leakage check input",
    "toxicity check input",
    "topic check input",
)

DEFAULT_OUTPUT_FLOWS = (
    "pii check output",
    "data leakage check output",
    "toxicity check output",
    "bias check output",
)

INPUT_ONLY = {
    "rails": {"input": True, "dialog": False, "retrieval": False, "output": False},
    "log": {"activated_rails": True},
}
OUTPUT_ONLY = {
    "rails": {"input": False, "dialog": False, "retrieval": False, "output": True},
    "log": {"activated_rails": True},
}


@lru_cache(maxsize=32)
def get_single_rail(stage: str, flow: str) -> LLMRails:
    cfg = RailsConfig.from_path(CONFIG_DIR)
    if stage == "input":
        cfg.rails.input.flows = [flow]
        cfg.rails.output.flows = []
    else:
        cfg.rails.input.flows = []
        cfg.rails.output.flows = [flow]
    return LLMRails(cfg)


def _blocking_rail(res) -> str | None:
    if res.log and res.log.activated_rails:
        for rail in res.log.activated_rails:
            if rail.stop:
                return rail.name
    return None


@_ls_traceable(name="check_input_rail", run_type="chain")
async def check_input_rail(flow: str, text: str) -> dict:
    rails = get_single_rail("input", flow)
    res = await rails.generate_async(
        messages=[{"role": "user", "content": text}],
        options=INPUT_ONLY,
    )
    msg = res.response[-1]["content"]
    rail = _blocking_rail(res)
    blocked = bool(rail) or msg != text
    return {"passed": not blocked, "blocked": blocked, "rail": rail or flow, "message": msg}


@_ls_traceable(name="check_output_rail", run_type="chain")
async def check_output_rail(flow: str, user_text: str, bot_text: str) -> dict:
    rails = get_single_rail("output", flow)
    msgs = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": bot_text},
    ]
    res = await rails.generate_async(messages=msgs, options=OUTPUT_ONLY)
    msg = res.response[-1]["content"]
    rail = _blocking_rail(res)
    blocked = bool(rail) or msg != bot_text
    return {"passed": not blocked, "blocked": blocked, "rail": rail or flow, "message": msg}


@_ls_traceable(name="input_guardrail", run_type="chain")
async def validate_input(text: str, input_flows: tuple[str, ...] = DEFAULT_INPUT_FLOWS) -> dict:
    if not input_flows:
        return {"blocked": False, "rail": None, "message": text}
    results = await asyncio.gather(
        *(check_input_rail(flow, text) for flow in input_flows)
    )
    for res in results:
        if not res["passed"]:
            return res
    return {"blocked": False, "rail": None, "message": text}


@_ls_traceable(name="output_guardrail", run_type="chain")
async def validate_output(
    user_text: str,
    bot_text: str,
    output_flows: tuple[str, ...] = DEFAULT_OUTPUT_FLOWS,
) -> dict:
    if not output_flows:
        return {"blocked": False, "rail": None, "message": bot_text}
    results = await asyncio.gather(
        *(check_output_rail(flow, user_text, bot_text) for flow in output_flows)
    )
    for res in results:
        if not res["passed"]:
            return res
    return {"blocked": False, "rail": None, "message": bot_text}


class RagPipeline:
    def __init__(
        self,
        fetch_k=10,
        top_k=5,
        use_guardrails: bool | None = None,
        input_flows: tuple[str, ...] = DEFAULT_INPUT_FLOWS,
        output_flows: tuple[str, ...] = DEFAULT_OUTPUT_FLOWS,
    ):
        # one retriever instance — loads the normal Pinecone vector store retriever
        self.retriever = build_retriever()
        if use_guardrails is None:
            use_guardrails = os.getenv("USE_GUARDRAILS", "0").lower() in ("1", "true", "yes")
        self.use_guardrails = use_guardrails
        self.input_flows = tuple(input_flows)
        self.output_flows = tuple(output_flows)

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        if self.use_guardrails and self.input_flows:
            inp = asyncio.run(validate_input(query, self.input_flows))
            if inp["blocked"]:
                return {
                    "query": query,
                    "context": [],
                    "answer": inp["message"],
                    "answered_by": f"input_guardrail ({inp['rail']})",
                }

        # 1. RETRIEVE: fetch top_k Documents from Pinecone
        docs = self.retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context
        answer = generate(query, context)

        if self.use_guardrails and self.output_flows:
            out = asyncio.run(validate_output(query, answer, self.output_flows))
            if out["blocked"]:
                return {
                    "query": query,
                    "context": context,
                    "answer": out["message"],
                    "answered_by": f"output_guardrail ({out['rail']})",
                }

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
            "answered_by": "rag_pipeline",
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    
    rag = RagPipeline()
    result = rag.invoke("Why do we need golden datasets?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")