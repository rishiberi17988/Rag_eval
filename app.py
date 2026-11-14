import asyncio
import os
from pathlib import Path
import sys
import time

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

nest_asyncio.apply()
load_dotenv(ROOT / ".env", override=True)

from src.generator import generate_stream  # noqa: E402
from src.rag_pipeline import (  # noqa: E402
    RagPipeline,
    validate_input,
    validate_output,
)
from src.tracing import _ls_traceable, tracing_context, tracing_enabled  # noqa: E402

os.environ["LANGSMITH_PROJECT"] = "rag-eval"
os.environ["LANGCHAIN_PROJECT"] = "rag-eval"

st.set_page_config(page_title="RAG Chatbot with Guardrails", layout="wide")
st.title("LLM Evals RAG Chatbot with 8 Guardrails")

GUARDRAILS = {
    "Rate limiting": (["rate limit input"], [], True),
    "Jailbreak detection": (["jailbreak check input"], [], True),
    "Prompt injection detection": (["prompt injection check input"], [], True),
    "PII protection": (["pii check input"], ["pii check output"], True),
    "Data leakage prevention": (["data leakage check input"], ["data leakage check output"], True),
    "Toxicity detection": (["toxicity check input"], ["toxicity check output"], True),
    "Topic restriction": (["topic check input"], [], True),
    "Bias detection": ([], ["bias check output"], True),
}


@st.cache_resource(show_spinner="Loading RAG pipeline (Pinecone retriever + generator)…")
def get_pipeline() -> RagPipeline:
    return RagPipeline()


@_ls_traceable(name="rag_pipeline_stream", run_type="chain")
def stream_rag(text: str, answer_placeholder) -> dict:
    docs = get_pipeline().retriever.invoke(text)
    context = [doc.page_content for doc in docs]
    bot = answer_placeholder.write_stream(generate_stream(text, context))
    return {"query": text, "context": context, "answer": bot}


@_ls_traceable(name="chat_turn", run_type="chain")
async def _run_chat_turn(
    text: str,
    input_flows: tuple[str, ...],
    output_flows: tuple[str, ...],
    answer_placeholder,
) -> dict:
    inp = await validate_input(text, input_flows)
    if inp["blocked"]:
        answer_placeholder.write(inp["message"])
        return {
            "answer": inp["message"],
            "answered_by": f"input_guardrail ({inp['rail']})",
            "context": [],
        }

    rag_out = stream_rag(text, answer_placeholder)
    bot = rag_out["answer"]
    context = rag_out["context"]

    out = await validate_output(text, bot, output_flows)
    if out["blocked"]:
        answer_placeholder.write(out["message"])
        return {
            "answer": out["message"],
            "answered_by": f"output_guardrail ({out['rail']})",
            "context": [],
        }
    return {"answer": bot, "answered_by": "rag_pipeline", "context": context}


def chat(
    text: str,
    input_flows: tuple[str, ...],
    output_flows: tuple[str, ...],
    answer_placeholder,
) -> dict:
    with tracing_context(enabled=tracing_enabled(), project_name="rag-eval"):
        return asyncio.run(
            _run_chat_turn(text, input_flows, output_flows, answer_placeholder)
        )


def render_context(context: list[str]) -> None:
    if not context:
        return
    with st.expander(f"📚 {len(context)} retrieved chunk{'s' if len(context) != 1 else ''}"):
        for i, chunk in enumerate(context, start=1):
            st.markdown(f"**Chunk {i}**")
            st.text(chunk)
            if i != len(context):
                st.divider()


# Sidebar guardrail controls
with st.sidebar:
    st.header("Active Guardrails")
    active_input, active_output = [], []
    for name, (in_flows, out_flows, default) in GUARDRAILS.items():
        if st.checkbox(name, value=default):
            active_input.extend(in_flows)
            active_output.extend(out_flows)

    st.divider()
    show_context = st.toggle("Show retrieved context", value=True)

    rate_state = sys.modules.get("_guardrail_rate_state")
    timestamps = getattr(rate_state, "timestamps", None)
    now = time.monotonic()
    recent_count = sum(1 for t in (timestamps or ()) if now - t < 60)
    st.caption(f"Rate limit usage (60s): {recent_count} / 5")
    if st.button("Reset Rate Limit") and timestamps is not None:
        timestamps.clear()
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("answered_by"):
            st.caption(f"Answered by: `{m['answered_by']}`")
        if m["role"] == "assistant" and show_context and m.get("context"):
            render_context(m["context"])

if user_input := st.chat_input("Ask a question about LLM / RAG evaluation..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        with st.spinner("Checking guardrails…"):
            result = chat(
                user_input,
                tuple(active_input),
                tuple(active_output),
                answer_placeholder,
            )
        st.caption(f"Answered by: `{result['answered_by']}`")
        if show_context and result.get("context"):
            render_context(result["context"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "answered_by": result["answered_by"],
            "context": result.get("context", []),
        }
    )
