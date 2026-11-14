# RAG Evaluation

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-000000)](https://www.pinecone.io/)
[![DeepEval](https://img.shields.io/badge/Evaluation-DeepEval-6C47FF)](https://deepeval.com/)
[![NeMo Guardrails](https://img.shields.io/badge/Guardrails-NeMo-76B900)](https://github.com/NVIDIA/NeMo-Guardrails)
[![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-1F6FEB)](https://www.langchain.com/langsmith)

A hands-on framework for evaluating and hardening a Retrieval-Augmented Generation (RAG) application. The project covers retrieval, generation, application quality, safety, operational performance, configurable runtime guardrails, and baseline-versus-candidate regression testing.

It uses **LangChain, Pinecone, OpenAI models, DeepEval, NVIDIA NeMo Guardrails, and LangSmith** to measure RAG quality, enforce runtime policies, investigate failures, and make changes safer to ship.

## What this project covers

- **Retrieval evaluation:** Contextual Precision and Contextual Recall for the standard Pinecone retriever.
- **Reranking evaluation:** A separate experiment that over-retrieves candidates from Pinecone and reranks them with an LLM.
- **Generator evaluation:** Faithfulness and Answer Relevancy using known-good context, isolating generation from retrieval.
- **End-to-end evaluation:** The active pipeline evaluation currently measures Contextual Relevancy on the context returned by the live RAG pipeline.
- **Application quality:** LLM-as-a-judge checks for correctness, completeness, and style.
- **Safety evaluation:** Scope adherence, protected-information/PII leakage, and toxicity.
- **Runtime guardrails:** Parallel input and output checks for jailbreaks, prompt injection, PII, data leakage, toxicity, topic restriction, and bias.
- **Operational evaluation:** End-to-end latency, time to first token, token usage, estimated cost, and reliability.
- **Regression testing:** Compare a candidate snapshot against a saved baseline using metric-specific directions, gates, guardrails, and tolerances.
- **Failure-focused observability:** Send failed DeepEval test cases and their failure reasons to LangSmith rather than tracing every normal model call.

## Evaluation map

| Evaluation area | What it answers | Entry point |
|---|---|---|
| Standard retrieval | Did the Pinecone retriever find useful evidence? | `evals/eval_retriever.py` |
| Retrieval with reranking | Does LLM reranking improve the retrieved ranking? | `evals/eval_retriever_with_reranker.py` |
| Generator in isolation | Can the model answer faithfully with known-good context? | `evals/eval_generator.py` |
| End-to-end context relevance | Is the context returned by the live pipeline relevant to the query? | `evals/eval_rag_pipeline.py` |
| Application correctness | Is the final response correct, complete, and well written? | `evals/eval_application.py` |
| Safety | Does the application stay in scope and avoid unsafe output or leakage? | `evals/eval_safety.py` |
| Operations | Is the application fast, affordable, and reliable? | `evals/eval_ops.py` |
| Full evaluation snapshot | What is the quality/safety/ops state of one pipeline version? | `evals/run_suite.py` |
| Regression decision | Did the candidate regress against the baseline? | `evals/compare.py` |

> **Scope note:** The reranker evaluation is a separate run; it is not currently included in `evals/run_suite.py`. The end-to-end evaluation currently enables Contextual Relevancy only. Faithfulness and Answer Relevancy are present in the module as commented-out metrics and are not part of the current snapshot.

## Architecture

### RAG application

```text
VTT course transcripts
        |
        v
Load + clean transcripts
        |
        v
Recursive chunking
        |
        v
OpenAI embeddings -----> Pinecone vector index
                                |
                         Retrieve top-k chunks
                                |
                         Retrieved context
                                |
                         Answer generation
                                |
                            RAG output
```

The standard `RagPipeline` uses the Pinecone vector retriever returned by `build_retriever()`. The LLM reranker is implemented separately in `src/reranker.py` and evaluated through its own entry point; it is not silently substituted into the standard pipeline.

### Evaluation and regression

```text
Individual evaluation modules
  |-- Retrieval
  |-- Generator
  |-- End-to-end context relevance
  |-- Application quality
  |-- Safety
  |-- Operations
        |
        v
evals/run_suite.py
  - Builds one RagPipeline instance
  - Reuses its retriever for retrieval evaluation
  - Runs the configured evaluation modules
  - Flattens metrics into a versioned JSON snapshot
        |
        v
baselines/candidate.json
        |
        v
evals/compare.py <---- baselines/baseline.json
        |
        v
PASS / REVIEW / FAIL
```

The metric registry in `evals/metric_registry.py` is the single source of truth for metric direction, decision type, and tolerated change.

### Runtime guardrails

The application uses NVIDIA NeMo Guardrails around the RAG pipeline:

```text
User query
    |
    v
Input guardrails (parallel)
  - Jailbreak
  - Prompt injection
  - PII
  - Data leakage
  - Toxicity
  - Topic restriction
    |
    v
Pinecone retrieval
    |
    v
Answer generation
    |
    v
Output guardrails (parallel)
  - PII
  - Data leakage
  - Toxicity
  - Bias
    |
    v
Response / refusal
```

Independent flows within each stage are invoked concurrently with `asyncio.gather()`. PII and data-leakage flows also run deterministic regex checks before their model-based check. A blocking input result stops the request before retrieval; a blocking output result replaces the generated answer with the guardrail response.

Guardrails are enabled by default. Set `USE_GUARDRAILS=0` to disable them for controlled experiments. The active flow lists are defined in `src/rag_pipeline.py`; the NeMo configuration and flow definitions live under `guardrails/config/`. The rate-limit flow is defined in the NeMo configuration but is not included in the pipeline's current default input-flow tuple.

See [Guardrails documentation](docs/guardrails.md) for configuration, execution behavior, and latency considerations.

## Repository structure

```text
Rag_eval/
├── baselines/
│   └── baseline.json             # Saved reference evaluation snapshot
├── data/                         # VTT transcripts used as the knowledge base
├── goldens/                      # Curated evaluation datasets
├── guardrails/
│   └── config/
│       ├── config.yml            # NeMo model and active rail configuration
│       └── rails.co              # Input/output flow definitions and refusals
├── docs/
│   └── guardrails.md             # Runtime guardrails guide
├── src/
│   ├── retriever.py              # Transcript ingestion, chunking, Pinecone index
│   ├── reranker.py               # Separate LLM-based reranking retriever
│   ├── generator.py              # Grounded answer generation
│   ├── rag_pipeline.py           # Guardrails → retrieve → generate → guardrails
│   ├── llm.py                    # Shared OpenAI clients and model settings
│   └── tracing.py                # LangSmith tracing configuration
├── evals/
│   ├── harness.py                # Shared DeepEval judge, summaries, failure logging
│   ├── metric_registry.py         # Metric direction, gate/guardrail, tolerances
│   ├── run_suite.py              # Runs the snapshot suite
│   ├── compare.py                # Baseline/candidate regression decision
│   ├── eval_retriever.py
│   ├── eval_retriever_with_reranker.py
│   ├── eval_generator.py
│   ├── eval_rag_pipeline.py
│   ├── eval_application.py
│   ├── eval_safety.py
│   ├── eval_ops.py
│   └── ...
├── requirements.txt
└── README.md
```

## Getting started

### 1. Prerequisites

- Python 3.9+
- An OpenAI API key
- A Pinecone API key and a Pinecone serverless region
- A LangSmith API key if you want failure-only evaluation logging

### 2. Clone and install

```bash
git clone https://github.com/rishiberi17988/Rag_eval.git
cd Rag_eval

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\\Scripts\\activate

pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the repository root. Keep API keys out of source control.

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional Pinecone configuration
PINECONE_INDEX_NAME=rag-eval
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Optional model overrides
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=your_supported_chat_model
JUDGE_MODEL=your_supported_judge_model

# Runtime guardrails (enabled by default)
USE_GUARDRAILS=1

# Optional LangSmith failure-only evaluation logging
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-eval
```

The Pinecone index is created by the retriever when it does not already exist. The configured embedding model must match the index dimension. The default embedding model is `text-embedding-3-small` (1536 dimensions).

**Tracing behavior:** The project disables automatic LangSmith tracing for normal pipeline/model calls. When LangSmith is configured, failed DeepEval test cases are logged with their inputs, outputs, failed metrics, and reasons. This avoids sending every successful evaluation/model call to LangSmith. Runtime guardrail checks are not the same as this offline evaluation-failure tracing.

## Run the project

Run commands from the repository root.

### Build / test retrieval

```bash
python -m src.retriever
```

### Run the RAG pipeline

```bash
python -m src.rag_pipeline
```

### Run individual evaluation modules

```bash
# Standard retrieval: Contextual Precision and Contextual Recall
python -m evals.eval_retriever

# Separate retrieval experiment with LLM reranking
python -m evals.eval_retriever_with_reranker

# Generator in isolation, using known-good context
python -m evals.eval_generator

# End-to-end pipeline context relevance
python -m evals.eval_rag_pipeline

# Application correctness, completeness, and style
python -m evals.eval_application

# Scope, leakage/PII, and toxicity
python -m evals.eval_safety

# Latency, cost, and reliability
python -m evals.eval_ops
```

### Run the complete evaluation suite

The orchestrator builds the standard RAG pipeline once and reuses that instance across pipeline, application, safety, and operational evaluations. The retriever evaluation receives the retriever exposed by that pipeline. The generator-isolation evaluation intentionally uses the generator with known-good golden context.

```bash
# Write a candidate snapshot to baselines/candidate.json
python -m evals.run_suite

# Save a reference baseline (use intentionally)
python -m evals.run_suite --baseline

# Add a human-readable label and include informational metrics
python -m evals.run_suite --label "reranker experiment" --full

# Write to a custom path
python -m evals.run_suite --out results/experiment.json
```

By default, the snapshot retains metrics that participate in regression decisions. Use `--full` to also include informational metrics such as score distributions and secondary operational measurements.

### Compare baseline and candidate

```bash
python -m evals.compare

# Compare custom snapshots and show informational metrics too
python -m evals.compare --baseline a.json --candidate b.json --all
```

Each metric has a rule in `evals/metric_registry.py`:

- **Direction:** higher-is-better or lower-is-better.
- **Gate:** a regression blocks promotion and produces `FAIL`.
- **Guardrail:** a regression beyond tolerance requires human review and produces `REVIEW`.
- **Info:** reported for visibility but does not affect the verdict.
- **Tolerance:** filters out small changes likely caused by evaluation noise.

The comparison exits with code `0` for PASS, `1` for FAIL, and `2` for REVIEW, so it can be used as a command-line release gate.

The checked-in baseline is a reference snapshot, not a universal quality target. Re-run and version your own baseline when you intentionally change the evaluation dataset, model, or evaluation setup.

## Evaluation concepts explored

- **Contextual Precision:** Whether relevant retrieved chunks are ranked ahead of irrelevant chunks.
- **Contextual Recall:** Whether retrieved context contains the information needed to answer.
- **Faithfulness:** Whether answer claims are supported by the supplied context.
- **Answer Relevancy:** Whether the response addresses the user's question.
- **Contextual Relevancy:** Whether retrieved context is relevant to the question.
- **LLM-as-a-judge:** Using a model and evaluation criteria to assess responses that cannot be reliably checked with exact string matching.
- **Golden datasets:** Curated examples used as repeatable evaluation cases and regression references.
- **Metric registry:** Explicit rules for interpreting metric changes rather than treating every score as equally important.
- **Operational metrics:** Latency percentiles, time to first token, token-based cost estimates, and reliability rates.
- **Runtime guardrails:** Input/output policy checks that can block unsafe, off-topic, or sensitive content before it reaches the user.

## Learning outcomes

By working through this repository, you can learn how to:

1. Evaluate retrieval and generation independently instead of treating RAG as one black box.
2. Compare a standard retriever with a separate LLM-based reranking strategy.
3. Build and use golden datasets for repeatable evaluation.
4. Apply DeepEval metrics and a custom LLM judge.
5. Add runtime input/output guardrails with NVIDIA NeMo Guardrails.
6. Log failed evaluation cases to LangSmith with metric-level failure reasons.
7. Build repeatable evaluation snapshots and compare them with a baseline.
8. Define release gates, quality guardrails, and noise tolerances for RAG metrics.
9. Measure quality, safety, latency, cost, and reliability as separate dimensions.

## Notes

- Evaluation scores depend on the golden dataset, judge model, metric threshold, and model version. Treat them as measurements for a defined setup, not universal quality guarantees.
- LLM-as-a-judge evaluations and model-backed runtime guardrails call external models and can incur API costs and latency.
- Keep `.env`, credentials, and sensitive trace payloads private. Do not commit API keys.

## License

See [LICENSE](LICENSE).
