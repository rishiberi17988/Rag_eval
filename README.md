# RAG Evaluation

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-000000)](https://www.pinecone.io/)
[![DeepEval](https://img.shields.io/badge/Evaluation-DeepEval-6C47FF)](https://deepeval.com/)
[![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-1F6FEB)](https://www.langchain.com/langsmith)

A hands-on framework for evaluating a Retrieval-Augmented Generation (RAG) application across **retrieval, generation, end-to-end quality, safety, latency, cost, and reliability**. It also includes a baseline-versus-candidate regression workflow and online evaluation of LangSmith traces.

The project uses **LangChain, Pinecone, OpenAI models, DeepEval, and LangSmith** to measure RAG quality, diagnose failures, and make changes safer to ship.

## What this project covers

- **Retrieval evaluation:** Contextual Precision and Contextual Recall.
- **Reranking evaluation:** Compare baseline vector retrieval with an LLM-based reranking strategy.
- **Generation evaluation:** Faithfulness and Answer Relevancy, using known-good context to isolate the generator.
- **End-to-end evaluation:** The RAG triad—Faithfulness, Answer Relevancy, and Contextual Relevancy.
- **Application quality:** LLM-as-a-judge checks for correctness, completeness, and style.
- **Safety evaluation:** Scope adherence, sensitive-information leakage/PII, and toxicity.
- **Operational evaluation:** End-to-end latency, time to first token, token usage, cost, and reliability.
- **Regression testing:** Compare a candidate evaluation snapshot against a saved baseline using metric-specific directions, gates, guardrails, and tolerances.
- **Online evaluation:** Periodically score LangSmith production traces and attach evaluation feedback to runs.
- **Failure-focused observability:** Send failed DeepEval test cases and their failure reasons to LangSmith without tracing every normal model call.

## Evaluation map

| Evaluation area | What it answers | Main implementation |
|---|---|---|
| Retrieval | Did we retrieve the evidence needed to answer? | `evals/eval_retriever.py` |
| Retrieval + reranking | Does LLM reranking improve retrieved context? | `evals/eval_retriever_with_reranker.py` |
| Generator in isolation | Can the model answer faithfully with good context? | `evals/eval_generator.py` |
| Full RAG pipeline | Are retrieval and generation working together? | `evals/eval_rag_pipeline.py` |
| Application correctness | Is the final response correct and useful? | `evals/eval_application.py` |
| Safety | Does the application stay in scope and avoid unsafe output or leakage? | `evals/eval_safety.py` |
| Operations | Is the application fast, affordable, and reliable? | `evals/eval_ops.py` |
| Full evaluation snapshot | What is the quality/safety/ops state of one pipeline version? | `evals/run_suite.py` |
| Regression decision | Did the candidate regress against the baseline? | `evals/compare.py` |
| Online trace evaluation | How well do recent LangSmith runs perform? | `evals/eval_online.py` |

## Architecture

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
                         Retrieve documents
                                |
                         Retrieved context
                                |
                         Answer generation
                                |
                            RAG output
                                |
          +---------------------+----------------------+
          |                     |                      |
      Quality evals         Safety evals          Ops evals
      DeepEval judges       DeepEval judges       Latency / cost /
      Golden datasets       Golden datasets       reliability
          |                     |                      |
          +---------------------+----------------------+
                                |
                     Snapshot + metric registry
                                |
                     Baseline / candidate compare
                                |
                       PASS / REVIEW / FAIL

LangSmith:
  - Failed offline evaluation cases are logged with failure reasons.
  - The online worker reads existing production traces and writes
    Faithfulness, Answer Relevancy, and Contextual Relevancy feedback.
```

**Implementation note:** The current `RagPipeline` uses the retriever returned by `build_retriever()`. The reranker has a separate implementation and evaluation entry point, so the baseline and reranked retrieval approaches can be compared independently.

## Repository structure

```text
Rag_eval/
├── baselines/
│   └── baseline.json             # Saved reference evaluation snapshot
├── data/                         # VTT transcripts used as the knowledge base
├── goldens/                      # Curated evaluation datasets
├── src/
│   ├── retriever.py              # Transcript ingestion, chunking, Pinecone index
│   ├── reranker.py               # LLM-based reranking retriever
│   ├── generator.py              # Grounded answer generation
│   ├── rag_pipeline.py           # Retrieve → generate pipeline
│   ├── llm.py                    # Shared OpenAI clients and model settings
│   └── tracing.py                # LangSmith tracing configuration
├── evals/
│   ├── harness.py                # Shared DeepEval judge, summaries, failure logging
│   ├── metric_registry.py        # Metric direction, gate/guardrail, tolerances
│   ├── run_suite.py              # Runs the full suite and writes a snapshot
│   ├── compare.py                # Baseline/candidate regression decision
│   ├── eval_online.py             # Online scoring of LangSmith traces
│   ├── eval_retriever.py
│   ├── eval_retriever_with_reranker.py
│   ├── eval_generator.py
│   ├── eval_rag_pipeline.py
│   ├── eval_application.py
│   ├── eval_safety.py
│   ├── eval_ops.py
│   └── ...                       # Focused quality, safety, and ops evaluations
├── requirements.txt
└── README.md
```

## Getting started

### 1. Prerequisites

- Python 3.9+
- An OpenAI API key
- A Pinecone API key and a Pinecone serverless region
- A LangSmith API key if you want failed-case logging or online trace evaluation

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

# LangSmith: enables failure-only evaluation logging
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-eval
```

The Pinecone index is created by the retriever when it does not already exist. The configured embedding model must match the index dimension. The default embedding model is `text-embedding-3-small` (1536 dimensions).

**Tracing behavior:** The project deliberately disables automatic LangSmith tracing for normal pipeline/model calls. When LangSmith is configured, failed DeepEval test cases are logged with their inputs, outputs, failed metrics, and reasons. This avoids sending every successful evaluation/model call to LangSmith.

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

### Run individual evaluation suites

```bash
# Retrieval: Contextual Precision and Contextual Recall
python -m evals.eval_retriever

# Retrieval with LLM reranking
python -m evals.eval_retriever_with_reranker

# Generator in isolation
python -m evals.eval_generator

# End-to-end RAG triad
python -m evals.eval_rag_pipeline

# Application correctness, completeness, and style
python -m evals.eval_application

# Safety
python -m evals.eval_safety

# Latency, cost, and reliability
python -m evals.eval_ops
```

### Run the complete evaluation suite

The orchestrator builds the RAG pipeline once and injects that same instance into the component, pipeline, application, safety, and operational evaluations. It flattens the results into a single versioned snapshot.

```bash
# Write a candidate snapshot to baselines/candidate.json
python -m evals.run_suite

# Save a reference baseline (use intentionally)
python -m evals.run_suite --baseline

# Add a human-readable label and include all informational metrics
python -m evals.run_suite --label "reranker experiment" --full

# Write to a custom path
python -m evals.run_suite --out results/experiment.json
```

By default, the snapshot retains metrics that participate in regression decisions. Use `--full` to also include informational metrics such as score distributions and secondary operational measurements.

### Compare baseline and candidate

```bash
python -m evals.compare

# Compare custom snapshots and show informational metrics too
python -m evals.compare --baseline baselines/baseline.json --candidate baselines/candidate.json --all
```

Each metric has a rule in `evals/metric_registry.py`:

- **Direction:** higher-is-better or lower-is-better.
- **Gate:** a regression blocks promotion and produces `FAIL`.
- **Guardrail:** a regression beyond tolerance requires human review and produces `REVIEW`.
- **Info:** reported for visibility but does not affect the verdict.
- **Tolerance:** filters out small changes likely caused by evaluation noise.

The comparison exits with code `0` for PASS, `1` for FAIL, and `2` for REVIEW, so it can be used as a command-line release gate.

The checked-in baseline is a reference snapshot, not a universal quality target. Re-run and version your own baseline when you intentionally change the evaluation dataset, model, or evaluation setup.

## Online evaluation with LangSmith

`evals/eval_online.py` is a polling worker for existing LangSmith root chain runs. It periodically:

1. Reads recent runs from the configured LangSmith project (`cx doubt solver` in the current script).
2. Selects runs with an answer and retrieval context.
3. Scores each run with **Faithfulness**, **Answer Relevancy**, and **Contextual Relevancy** using DeepEval.
4. Writes each score and its reason back to the LangSmith run as feedback.
5. Checks existing feedback keys so already-scored metrics are not needlessly re-evaluated.

Run it as a long-lived worker:

```bash
python -m evals.eval_online
```

The worker polls every 60 seconds. It is intended to run against a LangSmith project that already contains compatible root chain traces with `query`, `answer`, and `context` fields. Update the `PROJECT`, judge model, threshold, and sampling settings in `evals/eval_online.py` to match your tracing environment. Online LLM evaluation can add API cost; use an appropriate sampling rate for production traffic.

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
- **Safety metrics:** Scope adherence, sensitive-information leakage, and toxicity.
- **Online feedback:** Attaching evaluation scores and explanations to individual LangSmith runs.

## Learning outcomes

By working through this repository, you can learn how to:

1. Evaluate retrieval and generation independently instead of treating RAG as one black box.
2. Compare a baseline retriever with an LLM-based reranking strategy.
3. Build and use golden datasets for repeatable evaluation.
4. Apply DeepEval metrics and a custom LLM judge.
5. Log failed evaluation cases to LangSmith and score existing traces online.
6. Build a repeatable evaluation snapshot and compare it with a baseline.
7. Define release gates, quality guardrails, and noise tolerances for RAG metrics.
8. Measure quality, safety, latency, cost, and reliability as separate dimensions.

## Notes

- Evaluation scores depend on the golden dataset, judge model, metric threshold, and model version. Treat them as measurements for a defined setup, not universal quality guarantees.
- LLM-as-a-judge evaluations call external models and can incur API costs.
- Online evaluation runs repeatedly and can incur additional costs; tune its sampling rate and polling cadence.
- Keep `.env` and credentials private. Do not commit API keys or sensitive trace payloads.

## License

See [LICENSE](LICENSE).
