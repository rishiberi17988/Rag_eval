# RAG Evaluation

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-000000)](https://www.pinecone.io/)
[![DeepEval](https://img.shields.io/badge/Evaluation-DeepEval-6C47FF)](https://deepeval.com/)
[![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-1F6FEB)](https://www.langchain.com/langsmith)

A hands-on project for **evaluating a Retrieval-Augmented Generation (RAG) application beyond answer quality**. It brings together component-level evaluation, end-to-end quality checks, safety testing, and operational measurements using a transcript-based knowledge base.

The project uses **LangChain, Pinecone, OpenAI models, DeepEval, and LangSmith** to study how to measure a RAG system, diagnose failures, and monitor its behavior.

## What this project covers

- **Retrieval evaluation:** Contextual Precision and Contextual Recall to measure whether the retriever finds useful evidence and ranks it well.
- **Reranking evaluation:** Compare the plain vector retriever with an LLM-based reranker that over-retrieves candidates and reorders them.
- **Generation evaluation:** Faithfulness and answer relevancy, with known-good context supplied to isolate generator behavior.
- **End-to-end RAG evaluation:** The RAG triad—Faithfulness, Answer Relevancy, and Contextual Relevancy—on live pipeline outputs.
- **Application quality:** LLM-as-a-judge evaluation for correctness and response quality.
- **Safety evaluation:** Scope adherence, leakage/PII risks, and toxicity checks.
- **Operational evaluation:** Latency (including time to first token), token usage and cost, and reliability (success, error, and retry rates).
- **Tracing and debugging:** Instrument pipeline steps and model calls with LangSmith; inspect per-case evaluation results and failures.

## Evaluation map

| Evaluation area | What it answers | Main implementation |
|---|---|---|
| Retrieval | Did we retrieve the evidence needed to answer? | `evals/eval_retriever.py` |
| Retrieval + reranking | Does LLM reranking improve the retrieved context? | `evals/eval_retriever_with_reranker.py` |
| Generator in isolation | Can the model answer faithfully when given good context? | `evals/eval_generator.py` |
| Full RAG pipeline | Are retrieval and generation working together? | `evals/eval_rag_pipeline.py` |
| Application correctness | Is the final response correct and useful? | `evals/eval_application.py` |
| Safety | Does the application stay in scope and avoid unsafe output or leakage? | `evals/eval_safety.py` |
| Operations | Is the application fast, affordable, and reliable? | `evals/eval_ops.py` |

There are also focused scripts for latency, cost, reliability, scope, leakage, and toxicity under `evals/`.

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
                     +----------+----------+
                     |                     |
                Plain retrieval      Over-retrieve
                                           |
                                      LLM reranker
                     |                     |
                     +----------+----------+
                                |
                         Retrieved context
                                |
                         Answer generation
                                |
                         RAG response
                                |
          +---------------------+----------------------+
          |                     |                      |
      Quality evals         Safety evals          Ops evals
      DeepEval judges       DeepEval judges       Latency / cost /
      Golden datasets       Golden datasets       reliability
          |                     |                      |
          +---------------------+----------------------+
                                |
                         LangSmith traces
```

**Implementation note:** `src/rag_pipeline.py` currently wires the default pipeline to the plain Pinecone retriever. The reranking retriever is implemented separately in `src/reranker.py` and has its own evaluation entry point, so the two retrieval approaches can be assessed independently.

## Repository structure

```text
Rag_eval/
├── data/                         # VTT transcripts used as the knowledge base
├── goldens/                      # Curated evaluation datasets
│   ├── correctness_goldens.json
│   ├── faithfulness_dataset.json
│   ├── leakage_goldens.json
│   ├── retriever_goldens.json
│   ├── scope_goldens.json
│   └── toxicity_goldens.json
├── src/
│   ├── retriever.py              # Transcript ingestion, chunking, Pinecone index
│   ├── reranker.py               # LLM-based reranking retriever
│   ├── generator.py              # Grounded answer generation
│   ├── rag_pipeline.py           # Retrieve → generate pipeline
│   ├── llm.py                    # Shared OpenAI clients and model settings
│   └── tracing.py                # LangSmith tracing configuration
├── evals/
│   ├── harness.py                # Shared DeepEval judge, golden loader, summaries
│   ├── eval_retriever.py         # Contextual Precision / Recall
│   ├── eval_retriever_with_reranker.py
│   ├── eval_generator.py         # Generator-only evaluation
│   ├── eval_rag_pipeline.py      # RAG triad
│   ├── eval_application.py       # Application correctness
│   ├── eval_safety.py            # Combined safety evaluation
│   ├── eval_ops.py               # Combined operational snapshot
│   └── ...                       # Focused quality, safety, and ops evaluations
├── requirements.txt
└── README.md
```

## Getting started

### 1. Prerequisites

- Python 3.9+
- An OpenAI API key
- A Pinecone API key and a Pinecone serverless region
- A LangSmith API key (optional; required only if you want hosted tracing)

### 2. Clone and install

```bash
git clone https://github.com/rishiberi17988/Rag_eval.git
cd Rag_eval

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

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

# Optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-eval
```

The Pinecone index is created by the retriever when it does not already exist. The configured embedding model must match the index dimension. The defaults in the code use `text-embedding-3-small` (1536 dimensions); choose a compatible index name if changing embedding models.

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

### Run evaluation suites

```bash
# Retrieval quality: Contextual Precision and Contextual Recall
python -m evals.eval_retriever

# Retrieval quality after LLM reranking
python -m evals.eval_retriever_with_reranker

# Generator in isolation (uses golden contexts)
python -m evals.eval_generator

# End-to-end RAG triad
python -m evals.eval_rag_pipeline

# Application correctness
python -m evals.eval_application

# Combined safety evaluation
python -m evals.eval_safety

# Combined latency, cost, and reliability snapshot
python -m evals.eval_ops
```

Focused scripts are also available, including `evals/eval_latency.py`, `evals/eval_cost.py`, and `evals/eval_reliability.py`. Check each script's configuration before running it; LLM-based evaluations make model/API calls and may incur usage costs.

## Evaluation concepts explored

- **Contextual Precision:** Whether relevant retrieved chunks are ranked ahead of irrelevant chunks.
- **Contextual Recall:** Whether the retrieved context contains the information needed to answer the question.
- **Faithfulness:** Whether answer claims are supported by the supplied context.
- **Answer Relevancy:** Whether the response addresses the user's question.
- **Contextual Relevancy:** Whether the retrieved context is relevant to the question.
- **LLM-as-a-judge:** Using a model and evaluation criteria to assess responses that cannot be reliably checked with exact string matching.
- **Golden datasets:** Curated examples used as repeatable evaluation cases and regression references.
- **Operational metrics:** Latency distributions and percentiles, time to first token, token-based cost estimates, and reliability rates.
- **Safety metrics:** Scope adherence, sensitive-information leakage, and toxicity.

## Learning outcomes

By working through this repository, you can learn how to:

1. Evaluate retrieval and generation independently instead of treating RAG as one black box.
2. Compare a baseline retriever with an LLM-based reranking strategy.
3. Build and use golden datasets for repeatable evaluation.
4. Apply DeepEval metrics and a custom LLM judge.
5. Trace retrieval, generation, and evaluation runs with LangSmith.
6. Measure quality, safety, latency, cost, and reliability as separate dimensions.
7. Use evaluation results to identify which part of a RAG application needs improvement.

## Notes

- Evaluation scores depend on the golden dataset, judge model, metric threshold, and model version. Treat them as measurements for a defined setup, not universal quality guarantees.
- LLM-as-a-judge evaluations call external models and can incur API costs.
- Keep `.env` and credentials private. Do not commit API keys.

## License

See [LICENSE](LICENSE).
