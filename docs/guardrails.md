# Runtime Guardrails

This guide documents the runtime guardrail layer implemented with NVIDIA NeMo Guardrails.

## Why guardrails are in the RAG pipeline

Offline evaluation measures the behavior of a system against a curated test set. Runtime guardrails enforce policy on each live request and generated response. They serve different purposes:

- **Offline evaluation:** measures quality, safety, and operational behavior over a repeatable dataset.
- **Runtime guardrails:** decide whether an individual input or output is allowed to continue to the next stage.

The runtime flow is implemented in `src/rag_pipeline.py`; NeMo configuration and flow definitions are under `guardrails/config/`.

## Request lifecycle

```text
Input query
   |
   v
Input rails (concurrent)
   |
   +-- Jailbreak detection
   +-- Prompt-injection detection
   +-- PII detection
   +-- Data-leakage detection
   +-- Toxicity detection
   +-- Topic restriction
   |
   v
Pinecone retrieval
   |
   v
Grounded answer generation
   |
   v
Output rails (concurrent)
   |
   +-- PII detection
   +-- Data-leakage detection
   +-- Toxicity detection
   +-- Bias detection
   |
   v
Return answer or refusal
```

### Input checks

The default input flow tuple in `src/rag_pipeline.py` contains:

| Flow | Purpose |
|---|---|
| Jailbreak | Detect attempts to bypass the application's rules |
| Prompt injection | Detect instruction overrides and malicious instructions |
| PII | Detect personal information in the incoming query |
| Data leakage | Detect requests for system prompts, secrets, or keys |
| Toxicity | Detect abusive or harmful input |
| Topic | Restrict the assistant to the intended course domain |

### Output checks

| Flow | Purpose |
|---|---|
| PII | Prevent responses containing personal information |
| Data leakage | Prevent disclosure of confidential information or secrets |
| Toxicity | Prevent harmful or abusive output |
| Bias | Detect unfair stereotyping or biased output |

The NeMo configuration also defines a rate-limit input flow, but it is not currently included in `DEFAULT_INPUT_FLOWS` in `src/rag_pipeline.py`. Do not assume rate limiting is active in the default runtime path.

## Concurrency and latency

`validate_input()` and `validate_output()` use `asyncio.gather()` to run their independent flows concurrently. In the ideal case, stage latency is closer to the slowest flow than to the sum of all flow latencies. Actual latency still depends on model/API concurrency limits, network time, model queueing, and NeMo execution overhead.

PII and data-leakage flows perform a deterministic regex check first and invoke a model-backed check only if the regex does not block the content. The other configured checks use model-backed self-check flows.

A few practical optimization principles:

1. Keep deterministic checks (regex, schema, length and simple policy rules) out of LLM calls.
2. Run independent checks concurrently, as the current input/output validators do.
3. Avoid duplicate checks that enforce the same policy at the same stage.
4. Measure each flow and the total input/output stage separately before changing the policy.
5. Preserve fail-closed behavior for checks that protect secrets or sensitive information; don't skip a required check just to reduce latency.
6. Test both false positives and false negatives after changing prompts, models, or flow composition.

## Configuration

The configuration is split across:

- `guardrails/config/config.yml`: NeMo model configuration and rail activation settings.
- `guardrails/config/rails.co`: flow definitions and user-facing refusal messages.
- `src/rag_pipeline.py`: the exact default input/output flow tuples, concurrent execution, and integration with retrieval/generation.

The flow definitions include a rate-limit flow, but the current `RagPipeline` default tuple omits it. To activate or deactivate a flow, update the runtime tuple deliberately and keep the NeMo configuration consistent.

### Enable or disable guardrails

Guardrails are enabled by default:

```env
USE_GUARDRAILS=1
```

For controlled local experiments, they can be disabled:

```env
USE_GUARDRAILS=0
```

Do not disable runtime guardrails in a production deployment as a latency workaround without an explicit security review.

### Custom flow selection

`RagPipeline` accepts `input_flows` and `output_flows` arguments, so a test or experiment can supply a different set of flow names:

```python
from src.rag_pipeline import RagPipeline

rag = RagPipeline(
    input_flows=(
        "jailbreak check input",
        "prompt injection check input",
        "pii check input",
    ),
    output_flows=(
        "pii check output",
        "data leakage check output",
    ),
)
```

Only use flow names that are defined in the NeMo configuration.

## Failure behavior

- If an input flow blocks a query, the pipeline returns the rail's refusal message and does not retrieve documents or generate an answer.
- If an output flow blocks a generated answer, the pipeline returns the rail's refusal message instead of the generated answer.
- The validators gather all flow results before selecting the first failed result in the configured flow order. The checks execute concurrently, but the selected refusal is deterministic with respect to that tuple order.

## Relationship to LangSmith

LangSmith failure-only evaluation logging is implemented separately in `evals/harness.py`. It records failed DeepEval test cases and their metric-level reasons. It should not be confused with the NeMo runtime guardrail decisions or interpreted as tracing every successful application request.
