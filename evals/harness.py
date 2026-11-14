# evals/harness.py
"""
Shared harness for the eval suite.

The eval files each run their own DeepEval evaluation and return results; this
module holds the pieces they all share:

  load_goldens(path)          -- read a golden JSON file (resolves goldens/ <-> golden/)
  summarize_by_metric(result) -- turn a DeepEval EvaluationResult into a
                                 PER-METRIC summary (pass rate + score stats)
                                 and log ONLY failed test cases to LangSmith
  print_summary(title, sum)   -- print a readable per-metric recap + flush traces
  OpenAIJudge / JUDGE_MODEL   -- DeepEval judge (gpt-6.1-terra)
"""

import glob
import json
import os
import re
import sys

from deepeval.models import DeepEvalBaseLLM

from src.llm import chat, achat
from src.retriever import DATA_DIR
from src.tracing import (
    traceable,
    _ls_traceable,
    tracing_context,
    tracing_enabled,
    flush_traces,
)

JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL", "gpt-6.1-terra")


# ---------------------------------------------------------------- judge
class OpenAIJudge(DeepEvalBaseLLM):
    """DeepEval judge that calls the OpenAI chat API through src/llm.py.

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
def _resolve_golden_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if path.startswith("goldens/"):
        alt = "golden/" + path[len("goldens/"):]
        if os.path.exists(alt):
            return alt
    elif path.startswith("golden/"):
        alt = "goldens/" + path[len("golden/"):]
        if os.path.exists(alt):
            return alt
    return path


def load_goldens(path):
    resolved = _resolve_golden_path(path)
    with open(resolved, encoding="utf-8") as f:
        return json.load(f)


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


# ---------------------------------------------------------------- LangSmith failure-only logging
def _extract_no_verdicts(metric_obj):
    """Pull individual 'no' verdicts + their reasons from a DeepEval metric or
    its verbose_logs (works even when include_reason=False)."""
    bad = []
    verdicts = getattr(metric_obj, "verdicts", None)
    if verdicts:
        for v in verdicts:
            verdict_val = str(getattr(v, "verdict", "")).strip().lower()
            if verdict_val == "no":
                reason = getattr(v, "reason", None) or getattr(v, "statement", None) or str(v)
                bad.append(reason)
        if bad:
            return bad

    verbose_logs = getattr(metric_obj, "verbose_logs", None) or ""
    if verbose_logs:
        for match in re.finditer(
            r'\{\s*"verdict"\s*:\s*"no"\s*,\s*"(?:reason|statement)"\s*:\s*"([^"]+)"',
            verbose_logs,
            flags=re.IGNORECASE,
        ):
            bad.append(match.group(1))
    return bad


@_ls_traceable(run_type="tool", name="failed_test_case", tags=["failed", "eval"])
def _trace_failed_case(query, expected_output, actual_output, retrieval_context, failed_metrics):
    """Logs a single failed test case + its failure reasons to LangSmith."""
    return {
        "query": query,
        "expected_output": expected_output,
        "actual_output": actual_output,
        "retrieval_context": retrieval_context,
        "failed_metrics": failed_metrics,
    }


_LOGGED_RESULT_IDS = set()


def log_failures_to_langsmith(result):
    """Inspect a DeepEval EvaluationResult and send ONLY failed test cases to
    LangSmith. Idempotent per result object so hooks + summarize_by_metric
    never double-log."""
    if not tracing_enabled() or result is None:
        return
    if id(result) in _LOGGED_RESULT_IDS:
        return
    _LOGGED_RESULT_IDS.add(id(result))

    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    logged_any = False
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        failed_metrics = []
        for m in metrics:
            if not bool(getattr(m, "success", False)):
                name = getattr(m, "name", "unknown")
                score = getattr(m, "score", None)
                threshold = getattr(m, "threshold", None)
                reason = getattr(m, "reason", None) or getattr(m, "error", None)
                no_verdicts = _extract_no_verdicts(m)
                failed_metrics.append({
                    "metric": name,
                    "score": score,
                    "threshold": threshold,
                    "reason": reason or ("; ".join(no_verdicts) if no_verdicts else "Below threshold"),
                    "wrong_verdicts": no_verdicts,
                })

        if failed_metrics:
            logged_any = True
            with tracing_context(enabled=True, project_name="rag-eval"):
                _trace_failed_case(
                    query=getattr(tr, "input", None) or getattr(tr, "name", "unknown"),
                    expected_output=getattr(tr, "expected_output", None),
                    actual_output=getattr(tr, "actual_output", None),
                    retrieval_context=getattr(tr, "retrieval_context", None),
                    failed_metrics=failed_metrics,
                )

    if logged_any:
        flush_traces()


def _install_deepeval_hooks():
    """1. Route string model names through OpenAIJudge.
       2. Hook deepeval.evaluate so even standalone eval files (eval_toxicity,
          eval_leakage, eval_scope_safety) automatically log failed test cases
          to LangSmith."""
    try:
        from deepeval.metrics import utils as de_utils
        orig_init = de_utils.initialize_model
        if not getattr(orig_init, "_rag_eval_wrapped", False):
            def _patched_initialize_model(model=None, *args, **kwargs):
                if model is None or isinstance(model, str):
                    model = OpenAIJudge(model or JUDGE_MODEL_NAME)
                return orig_init(model, *args, **kwargs)

            _patched_initialize_model._rag_eval_wrapped = True
            for mod_name, mod in list(sys.modules.items()):
                if mod_name.startswith("deepeval") and hasattr(mod, "initialize_model"):
                    setattr(mod, "initialize_model", _patched_initialize_model)
    except Exception:
        pass

    try:
        import deepeval
        orig_eval = deepeval.evaluate
        if not getattr(orig_eval, "_rag_eval_wrapped", False):
            def _patched_evaluate(*args, **kwargs):
                res = orig_eval(*args, **kwargs)
                log_failures_to_langsmith(res)
                return res

            _patched_evaluate._rag_eval_wrapped = True
            for mod_name, mod in list(sys.modules.items()):
                if hasattr(mod, "evaluate") and getattr(mod, "evaluate", None) is orig_eval:
                    setattr(mod, "evaluate", _patched_evaluate)
    except Exception:
        pass


_install_deepeval_hooks()


def summarize_by_metric(result):
    """DeepEval EvaluationResult -> {metric_name: {n, pass_rate, avg/min/max score}}."""
    log_failures_to_langsmith(result)

    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets = {}   # metric_name -> {"scores": [...], "passed": int, "total": int}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def print_summary(title, summary):
    """Readable per-metric recap, printed under DeepEval's own report when a file
    is run directly."""
    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)
    for name, s in summary.items():
        avg = f"{s['avg_score']:.2f}" if s["avg_score"] == s["avg_score"] else "nan"   # nan != nan
        print(f"  {name:<26} pass_rate={s['pass_rate']:5.0f}%  avg={avg}  n={s['n']}")
    print("=" * 60)
    flush_traces()
