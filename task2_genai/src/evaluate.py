"""
evaluate.py
-----------
Task 2C - Evaluation and Baseline Comparison.

  * ROUGE-L on the held-out test set, base model vs fine-tuned model.
  * An additional metric: LLM-as-judge, scoring each response on a defined
    rubric (faithfulness, citation accuracy, escalation correctness) and
    returning structured, Pydantic-validated JSON.
  * A manual hallucination-rate review over >= 10 responses.
  * A qualitative analysis scaffold (fill in with specific examples after running).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List

import requests
from pydantic import ValidationError
from rouge_score import rouge_scorer

from . import config
from .schemas import JudgeScore, ManualReviewEntry, ReviewLabel, RougeComparisonRow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator scoring a compliance-assistant model's \
response against a reference answer and the source policy excerpt. Respond with ONLY a valid \
JSON object with exactly these keys: "faithfulness" (1-5: does the response only use facts from \
the excerpt?), "citation_accuracy" (1-5: is cited_section correct?), "escalation_correctness" \
(1-5: is requires_escalation appropriate given the question and excerpt?), "overall" (1-5), and \
"reasoning" (one or two sentences explaining the scores)."""

JUDGE_USER_TEMPLATE = """Policy excerpt:
\"{policy_excerpt}\"

Employee question: \"{employee_question}\"

Reference (gold) answer JSON:
{reference_json}

Model's response JSON to evaluate:
{model_response_json}

Return the JSON score object now."""


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------
def compute_rouge_l(predictions: List[str], references: List[str]) -> List[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return scores


def build_rouge_comparison_table(
    base_predictions: List[str], finetuned_predictions: List[str], references: List[str]
) -> List[RougeComparisonRow]:
    base_scores = compute_rouge_l(base_predictions, references)
    ft_scores = compute_rouge_l(finetuned_predictions, references)
    return [
        RougeComparisonRow(example_id=i, base_model_rouge_l=b, fine_tuned_rouge_l=f)
        for i, (b, f) in enumerate(zip(base_scores, ft_scores))
    ]


def summarize_rouge_table(rows: List[RougeComparisonRow]) -> Dict[str, float]:
    if not rows:
        return {"base_mean": 0.0, "fine_tuned_mean": 0.0, "delta": 0.0}
    base_mean = sum(r.base_model_rouge_l for r in rows) / len(rows)
    ft_mean = sum(r.fine_tuned_rouge_l for r in rows) / len(rows)
    return {"base_mean": round(base_mean, 4), "fine_tuned_mean": round(ft_mean, 4), "delta": round(ft_mean - base_mean, 4)}


# ---------------------------------------------------------------------------
# Additional metric: LLM-as-judge (structured JSON output)
# ---------------------------------------------------------------------------
def _call_judge(system_prompt: str, user_prompt: str, max_retries: int = 4) -> str | None:
    if config.JUDGE_PROVIDER == "groq":
        url, api_key, model = config.GROQ_BASE_URL, config.GROQ_API_KEY, config.JUDGE_MODEL_GROQ
    else:
        url, api_key, model = config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY, config.TEACHER_MODEL_OPENROUTER

    if not api_key:
        logger.error("No API key configured for judge provider '%s'.", config.JUDGE_PROVIDER)
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=config.LLM_REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            is_rate_limit = "429" in str(exc)
            wait_seconds = (2 ** attempt) if is_rate_limit else 0
            logger.warning(
                "Judge call attempt %d/%d failed: %s%s",
                attempt + 1, max_retries, exc,
                f" - retrying in {wait_seconds}s" if is_rate_limit and attempt < max_retries - 1 else "",
            )
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(wait_seconds)
            elif not is_rate_limit:
                break
    return None


def _extract_json(raw: str) -> Dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judge_response(
    policy_excerpt: str, employee_question: str, reference: Dict[str, Any], model_response: Dict[str, Any]
) -> JudgeScore | None:
    user_prompt = JUDGE_USER_TEMPLATE.format(
        policy_excerpt=policy_excerpt,
        employee_question=employee_question,
        reference_json=json.dumps(reference),
        model_response_json=json.dumps(model_response),
    )
    raw = _call_judge(JUDGE_SYSTEM_PROMPT, user_prompt)
    parsed = _extract_json(raw) if raw else None
    if parsed is None:
        logger.warning("Could not parse judge JSON response")
        return None
    try:
        return JudgeScore.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("Judge score validation failed: %s", exc)
        return None


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Optional alternative additional metric. Requires network access to download
    the `bert-base-uncased` scoring model on first call - use LLM-as-judge above
    if running offline / on a slow connection.
    """
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}


# ---------------------------------------------------------------------------
# Hallucination rate (manual review)
# ---------------------------------------------------------------------------
def compute_hallucination_rate(reviews: List[ManualReviewEntry]) -> Dict[str, Any]:
    if not reviews:
        return {"hallucination_rate_pct": 0.0, "n_reviewed": 0}
    if len(reviews) < config.MIN_MANUAL_REVIEW_SAMPLES:
        logger.warning(
            "Only %d manual reviews provided (< required minimum of %d).",
            len(reviews), config.MIN_MANUAL_REVIEW_SAMPLES,
        )
    n = len(reviews)
    n_hallucinated = sum(1 for r in reviews if r.label == ReviewLabel.HALLUCINATED)
    n_correct = sum(1 for r in reviews if r.label == ReviewLabel.CORRECT)
    n_partial = sum(1 for r in reviews if r.label == ReviewLabel.PARTIALLY_CORRECT)
    return {
        "n_reviewed": n,
        "n_correct": n_correct,
        "n_partially_correct": n_partial,
        "n_hallucinated": n_hallucinated,
        "hallucination_rate_pct": round(100 * n_hallucinated / n, 1),
    }


def save_manual_review_template(example_outputs: List[str], path: str) -> None:
    """
    Write a CSV template the candidate fills in by hand while reading each
    fine-tuned model response (Task 2C requires >= 10 manually reviewed and
    labelled responses - this cannot and should not be automated).
    """
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["example_id", "model_output", "label (correct/partially_correct/hallucinated)", "notes"])
        for i, output in enumerate(example_outputs):
            writer.writerow([i, output, "", ""])
    logger.info("Manual review template written to %s - fill in the 'label' column by hand.", path)


QUALITATIVE_ANALYSIS_TEMPLATE = """## Qualitative Analysis

### Where fine-tuning improved behaviour
_Fill in with 2-3 specific before/after examples from your test set: e.g. "on example #{{id}}, \
the base model invented a $5,000 threshold not present in the excerpt, while the fine-tuned model \
correctly cited the $1,000 threshold and set requires_escalation=false."_

### Remaining failure modes and next steps
_Fill in with the failure patterns you actually observed in the manual review CSV, and what \
additional data or training strategy (more examples for a specific topic, harder negative \
examples, RLHF/DPO pass, larger LoRA rank, etc.) would address them._
"""


if __name__ == "__main__":
    print("This module exposes compute_rouge_l, judge_response, compute_hallucination_rate.")
    print("Run it via the Task2 notebook against your actual base/fine-tuned model outputs.")
