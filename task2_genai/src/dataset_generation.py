"""
dataset_generation.py
----------------------
Task 2A - Use Case Definition and Dataset Engineering.

Generates synthetic (policy_excerpt, employee_question, answer, cited_section,
requires_escalation, confidence) examples using a capable teacher model (Groq
Llama-3.3-70B by default), across 8 distinct compliance topics to guarantee
genuine diversity (not near-duplicate variations of one scenario), formats them
as JSONL chat-template examples, and splits 80/10/10.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List

import requests
from pydantic import ValidationError

from . import config
from .schemas import CompliancePolicyExample

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = (
    "You are a compliance policy assistant for a financial services firm. Given a "
    "policy excerpt and an employee question, respond with ONLY a valid JSON object "
    "with exactly these keys: \"answer\" (grounded strictly in the excerpt), "
    "\"cited_section\" (which part of the excerpt supports the answer), "
    "\"requires_escalation\" (true/false - true if the question is not clearly "
    "covered by the excerpt, involves suspected wrongdoing, or is at/above any "
    "stated threshold), and \"confidence\" (\"low\", \"medium\", or \"high\")."
)

TEACHER_GENERATION_SYSTEM_PROMPT = """You are generating synthetic training data for fine-tuning \
a compliance policy assistant used at a financial services firm. You must respond with ONLY a \
valid JSON array (no markdown fences, no commentary) of exactly {batch_size} objects. Each object \
must have exactly these keys:
  - "policy_excerpt": a realistic, self-contained 100-300 word excerpt from an internal compliance \
policy on the topic of {topic_readable}. Invent specific, concrete numeric thresholds, deadlines, \
or procedures (e.g. dollar amounts, day counts, approval chains) - do not write vague policy text.
  - "employee_question": a natural question an employee might ask about that excerpt. Vary the \
scenario significantly across the {batch_size} objects - different amounts, different edge cases, \
different phrasing/tone (some casual, some formal), some questions clearly covered by the excerpt \
and some that are NOT covered (to teach appropriate escalation).
  - "answer": a correct, grounded answer using ONLY facts stated in policy_excerpt.
  - "cited_section": which specific part of the excerpt the answer relies on.
  - "requires_escalation": true or false, following the rule that anything not clearly covered by \
the excerpt, anything involving suspected wrongdoing, or anything at/above a stated threshold must \
be escalated (true).
  - "confidence": "low", "medium", or "high" - use "low" when the excerpt only partially covers the \
question.

Make sure roughly a third of the {batch_size} objects have requires_escalation = true, and vary \
excerpt content substantially - do not reuse the same numbers, entities, or scenario twice."""

TEACHER_GENERATION_USER_PROMPT = "Generate the JSON array now for topic: {topic_readable}."


# ---------------------------------------------------------------------------
# Teacher LLM call
# ---------------------------------------------------------------------------
def _call_teacher(system_prompt: str, user_prompt: str) -> str | None:
    if config.TEACHER_PROVIDER == "groq":
        url, api_key, model = config.GROQ_BASE_URL, config.GROQ_API_KEY, config.TEACHER_MODEL_GROQ
    else:
        url, api_key, model = config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY, config.TEACHER_MODEL_OPENROUTER

    if not api_key:
        logger.error("No API key configured for teacher provider '%s'.", config.TEACHER_PROVIDER)
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": config.TEACHER_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    for attempt in range(1, config.LLM_MAX_RETRIES + 2):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=config.LLM_REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            logger.warning("Teacher call attempt %d failed: %s; response=%s", attempt, exc, detail)
        except requests.RequestException as exc:
            logger.warning("Teacher call attempt %d failed: %s", attempt, exc)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Teacher call attempt %d returned an unexpected response: %s", attempt, exc)
    return None


def _extract_json_array(raw_text: str) -> List[Dict[str, Any]] | None:
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse extracted JSON array: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_examples_for_topic(topic: str, n: int = config.EXAMPLES_PER_TOPIC) -> List[CompliancePolicyExample]:
    topic_readable = topic.replace("_", " ")
    system_prompt = TEACHER_GENERATION_SYSTEM_PROMPT.format(batch_size=n, topic_readable=topic_readable)
    user_prompt = TEACHER_GENERATION_USER_PROMPT.format(topic_readable=topic_readable)

    raw = _call_teacher(system_prompt, user_prompt)
    raw_items = _extract_json_array(raw) if raw else None
    if not raw_items:
        logger.error("Teacher generation failed or returned unparseable JSON for topic=%s", topic)
        return []

    examples: List[CompliancePolicyExample] = []
    for item in raw_items:
        item.setdefault("topic", topic)
        item["topic"] = topic
        try:
            examples.append(CompliancePolicyExample.model_validate(item))
        except ValidationError as exc:
            logger.warning("Dropping invalid generated example for topic=%s: %s", topic, exc)
    return examples


def generate_full_dataset(
    topics: List[str] = config.POLICY_TOPICS,
    per_topic: int = config.EXAMPLES_PER_TOPIC,
) -> List[CompliancePolicyExample]:
    all_examples: List[CompliancePolicyExample] = []
    for topic in topics:
        logger.info("Generating %d examples for topic=%s", per_topic, topic)
        all_examples.extend(generate_examples_for_topic(topic, per_topic))

    if len(all_examples) < config.MIN_TOTAL_EXAMPLES:
        logger.warning(
            "Only generated %d examples (< minimum %d required). Consider re-running "
            "failed topics or lowering per-topic batch size to reduce truncation risk.",
            len(all_examples),
            config.MIN_TOTAL_EXAMPLES,
        )
    return all_examples


# ---------------------------------------------------------------------------
# Diversity analysis (Task 2A - "Dataset Diversity" criterion)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "for", "in", "on", "this",
    "that", "it", "as", "be", "or", "if", "with", "was", "were", "at", "by", "an",
}


def analyze_diversity(examples: List[CompliancePolicyExample]) -> Dict[str, Any]:
    """
    Report prompt-length distribution and topic/keyword frequency so a reviewer can
    confirm the dataset isn't a homogeneous set of near-duplicate scenarios.
    """
    prompt_lengths = [len(ex.employee_question.split()) + len(ex.policy_excerpt.split()) for ex in examples]
    topic_counts = Counter(ex.topic for ex in examples)

    all_words = []
    for ex in examples:
        words = re.findall(r"[a-zA-Z]+", (ex.employee_question + " " + ex.policy_excerpt).lower())
        all_words.extend(w for w in words if w not in _STOPWORDS and len(w) > 2)
    keyword_freq = Counter(all_words).most_common(25)

    escalation_ratio = (
        sum(1 for ex in examples if ex.requires_escalation) / len(examples) if examples else 0.0
    )

    report = {
        "total_examples": len(examples),
        "topic_distribution": dict(topic_counts),
        "num_distinct_topics": len(topic_counts),
        "prompt_length_words": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
            "mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        },
        "top_keywords": keyword_freq,
        "escalation_ratio": round(escalation_ratio, 3),
    }
    return report


# ---------------------------------------------------------------------------
# JSONL formatting + split
# ---------------------------------------------------------------------------
def to_jsonl_records(examples: List[CompliancePolicyExample]) -> List[Dict[str, Any]]:
    """Each record is {"messages": [...]} - the standard chat-SFT JSONL shape,
    compatible with TRL's SFTTrainer via tokenizer.apply_chat_template()."""
    return [{"messages": ex.to_chat_messages(SYSTEM_PROMPT)} for ex in examples]


def split_dataset(
    records: List[Dict[str, Any]],
    train_frac: float = config.TRAIN_SPLIT,
    val_frac: float = config.VAL_SPLIT,
) -> tuple[list, list, list]:
    import random

    rng = random.Random(config.SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_dataset_pipeline() -> Dict[str, Any]:
    examples = generate_full_dataset()
    if len(examples) < config.MIN_TOTAL_EXAMPLES:
        raise RuntimeError(
            f"Dataset generation produced {len(examples)} valid examples, but "
            f"at least {config.MIN_TOTAL_EXAMPLES} are required. "
            "Existing dataset files were not overwritten. Check the provider/model "
            "configuration and rerun after fixing the API error."
        )

    diversity_report = analyze_diversity(examples)

    with open(config.DIVERSITY_REPORT_PATH, "w") as f:
        json.dump(diversity_report, f, indent=2)

    records = to_jsonl_records(examples)
    train, val, test = split_dataset(records)

    write_jsonl(train, config.TRAIN_PATH)
    write_jsonl(val, config.VAL_PATH)
    write_jsonl(test, config.TEST_PATH)

    logger.info(
        "Dataset written: train=%d val=%d test=%d (total=%d)",
        len(train), len(val), len(test), len(records),
    )
    return {
        "total": len(records),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "diversity_report": diversity_report,
    }


if __name__ == "__main__":
    result = run_dataset_pipeline()
    print(json.dumps({k: v for k, v in result.items() if k != "diversity_report"}, indent=2))
    print(json.dumps(result["diversity_report"], indent=2))
