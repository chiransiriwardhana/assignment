"""
llm_reasoning.py
-----------------
Task 1B - LLM Sentiment and Signal Reasoning.

Calls a free-tier LLM inference API (Groq or OpenRouter) to:
  1. Classify sentiment for each headline into a validated structured object.
  2. Aggregate per-headline sentiment into a single overall score.
  3. Produce a reasoned Buy/Hold/Sell signal from the combined technical
     indicators (not just an echo of individual values).

All LLM output is validated with Pydantic before use; validation failures
are caught, logged, and handled gracefully rather than crashing the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests
from pydantic import ValidationError

from . import config, prompts
from .schemas import (
    AggregatedSentiment,
    HeadlineSentiment,
    SentimentLabel,
    SignalAction,
    TradingSignal,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SENTIMENT_TO_SCORE = {
    SentimentLabel.POSITIVE: 1.0,
    SentimentLabel.NEUTRAL: 0.0,
    SentimentLabel.NEGATIVE: -1.0,
}


# ---------------------------------------------------------------------------
# Low-level LLM call - provider-agnostic (Groq / OpenRouter both use an
# OpenAI-compatible chat-completions schema, so one function covers both).
# ---------------------------------------------------------------------------
def call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call the configured LLM provider and return the raw text response, or None on failure."""
    if config.LLM_PROVIDER == "groq":
        url, api_key, model = config.GROQ_BASE_URL, config.GROQ_API_KEY, config.GROQ_MODEL
    else:
        url, api_key, model = config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL

    if not api_key:
        logger.error(
            "No API key configured for provider '%s'. Set GROQ_API_KEY or OPENROUTER_API_KEY.",
            config.LLM_PROVIDER,
        )
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": config.LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, config.LLM_MAX_RETRIES + 2):
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=config.LLM_REQUEST_TIMEOUT_SECONDS
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - want to retry on any transient failure
            last_error = exc
            logger.warning("LLM call attempt %d/%d failed: %s", attempt, config.LLM_MAX_RETRIES + 1, exc)

    logger.error("LLM call failed after retries: %s", last_error)
    return None


def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    LLMs sometimes wrap JSON in markdown fences or add stray text despite
    instructions. Try a direct parse first, then fall back to regex-extracting
    the first {...} block before giving up.
    """
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse extracted JSON block: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Per-headline sentiment
# ---------------------------------------------------------------------------
def analyze_headline_sentiment(headline: str) -> Optional[HeadlineSentiment]:
    """Classify a single headline, returning a validated HeadlineSentiment or None on failure."""
    user_prompt = prompts.SENTIMENT_USER_TEMPLATE.format(headline=headline)
    raw = call_llm(prompts.SENTIMENT_SYSTEM_PROMPT, user_prompt)

    parsed = _extract_json(raw) if raw else None
    if parsed is None:
        logger.warning("Could not parse LLM JSON for headline: %r", headline)
        return None

    # Ensure the original headline text is preserved even if the model paraphrased it.
    parsed.setdefault("headline", headline)
    parsed["headline"] = headline

    try:
        return HeadlineSentiment.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("Validation failed for headline sentiment %r: %s", headline, exc)
        return None


def analyze_all_headlines(headlines: List[str]) -> List[HeadlineSentiment]:
    """Run sentiment analysis over every headline, skipping (and logging) any failures."""
    results: List[HeadlineSentiment] = []
    for headline in headlines:
        result = analyze_headline_sentiment(headline)
        if result is not None:
            results.append(result)
        else:
            logger.warning("Skipping headline due to failed sentiment analysis: %r", headline)
    return results


def aggregate_sentiment(results: List[HeadlineSentiment]) -> AggregatedSentiment:
    """Aggregate validated per-headline results into a single overall sentiment score."""
    if not results:
        return AggregatedSentiment(
            overall_score=0.0,
            overall_label=SentimentLabel.NEUTRAL,
            headline_count=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
        )

    weighted_sum = sum(_SENTIMENT_TO_SCORE[r.sentiment] * r.confidence for r in results)
    total_weight = sum(r.confidence for r in results) or 1.0
    overall_score = weighted_sum / total_weight

    if overall_score > 0.15:
        overall_label = SentimentLabel.POSITIVE
    elif overall_score < -0.15:
        overall_label = SentimentLabel.NEGATIVE
    else:
        overall_label = SentimentLabel.NEUTRAL

    return AggregatedSentiment(
        overall_score=round(overall_score, 4),
        overall_label=overall_label,
        headline_count=len(results),
        positive_count=sum(1 for r in results if r.sentiment == SentimentLabel.POSITIVE),
        negative_count=sum(1 for r in results if r.sentiment == SentimentLabel.NEGATIVE),
        neutral_count=sum(1 for r in results if r.sentiment == SentimentLabel.NEUTRAL),
    )


# ---------------------------------------------------------------------------
# Trading signal reasoning
# ---------------------------------------------------------------------------
def generate_trading_signal(
    summary: Dict[str, Any], aggregated_sentiment: AggregatedSentiment
) -> Optional[TradingSignal]:
    """Ask the LLM to reason over the combined indicators + sentiment and return a validated signal."""
    user_prompt = prompts.SIGNAL_USER_TEMPLATE.format(
        ticker=summary.get("ticker"),
        current_price=summary.get("current_price"),
        fifty_two_week_high=summary.get("fifty_two_week_high"),
        fifty_two_week_low=summary.get("fifty_two_week_low"),
        pe_ratio=summary.get("pe_ratio"),
        ytd_return_pct=summary.get("ytd_return_pct"),
        sma_50=summary.get("sma_50"),
        sma_200=summary.get("sma_200"),
        rsi_14=summary.get("rsi_14"),
        macd=summary.get("macd"),
        macd_signal=summary.get("macd_signal"),
        bollinger_upper=summary.get("bollinger_upper"),
        bollinger_lower=summary.get("bollinger_lower"),
        momentum_signal=summary.get("momentum_signal"),
        sentiment_score=aggregated_sentiment.overall_score,
        sentiment_label=aggregated_sentiment.overall_label.value,
    )

    raw = call_llm(prompts.SIGNAL_SYSTEM_PROMPT, user_prompt)
    parsed = _extract_json(raw) if raw else None
    if parsed is None:
        logger.error("Could not parse LLM JSON for trading signal")
        return None

    try:
        return TradingSignal.model_validate(parsed)
    except ValidationError as exc:
        logger.error("Validation failed for trading signal: %s", exc)
        return None


if __name__ == "__main__":
    # Minimal smoke test (requires GROQ_API_KEY / OPENROUTER_API_KEY to be set).
    demo_headline = "Apple beats Q3 earnings expectations on strong iPhone sales"
    result = analyze_headline_sentiment(demo_headline)
    print(result)