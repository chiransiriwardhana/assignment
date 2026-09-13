"""
schemas.py
----------
Pydantic models used to validate every structured object that comes back
from the LLM. Nothing produced by the LLM is used downstream until it has
passed validation here (Task 1B - "Structured Output Validation", 10 marks).
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SignalAction(str, Enum):
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class HeadlineSentiment(BaseModel):
    """Per-headline structured sentiment result (Task 1B, per-headline JSON)."""

    headline: str = Field(..., min_length=1)
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    brief_reason: str = Field(..., min_length=1)

    @field_validator("headline", "brief_reason")
    @classmethod
    def _strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("field must not be empty after stripping whitespace")
        return v


class AggregatedSentiment(BaseModel):
    """Aggregation of all per-headline sentiment results into one score."""

    overall_score: float = Field(..., ge=-1.0, le=1.0)
    overall_label: SentimentLabel
    headline_count: int = Field(..., ge=0)
    positive_count: int = Field(..., ge=0)
    negative_count: int = Field(..., ge=0)
    neutral_count: int = Field(..., ge=0)


class TradingSignal(BaseModel):
    """LLM-reasoned Buy/Hold/Sell signal (Task 1B, signal reasoning)."""

    signal: SignalAction
    justification: str = Field(..., min_length=1)

    @field_validator("justification")
    @classmethod
    def _check_sentence_count(cls, v: str) -> str:
        v = v.strip()
        # Rough sentence count check - the spec requires 3-5 sentences of
        # justification that reasons over the indicator combination.
        sentence_count = len([s for s in v.replace("!", ".").replace("?", ".").split(".") if s.strip()])
        if sentence_count < 2:
            raise ValueError(
                f"justification appears too short ({sentence_count} sentence(s)); "
                "expected a reasoned 3-5 sentence explanation"
            )
        return v


class EquitySummary(BaseModel):
    """Clean summary dictionary produced by Task 1A, validated for downstream use."""

    ticker: str
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    pe_ratio: float | None = None
    ytd_return_pct: float
    momentum_signal: str
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None


class HeadlineSentimentBatch(BaseModel):
    """Wrapper used purely for validating a full batch at once if needed."""

    results: List[HeadlineSentiment]