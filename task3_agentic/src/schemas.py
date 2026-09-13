"""
schemas.py
----------
Pydantic models used for the structured Agent A -> Agent B handoff (Task 3B
explicitly requires a typed schema, not raw string passing).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class DataBrief(BaseModel):
    """Structured quantitative output from Agent A (Data Analyst)."""

    ticker: str
    current_price: float
    annualized_volatility_pct: float
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_label: str
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    notes: str | None = None


class ClarificationRequest(BaseModel):
    """One specific follow-up question Agent B sends back to Agent A."""

    question: str = Field(..., min_length=5)


class ClarificationResponse(BaseModel):
    """Agent A's answer to Agent B's clarification request."""

    answer: str = Field(..., min_length=5)
    supporting_data: Dict[str, Any] = Field(default_factory=dict)


class RiskItem(BaseModel):
    risk: str = Field(..., min_length=3)
    evidence: str = Field(..., min_length=3)


class ResearchReport(BaseModel):
    """Final structured output from Agent B (Research Writer)."""

    ticker: str
    financial_health_summary: str = Field(..., min_length=10)
    top_risks: List[RiskItem]
    hedge_strategy: str = Field(..., min_length=10)

    @field_validator("top_risks")
    @classmethod
    def _exactly_three_risks(cls, v: List[RiskItem]) -> List[RiskItem]:
        if len(v) < 3:
            raise ValueError(f"expected at least 3 risks, got {len(v)}")
        return v[:3] if len(v) > 3 else v
