"""
schemas.py
----------
Pydantic models validating every structured object produced during dataset
generation (Task 2A) and evaluation (Task 2C).
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CompliancePolicyExample(BaseModel):
    """One synthetic training example produced by the teacher model (Task 2A)."""

    topic: str
    policy_excerpt: str = Field(..., min_length=20)
    employee_question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=5)
    cited_section: str = Field(..., min_length=2)
    requires_escalation: bool
    confidence: ConfidenceLevel

    @field_validator("policy_excerpt", "employee_question", "answer", "cited_section")
    @classmethod
    def _no_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("field must not be blank")
        return v

    def to_chat_messages(self, system_prompt: str) -> List[dict]:
        """Convert to the system/user/assistant chat-turn format used for SFT."""
        import json

        user_content = (
            f"Policy excerpt ({self.topic.replace('_', ' ')}):\n\"{self.policy_excerpt}\"\n\n"
            f"Employee question: \"{self.employee_question}\""
        )
        assistant_content = json.dumps(
            {
                "answer": self.answer,
                "cited_section": self.cited_section,
                "requires_escalation": self.requires_escalation,
                "confidence": self.confidence.value,
            }
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]


class ReviewLabel(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    HALLUCINATED = "hallucinated"


class ManualReviewEntry(BaseModel):
    """One row of the manual hallucination-rate review (Task 2C)."""

    example_id: int
    model_output: str
    label: ReviewLabel
    notes: Optional[str] = None


class JudgeScore(BaseModel):
    """Structured LLM-as-judge output (Task 2C additional metric)."""

    faithfulness: int = Field(..., ge=1, le=5, description="Does the answer only use facts from the excerpt?")
    citation_accuracy: int = Field(..., ge=1, le=5, description="Does cited_section correctly point to the relevant part?")
    escalation_correctness: int = Field(..., ge=1, le=5, description="Is requires_escalation appropriate?")
    overall: int = Field(..., ge=1, le=5)
    reasoning: str = Field(..., min_length=1)


class RougeComparisonRow(BaseModel):
    example_id: int
    base_model_rouge_l: float
    fine_tuned_rouge_l: float
