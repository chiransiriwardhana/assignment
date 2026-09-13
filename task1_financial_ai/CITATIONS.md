# CITATIONS

This file documents AI assistance used in Task 1, per Section 2.2 of the assessment.

> **Note to candidate:** this entire codebase was scaffolded by Claude in a single
> generation pass based on the assessment PDF. Update the dates/prompts below to
> reflect your actual usage before submitting, and add entries for anything else you
> generate or adapt while integrating this with your own API keys, testing it, and
> writing your REFLECTION.md. Per Section 2, you are expected to be able to explain and
> defend every part of this submission in the follow-up interview — review the code
> thoroughly rather than submitting it unread.

## AI-Assisted Code Generation

```
# AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: "give me required source code and
# files for financial AI task [Task 1 of the CDAZZDEV Senior MLE assessment];
# code should follow all the requirements mentioned in the attached pdf", Date: 2026-09-10
# Scope: full task1_financial/ package - config.py, data_pipeline.py, schemas.py,
# prompts.py, llm_reasoning.py, report_generator.py, main.py, and the Colab notebook.
```

## Adapted Open-Source Code

None. All indicator calculations (SMA, RSI with Wilder smoothing, MACD, Bollinger Bands)
are implemented from first principles on top of pandas/numpy, per the Task 1A requirement
to avoid TA-Lib.

## Teacher-Model / Data-Generation Prompts

Not applicable to Task 1 (teacher-model data generation is a Task 2 requirement).
The exact system prompts used for LLM sentiment classification and signal reasoning
are defined verbatim in `src/prompts.py` (`SENTIMENT_SYSTEM_PROMPT`, `SIGNAL_SYSTEM_PROMPT`).