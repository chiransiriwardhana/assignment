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

## Bugfix

```
# AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: "give me notebook for this code
# (uploaded task1_financial_ai.zip)", Date: 2026-09-13
# Scope: src/report_generator.py used plain `import config` / `from schemas import ...`,
# which raised ModuleNotFoundError when imported as `src.report_generator` (the way
# main.py and notebooks/Task1_Equity_Research.ipynb both import it). Fixed to relative
# imports (`from . import config`, `from .schemas import ...`) to match every other
# module in src/. Also generated notebooks/Task1_Equity_Research.ipynb end-to-end.
```

```
# AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: "diagnose 404 Client Error from
# OpenRouter during a live run of src/llm_reasoning.py", Date: 2026-09-13
# Scope: src/config.py - OPENROUTER_MODEL was hardcoded to
# "meta-llama/llama-3.1-70b-instruct:free", which currently has no active provider
# endpoint on OpenRouter (returns HTTP 404 "no endpoints found", not an auth error).
# Updated to "meta-llama/llama-3.3-70b-instruct:free" and added a comment with a
# one-liner to re-check currently live free models via the OpenRouter /models endpoint,
# since free-tier slugs there rotate frequently. LLM_PROVIDER=groq (the default) with
# GROQ_API_KEY set is a more stable alternative and does not depend on OpenRouter's
# free-tier availability.
```

## Teacher-Model / Data-Generation Prompts

Not applicable to Task 1 (teacher-model data generation is a Task 2 requirement).
The exact system prompts used for LLM sentiment classification and signal reasoning
are defined verbatim in `src/prompts.py` (`SENTIMENT_SYSTEM_PROMPT`, `SIGNAL_SYSTEM_PROMPT`).