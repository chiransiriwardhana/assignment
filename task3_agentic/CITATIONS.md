# CITATIONS

This file documents AI assistance used in Task 3, per Section 2.2 of the assessment.

> **Note to candidate:** this entire codebase was scaffolded by Claude in a single
> generation pass based on the assessment PDF. Update the dates/prompts below to
> reflect your actual usage before submitting, and add entries for anything else you
> generate while running/debugging the notebook. Per Section 2, you must be able to
> explain and defend every part of this submission — including why LangGraph was
> chosen over LangChain/CrewAI, and how the ReAct loop actually decides tool order —
> during the follow-up interview.

## AI-Assisted Code Generation

```
# AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: "give me required source code and
# files for AI Workflows task [Task 3 - Agentic Workflows of the CDAZZDEV Senior MLE
# assessment]; code should follow all the requirements mentioned in the attached pdf",
# Date: 2026-09-11
# Scope: full task3_agentic/ package - config.py, tools.py, observability.py,
# memory.py, schemas.py, single_agent.py, multi_agent.py, dashboard.py, and the
# Colab notebook.
```

## Adapted Open-Source Code

None beyond standard usage patterns for LangGraph's `create_react_agent` and
`MemorySaver` checkpointer, and LangChain's `@tool` decorator, as documented in
their official documentation. No code was copied from a specific third-party
repository.

## LLM Prompts Used Inside the System

The system prompts for the single agent, Agent A, and Agent B are defined verbatim
in `src/single_agent.py` (`SYSTEM_PROMPT`) and `src/multi_agent.py`
(`AGENT_A_SYSTEM_PROMPT`, `AGENT_B_SYSTEM_PROMPT`). The `llm_sentiment` tool's
internal scoring prompt is defined in `src/tools.py` (`_SENTIMENT_SYSTEM_PROMPT`).
