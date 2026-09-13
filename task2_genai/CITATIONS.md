# CITATIONS

This file documents AI assistance used in Task 2, per Section 2.2 of the assessment.

> **Note to candidate:** this entire codebase was scaffolded by Claude in a single
> generation pass based on the assessment PDF. Update the dates/prompts below to
> reflect your actual usage before submitting, and add entries for anything else you
> generate while running the notebook, debugging on Colab, or writing your
> REFLECTION.md. Per Section 2, you must be able to explain and defend every part of
> this submission (including every hyperparameter choice in `config.py`) during the
> follow-up interview.

## AI-Assisted Code Generation

```
# AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: "give me required source code and
# files for Task 2 - Generative AI [of the CDAZZDEV Senior MLE assessment]; code
# should follow all the requirements mentioned in the attached pdf", Date: 2026-09-11
# Scope: full task2_genai/ package - config.py, problem_statement.md, schemas.py,
# dataset_generation.py, finetune.py, evaluate.py, rag_fallback.py, and the Colab
# notebook.
```

## Teacher-Model Data Generation Prompt

The full system prompt used to generate the synthetic compliance Q&A dataset is
defined verbatim in `src/dataset_generation.py` as `TEACHER_GENERATION_SYSTEM_PROMPT`,
and is also printed at the top of the dataset-generation section of the notebook
output per the Section 2.2 requirement to include the full system prompt used for
teacher-model data generation.

## LLM-as-Judge Evaluation Prompt

The full system prompt used for the Task 2C additional metric is defined verbatim in
`src/evaluate.py` as `JUDGE_SYSTEM_PROMPT`.

## Adapted Open-Source Code

None beyond standard library usage patterns for PEFT/TRL/BitsAndBytes as documented
in their official Hugging Face documentation (QLoRA config shape, `SFTTrainer`
arguments, `merge_and_unload()` call). No code was copied from a specific third-party
repository.
