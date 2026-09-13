# Task 2 - Generative AI: Domain-Specific Fine-Tuning Pipeline

Use case: **Financial Compliance Policy Assistant** (see `src/problem_statement.md`
for the full structured definition, input/output spec, and correctness criteria).

## Structure

```
task2_genai/
├── src/
│   ├── config.py              # ALL hyperparameters, each with a justification comment
│   ├── problem_statement.md   # Task 2A use-case definition
│   ├── schemas.py             # Pydantic models for dataset examples + eval outputs
│   ├── dataset_generation.py  # Task 2A: teacher-model generation, diversity report, JSONL split
│   ├── finetune.py            # Task 2B: QLoRA fine-tuning, loss logging, merge & save
│   ├── push_to_hub.py         # Standalone Hugging Face Hub push (model card included)
│   ├── inference.py           # Local `transformers` inference: base vs fine-tuned model
│   ├── evaluate.py            # Task 2C: ROUGE-L, LLM-as-judge, hallucination rate
│   └── rag_fallback.py        # Bonus: ChromaDB RAG fallback on low confidence
├── notebooks/
│   └── Task2_Finetuning.ipynb # Colab notebook (needs GPU runtime), run top to bottom
├── data/                      # train.jsonl / val.jsonl / test.jsonl / diversity_report.json
├── outputs/                   # lora_adapter/, merged_model/, generated reports
├── logs/                      # training_loss.json (per-epoch train/val loss)
├── requirements.txt
├── .env.example
└── README.md
```

## How each requirement is satisfied

**Task 2A - Use Case Definition and Dataset Engineering**
- `problem_statement.md` defines a non-trivial, domain-specific use case (compliance
  policy Q&A with grounding + escalation logic) with explicit input/output/correctness
  criteria — deliberately not a generic chatbot task.
- `dataset_generation.py` generates examples via a teacher model (Groq Llama-3.3-70B,
  or GPT-4o-mini via OpenRouter) across **8 distinct policy topics** x 15 examples =
  120 examples (> the 100 minimum), with the exact system prompt used included in the
  module itself (`TEACHER_GENERATION_SYSTEM_PROMPT`) and reproduced in the notebook.
- `analyze_diversity()` reports prompt-length distribution and keyword/topic frequency
  and is run **before** training - the dataset is explicitly generated to vary
  scenario numbers, entities, and escalation outcomes per topic, and the report makes
  that verifiable rather than asserted.
- `to_jsonl_records()` / `write_jsonl()` produce the standard `{"messages": [...]}`
  chat-format JSONL (system/user/assistant turns), applied via
  `tokenizer.apply_chat_template()` at train time so formatting always matches
  whatever base model `config.STUDENT_BASE_MODEL` is set to.
- `split_dataset()` does a seeded 80/10/10 split and `run_dataset_pipeline()` logs and
  returns the exact sizes of each.

**Task 2B - Fine-Tuning Execution**
- `finetune.py` loads `mistralai/Mistral-7B-Instruct-v0.2` in 4-bit NF4
  (`BitsAndBytesConfig`) and applies a LoRA adapter via PEFT — **every** required
  hyperparameter (`r`, `alpha`, target modules, learning rate, scheduler, epochs,
  batch size, gradient accumulation, max sequence length) is defined in `config.py`
  with an inline written justification (see the comments above each constant).
- The student model (Mistral-7B) is deliberately a different model/family from the
  teacher (Llama-3.3-70B) used for data generation.
- `LossLoggingCallback` writes per-epoch train/val loss to
  `logs/training_loss.json` independent of Weights & Biases (wandb support is also
  wired in via `use_wandb=True`), so the required loss evidence exists either way.
- After training, `merge_and_unload()` merges the adapter into the base weights and
  saves the merged model locally. `src/push_to_hub.py` is a standalone script (also
  wired into the notebook) that pushes the already-saved merged model + tokenizer to
  Hugging Face Hub with an auto-generated model card — decoupled from re-running
  training, and the token is read from the environment only (never hardcoded). See
  "Pushing to Hugging Face Hub" below for exact steps.
- An OOM handling path and its documented resolution (reduce batch size / raise
  grad-accum, or fall back to Phi-3-mini) is written directly into
  `build_model_and_tokenizer()`.

**Task 2C - Evaluation and Baseline Comparison**
- `src/inference.py` serves **both** the base model and the fine-tuned merged model
  locally via a `transformers` pipeline on the same GPU used for training — the
  assessment's own tool list (Section 1.2) only mentions Transformers + free Colab
  GPU compute for this stack, so no hosted endpoint or extra infra is used. Models
  are loaded once and cached (`@lru_cache`) so repeated eval calls don't reload from
  disk each time; `free_cached_models()` releases GPU memory between runs.
- `compute_rouge_l()` / `build_rouge_comparison_table()` score the **same held-out
  test set** for both the base model (system prompt, no fine-tuning) and the
  fine-tuned model, output as a row-per-example comparison table plus a summary.
- `judge_response()` implements the required additional metric as an LLM-as-judge
  pipeline scoring faithfulness, citation accuracy, and escalation correctness on a
  1-5 rubric, returned as **Pydantic-validated structured JSON** (`JudgeScore`).
  `compute_bertscore()` is included as a network-dependent alternative if preferred.
- `save_manual_review_template()` writes a CSV for the required >= 10 manually
  reviewed and labelled responses (correct / partially_correct / hallucinated);
  `compute_hallucination_rate()` turns the filled-in CSV into the required
  hallucination-rate percentage.
- `QUALITATIVE_ANALYSIS_TEMPLATE` scaffolds the required two-paragraph analysis —
  fill it in with your own specific before/after examples once you've actually run
  the notebook (this cannot be written before real outputs exist).

**Bonus - RAG Fallback Layer**
- `rag_fallback.py` builds a local ChromaDB collection from the source policy
  documents, triggers retrieval whenever the model's own structured `confidence`
  field is `"low"` (used as a confidence proxy in place of raw logit perplexity,
  which most free inference APIs don't expose), and re-queries the model with the
  retrieved context appended — with a documented before/after logging hook.

## Pushing to Hugging Face Hub

1. Create a free Hugging Face account, then generate a **write**-scoped token at
   https://huggingface.co/settings/tokens.
2. Set it as an environment variable or Colab secret named `HF_TOKEN` (never commit
   it to the repo). Also set `HF_HUB_MODEL_ID` to `your-username/model-name`
   (defaults to the placeholder in `config.py` otherwise).
3. After `finetune.py` has run and saved the merged model to
   `outputs/merged_model/`, push it:
   ```bash
   export HF_TOKEN=hf_xxx...
   export HF_HUB_MODEL_ID=your-username/mistral-7b-compliance-assistant
   python -m src.push_to_hub
   ```
   or from the notebook, just run the "Push the merged model to Hugging Face Hub"
   cell in the Task 2B section.
4. `push_to_hub.push()` creates the repo (idempotent — safe to re-run), pushes the
   model + tokenizer, and writes an auto-generated model card (`README.md` on the
   Hub repo) describing the fine-tune, base model, and intended use.
5. Per Section 3, the repo should be **public**; if you'd rather keep it private,
   run `python -m src.push_to_hub --private` and separately grant the CDAZZDEV
   reviewer account read access as instructed in the assessment email.
6. Verify the link opens in an incognito window before including it in your submission.

## Running it

1. `pip install -r requirements.txt` (the fine-tuning half needs a CUDA GPU — use
   Colab's free T4/L4 runtime for `finetune.py`; dataset generation and evaluation
   are CPU-only and can run anywhere).
2. Copy `.env.example` → `.env` / set Colab secrets with a free Groq or OpenRouter key.
3. Run `notebooks/Task2_Finetuning.ipynb` top to bottom: generates the dataset, fine-tunes,
   merges/saves the model, then evaluates base vs fine-tuned.
4. Fill in `outputs/manual_review.csv` by hand after the notebook produces it, then
   re-run the hallucination-rate cell.

## Known limitations / what I'd improve with more time

- Dataset generation makes one teacher call per topic requesting a batch of 15
  examples; very long batches risk truncation on some providers — retrying with a
  smaller batch size per call would make this more robust at scale.
- LLM-as-judge introduces its own noise (a 70B model judging a 7B model's output is
  not a perfect gold standard); a held-out human-labelled subset to sanity-check
  judge agreement would strengthen Task 2C further.
- The RAG fallback's confidence trigger relies on the model's *self-reported*
  confidence field rather than true perplexity — reasonable for free-tier APIs that
  don't expose logprobs, but worth revisiting if moving to a self-hosted endpoint
  that does.
