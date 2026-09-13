# REFLECTION - Task 2

_Max 600 words combined across all tasks attempted — merge this with your Task 1
(and Task 3) reflection into a single root-level REFLECTION.md before submitting._

## Architectural decisions

- Chose a compliance-policy Q&A use case specifically because correctness is
  checkable (grounding to a provided excerpt, numeric threshold reasoning, and an
  escalation flag) rather than subjective — this makes both the teacher-generated
  labels and the evaluation rubric meaningful instead of vibes-based.
- Deliberately used a different model family for teacher (Llama-3.3-70B) vs student
  (Mistral-7B-Instruct) to avoid trivial self-distillation.
- Targeted all linear projection layers with LoRA (not just attention) since the
  task requires learning both a strict output *format* (JSON schema) and a
  *behavior* (grounding + escalation logic) — attention-only targeting is usually
  tuned for style transfer, which isn't the bottleneck here.
- Kept loss logging independent of Weights & Biases (writes to
  `logs/training_loss.json`) so the required per-epoch evidence exists even without
  a wandb account, while still wiring in optional wandb support.

_(Candidate: replace this with your own words once you've run the notebook —
mention your actual final train/val loss curve shape, and whether validation loss
decreased as expected.)_

## What I'd improve with more time

- Add a held-out human-labelled subset to sanity-check the LLM-as-judge's agreement
  with human raters, since a 70B model judging a 7B model's output isn't a perfect
  gold standard on its own.
- Expand dataset generation to include a "trap" category — questions that sound
  like they're covered by the excerpt but subtly aren't — to stress-test the
  escalation logic harder than the current topic-based generation does.
- Automate the manual hallucination review partially by having the judge model
  pre-label examples, with a human only auditing a sample, to make repeated runs
  during development faster while keeping the final reported number human-verified.

## Limitations encountered

- Free-tier LLM APIs occasionally truncate long JSON-array responses when
  generating 15 examples per topic in one call; `dataset_generation.py`'s
  `_extract_json_array` regex fallback handles partial recovery but a failed batch
  still needs a re-run.
- Confidence-based RAG triggering relies on the model's self-reported `confidence`
  field rather than true token-level perplexity, since most free inference APIs
  don't expose logprobs — a reasonable proxy but not as principled as true
  perplexity-based triggering would be.

_(Candidate: add anything specific you hit while actually training on Colab —
OOM errors, dependency conflicts, etc.)_
