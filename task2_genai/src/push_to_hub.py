"""
push_to_hub.py
---------------
Standalone push of an already-saved merged model + tokenizer to Hugging Face
Hub. Separate from finetune.py so you can push (or re-push, e.g. after editing
the model card) without re-running the full training job.

Usage:
    export HF_TOKEN=hf_xxx...
    export HF_HUB_MODEL_ID=your-username/mistral-7b-compliance-assistant
    python -m src.push_to_hub

    # or, to push as a private repo:
    python -m src.push_to_hub --private
"""

from __future__ import annotations

import argparse
import logging
import re

from . import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: {base_model}
tags:
  - qlora
  - peft
  - compliance
  - financial-services
---

# {model_id}

QLoRA fine-tune of `{base_model}` for the CDAZZDEV Senior MLE Assessment (Task 2 -
Generative AI). Fine-tuned as a **financial compliance policy assistant**: given a
policy excerpt and an employee question, it returns a structured JSON response with
a grounded answer, a citation to the relevant part of the excerpt, an
escalation flag, and a confidence level.

See the `task2_genai/README.md` and `src/problem_statement.md` in the accompanying
GitHub repository for the full use-case definition, training data generation
process, hyperparameters, and evaluation results (ROUGE-L, LLM-as-judge,
hallucination rate).

**This model is a technical-assessment artifact, not a production compliance
tool.** Do not use its output as a substitute for guidance from a licensed
compliance officer.
"""


def push(private: bool = False) -> None:
    if not config.HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN is not set. Get a token with 'write' access from "
            "https://huggingface.co/settings/tokens and set it as an environment "
            "variable (or Colab secret) before running this script."
        )

    if (
        not config.HF_HUB_MODEL_ID
        or "your-username" in config.HF_HUB_MODEL_ID
        or not re.fullmatch(r"[^/\s]+/[^/\s]+", config.HF_HUB_MODEL_ID)
    ):
        raise SystemExit(
            "HF_HUB_MODEL_ID must be your Hugging Face repo ID in the form "
            "username/model-name, for example your-hf-name/mistral-7b-compliance-assistant. "
            "Set it as a Colab secret or environment variable before pushing."
        )

    import os

    from huggingface_hub import HfApi, create_repo

    logger.info("Creating (or reusing) repo %s (private=%s) ...", config.HF_HUB_MODEL_ID, private)
    create_repo(config.HF_HUB_MODEL_ID, token=config.HF_TOKEN, private=private, exist_ok=True)

    logger.info("Writing model card ...")
    api = HfApi(token=config.HF_TOKEN)
    card_content = MODEL_CARD_TEMPLATE.format(base_model=config.STUDENT_BASE_MODEL, model_id=config.HF_HUB_MODEL_ID)
    readme_path = os.path.join(config.MERGED_MODEL_LOCAL_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(card_content)

    logger.info("Uploading already-saved model files from %s ...", config.MERGED_MODEL_LOCAL_DIR)
    api.upload_folder(
        folder_path=config.MERGED_MODEL_LOCAL_DIR,
        repo_id=config.HF_HUB_MODEL_ID,
        repo_type="model",
        token=config.HF_TOKEN,
        commit_message="Upload merged fine-tuned model",
    )

    logger.info("Done: https://huggingface.co/%s", config.HF_HUB_MODEL_ID)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push the merged fine-tuned model to Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="Push as a private repo (grant reviewer read access per Section 3)")
    args = parser.parse_args()
    push(private=args.private)
