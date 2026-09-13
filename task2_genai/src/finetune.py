"""
finetune.py
-----------
Task 2B - Fine-Tuning Execution.

QLoRA (4-bit NF4) fine-tuning of `config.STUDENT_BASE_MODEL` on the dataset
produced by dataset_generation.py, using PEFT + TRL + BitsAndBytes.

NOTE: this module requires a GPU runtime (Colab T4/L4) with `torch`,
`transformers`, `peft`, `trl`, and `bitsandbytes` installed - it is not meant
to run in a CPU-only sandbox. Run it from the notebook or via:
    python -m src.finetune
"""

from __future__ import annotations

import json
import gc
import logging
import os
import time
import inspect
from typing import Any, Dict

from . import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_datasets():
    """Load train/val JSONL as a HuggingFace `datasets.Dataset`."""
    from datasets import load_dataset

    data_files = {"train": config.TRAIN_PATH, "validation": config.VAL_PATH}
    for split_name, path in data_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{split_name} dataset not found at {path}. "
                "Run the dataset-generation cells before fine-tuning."
            )
        if os.path.getsize(path) == 0:
            raise ValueError(
                f"{split_name} dataset is empty at {path}. "
                "Run the dataset-generation cells before fine-tuning."
            )

    try:
        dataset = load_dataset("json", data_files=data_files)
    except StopIteration as exc:
        raise ValueError(
            "The training or validation JSONL file contains no readable records. "
            "Regenerate the dataset, then rerun fine-tuning."
        ) from exc

    for split_name in data_files:
        if len(dataset[split_name]) == 0:
            raise ValueError(
                f"{split_name} dataset contains no records. "
                "Regenerate the dataset, then rerun fine-tuning."
            )
    return dataset


def build_model_and_tokenizer():
    """Load the base model in 4-bit NF4 and wrap it with a LoRA adapter (Task 2B)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA fine-tuning requires a CUDA GPU runtime. "
            "Run this notebook in Colab with a T4/L4 GPU; macOS CPU/MPS kernels "
            "cannot load the configured bitsandbytes 4-bit model."
        )

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.STUDENT_BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except torch.cuda.OutOfMemoryError as exc:
        # Documented OOM handling per the spec's "common errors" guidance:
        # if this triggers on a free Colab T4, reduce PER_DEVICE_TRAIN_BATCH_SIZE
        # to 1 and increase GRADIENT_ACCUMULATION_STEPS to 16 to preserve the same
        # effective batch size of 16, or switch STUDENT_BASE_MODEL to a smaller
        # base such as "microsoft/Phi-3-mini-4k-instruct" (3.8B parameters).
        logger.error("CUDA OOM while loading base model: %s", exc)
        raise

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def format_example(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """Apply the base model's chat template to a {"messages": [...]} record."""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}


class LossLoggingCallback:
    """Manual train/val loss-per-epoch logger (Task 2B - "Loss Monitoring").

    Used alongside (or instead of) Weights & Biases; writes a JSON log to
    `config.LOG_DIR/training_loss.json` so a screenshot/log excerpt can be
    included in the submission even without a wandb account.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history = []

    def log(self, epoch: int, train_loss: float, val_loss: float | None):
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "timestamp": time.time()}
        self.history.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Epoch %s | train_loss=%.4f | val_loss=%s", epoch, train_loss, val_loss)


def _build_sft_config(SFTConfig, use_wandb: bool):
    """Build SFTConfig across the TRL versions commonly available in Colab."""
    configured = {
        "output_dir": config.ADAPTER_LOCAL_DIR,
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": config.NUM_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
        "warmup_ratio": config.WARMUP_RATIO,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "max_length": config.MAX_SEQ_LENGTH,
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "bf16": True,
        "seed": config.SEED,
        "report_to": ["wandb"] if use_wandb else [],
        "dataset_text_field": "text",
    }
    supported = set(inspect.signature(SFTConfig).parameters)
    kwargs = {name: value for name, value in configured.items() if name in supported}
    if "eval_strategy" not in supported and "evaluation_strategy" not in supported:
        logger.warning("Installed SFTConfig has no evaluation-strategy argument; evaluation may be unavailable.")
    omitted = sorted(set(configured) - set(kwargs))
    if omitted:
        logger.warning("Ignoring unsupported SFTConfig arguments for this TRL version: %s", omitted)
    return SFTConfig(**kwargs)


def run_finetuning(use_wandb: bool = False) -> Dict[str, Any]:
    """Run the full QLoRA SFT job and save the adapter + merged model."""
    from trl import SFTTrainer, SFTConfig

    if use_wandb:
        import wandb

        wandb.init(project="cdazzdev-compliance-assistant", config=vars(config))

    dataset = load_datasets()
    model, tokenizer = build_model_and_tokenizer()

    formatted_train = dataset["train"].map(lambda ex: format_example(ex, tokenizer))
    formatted_val = dataset["validation"].map(lambda ex: format_example(ex, tokenizer))

    training_args = _build_sft_config(SFTConfig, use_wandb)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": formatted_train,
        "eval_dataset": formatted_val,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    trainer_parameters = set(inspect.signature(SFTTrainer).parameters)
    trainer_kwargs = {
        name: value for name, value in trainer_kwargs.items() if name in trainer_parameters
    }
    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting QLoRA fine-tuning: %d epochs, effective batch size=%d",
                config.NUM_EPOCHS, config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS)

    train_result = trainer.train()

    # Manual per-epoch loss log, independent of wandb, for submission evidence.
    loss_logger = LossLoggingCallback(os.path.join(config.LOG_DIR, "training_loss.json"))
    for log_entry in trainer.state.log_history:
        if "loss" in log_entry and "epoch" in log_entry:
            eval_entry = next(
                (e for e in trainer.state.log_history if "eval_loss" in e and e.get("epoch") == log_entry["epoch"]),
                None,
            )
            loss_logger.log(
                epoch=log_entry["epoch"],
                train_loss=log_entry["loss"],
                val_loss=eval_entry["eval_loss"] if eval_entry else None,
            )

    logger.info("Training complete: %s", train_result)

    # Save adapter
    trainer.save_model(config.ADAPTER_LOCAL_DIR)
    tokenizer.save_pretrained(config.ADAPTER_LOCAL_DIR)

    # Merge LoRA into the base model and save the merged model (Task 2B requirement)
    merged_model = trainer.model.merge_and_unload()
    os.makedirs(config.MERGED_MODEL_LOCAL_DIR, exist_ok=True)
    merged_model.save_pretrained(config.MERGED_MODEL_LOCAL_DIR)
    tokenizer.save_pretrained(config.MERGED_MODEL_LOCAL_DIR)
    logger.info("Merged model saved to %s", config.MERGED_MODEL_LOCAL_DIR)

    if config.HF_TOKEN:
        push_to_hub(merged_model, tokenizer)

    # Release the training and merged models before a later inference cell loads
    # another 7B model into the same GPU.
    del trainer, model, merged_model, tokenizer, formatted_train, formatted_val, dataset
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    return {"adapter_dir": config.ADAPTER_LOCAL_DIR, "merged_dir": config.MERGED_MODEL_LOCAL_DIR}


def push_to_hub(model, tokenizer) -> None:
    """Push the merged model to Hugging Face Hub using a token from the environment only."""
    if not config.HF_TOKEN:
        logger.warning("HF_TOKEN not set - skipping hub push. Model is still saved locally.")
        return
    model.push_to_hub(config.HF_HUB_MODEL_ID, token=config.HF_TOKEN)
    tokenizer.push_to_hub(config.HF_HUB_MODEL_ID, token=config.HF_TOKEN)
    logger.info("Pushed merged model to https://huggingface.co/%s", config.HF_HUB_MODEL_ID)


if __name__ == "__main__":
    run_finetuning(use_wandb=False)
