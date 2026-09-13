import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re
import os
import gc

import src.config as config

# Global cache for models and tokenizers to avoid reloading
_cached_base_model = None
_cached_base_tokenizer = None
_cached_finetuned_model = None
_cached_finetuned_tokenizer = None

# System prompt for base model inference
SYSTEM_PROMPT = """You are a helpful and accurate financial compliance assistant.
Your goal is to answer employee questions based on the provided policy excerpt.
You must always output a JSON object with two keys:
- \"answer\": (string) Your answer to the employee question.
- \"reasoning\": (string) Your reasoning for the answer, citing specific sentences from the policy excerpt.
If the policy excerpt does not contain enough information to answer the question, state this in the 'answer' field and explain why in the 'reasoning' field.
"""

def _load_causal_lm(model_id: str, is_base: bool = True):
    global _cached_base_model, _cached_base_tokenizer, _cached_finetuned_model, _cached_finetuned_tokenizer

    # Define the 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if is_base:
        if _cached_base_model is None or _cached_base_tokenizer is None:
            print(f"Loading base model: {model_id}")
            _cached_base_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _cached_base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
            )
            _cached_base_model.eval()
        return _cached_base_model, _cached_base_tokenizer
    else:
        if _cached_finetuned_model is None or _cached_finetuned_tokenizer is None:
            print(f"Loading fine-tuned model: {model_id}")
            _cached_finetuned_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _cached_finetuned_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
            )
            _cached_finetuned_model.eval()
        return _cached_finetuned_model, _cached_finetuned_tokenizer

def free_cached_models():
    global _cached_base_model, _cached_base_tokenizer, _cached_finetuned_model, _cached_finetuned_tokenizer
    del _cached_base_model
    del _cached_base_tokenizer
    del _cached_finetuned_model
    del _cached_finetuned_tokenizer
    _cached_base_model = None
    _cached_base_tokenizer = None
    _cached_finetuned_model = None
    _cached_finetuned_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Freed cached models and cleared GPU memory.")

def _build_user_prompt(example: dict) -> str:
    """Builds the user prompt from a dataset example."""
    messages = example["messages"]
    # The second message is always the user's question with policy excerpt
    user_message_content = messages[1]["content"]
    return user_message_content

def _extract_json(raw_text: str) -> dict | None:
    """Extracts a JSON object from a string, handling both plain JSON and
    markdown-fenced JSON (```json ... ```)."""
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def _generate(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
    """Generates a response from the model."""
    chat_template = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    # Apply chat template and get input_ids as a PyTorch tensor
    # Ensure attention_mask is also included for models that use it.
    inputs = tokenizer.apply_chat_template(chat_template, return_tensors="pt", add_generation_prompt=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return response

def query_base_model(example: dict) -> dict:
    model, tokenizer = _load_causal_lm(config.STUDENT_BASE_MODEL, is_base=True)
    user_prompt = _build_user_prompt(example)
    raw = _generate(model, tokenizer, SYSTEM_PROMPT, user_prompt)
    parsed = _extract_json(raw)
    if parsed is None:
        return {"answer": raw, "reasoning": "Could not parse JSON response."}
    return parsed

def query_finetuned_model(example: dict) -> dict:
    """Query the merged, fine-tuned model saved locally at config.MERGED_MODEL_LOCAL_DIR."""
    model, tokenizer = _load_causal_lm(config.MERGED_MODEL_LOCAL_DIR, is_base=False)
    user_prompt = _build_user_prompt(example)
    raw = _generate(model, tokenizer, SYSTEM_PROMPT, user_prompt)
    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning("Fine-tuned model returned unparseable output, using fallback: %r", raw[:200])
        return dict(_FALLBACK_RESPONSE)
    return parsed

def query_finetuned_model_with_context(user_question: str, policy_excerpt: str) -> dict:
    """Queries the fine-tuned model with a specific user question and policy excerpt, often for RAG."""
    model, tokenizer = _load_causal_lm(config.HF_HUB_MODEL_ID, is_base=False)
    user_prompt = f"Policy excerpt:\n```\n{policy_excerpt}\n```\nEmployee question: {user_question}"
    raw = _generate(model, tokenizer, SYSTEM_PROMPT, user_prompt)
    parsed = _extract_json(raw)
    if parsed is None:
        return {"answer": raw, "reasoning": "Could not parse JSON response."}
    return parsed

def needs_rag_fallback(model_response: dict) -> bool:
    """Determines if a model response indicates a need for RAG fallback."""
    answer = model_response.get("answer", "").lower()
    return "not contain enough information" in answer or "cannot answer" in answer
