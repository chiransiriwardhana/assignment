"""
config.py
---------
Central configuration for the Task 2 fine-tuning pipeline. Every hyperparameter
required to be "explicitly justified" by the spec (Task 2B) is defined here with
a comment explaining the choice - nothing is left at an unexplained default.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
for _d in (DATA_DIR, OUTPUT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "val.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
DIVERSITY_REPORT_PATH = os.path.join(DATA_DIR, "diversity_report.json")

# ---------------------------------------------------------------------------
# Teacher model (data generation) - Task 2A
# Deliberately a DIFFERENT model family from the student below, so we are not
# distilling a model into an identical copy of itself.
# ---------------------------------------------------------------------------
TEACHER_PROVIDER = os.environ.get("TEACHER_PROVIDER", "groq").split("#", 1)[0].strip().lower()
TEACHER_MODEL_GROQ = os.environ.get("TEACHER_MODEL_GROQ", "openai/gpt-oss-120b")
TEACHER_MODEL_OPENROUTER = os.environ.get("TEACHER_MODEL_OPENROUTER", "openai/gpt-4o-mini")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)

TEACHER_TEMPERATURE = 0.9   # higher temperature during generation -> more scenario diversity
LLM_REQUEST_TIMEOUT_SECONDS = 45
LLM_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Dataset generation targets - Task 2A
# ---------------------------------------------------------------------------
POLICY_TOPICS = [
    "anti_money_laundering_kyc",
    "personal_account_dealing_insider_trading",
    "gifts_and_entertainment",
    "conflicts_of_interest",
    "market_abuse_information_barriers",
    "whistleblowing",
    "data_privacy_client_confidentiality",
    "outside_business_activities",
]
EXAMPLES_PER_TOPIC = 15          # 8 topics x 15 = 120 examples, comfortably over the 100 minimum
MIN_TOTAL_EXAMPLES = 100
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1

# ---------------------------------------------------------------------------
# Student model + QLoRA hyperparameters - Task 2B
# Every value below is justified in-line and again in README.md / the notebook.
# ---------------------------------------------------------------------------
STUDENT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# Justification: 7B is the largest class that reliably fits 4-bit QLoRA fine-tuning
# on a free Colab T4/L4 (~15GB VRAM); Mistral-7B-Instruct already understands chat
# formatting well, which shortens the amount of formatting-only learning the LoRA
# adapter has to do, leaving its capacity for the compliance-domain behavior itself.
# It is also a different model family from the Llama-3.3-70B teacher above.

BNB_LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
# Justification: NF4 (NormalFloat4) is empirically the best-performing 4-bit
# quantization scheme for normally-distributed transformer weights (QLoRA paper),
# outperforming plain int4/fp4 at the same memory footprint - required by the spec.
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
# Justification: bfloat16 compute dtype avoids the overflow/underflow issues of
# fp16 on the wider dynamic range of LLM activations, and is natively supported on
# Colab's T4 (via upcast) / L4 (native) GPUs.
BNB_4BIT_USE_DOUBLE_QUANT = True
# Justification: double quantization additionally quantizes the quantization
# constants themselves, saving ~0.4 bits/parameter more with negligible quality
# loss - meaningful on a free-tier GPU's limited VRAM budget.

LORA_R = 16
# Justification: r=16 is the standard QLoRA-paper "sweet spot" for 7B instruction
# models - large enough to capture a new structured-output behavior (our JSON
# schema + escalation logic) without the adapter itself becoming a memory burden.
LORA_ALPHA = 32
# Justification: alpha = 2*r is the common QLoRA convention, which keeps the
# effective LoRA update scale (alpha/r = 2) consistent and well-tested regardless
# of what r is chosen.
LORA_DROPOUT = 0.05
# Justification: light dropout (0.05) regularizes the adapter against overfitting
# our relatively small (~100-120 example) dataset without slowing convergence.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Justification: targeting all attention AND MLP projection matrices (not just
# q_proj/v_proj) gives the adapter enough capacity to shift the model's output
# *format* (strict JSON) as well as its factual grounding behavior - QLoRA authors
# found "all-linear" targeting outperforms attention-only targeting on
# instruction-style tasks at a small additional memory cost that 4-bit loading
# comfortably absorbs.

LEARNING_RATE = 2e-4
# Justification: 2e-4 is the standard LoRA learning rate (roughly 10x a typical
# full fine-tune LR) because only ~0.1-1% of parameters are being updated; lower
# LRs under-fit within our small epoch budget, higher LRs destabilize training on
# a dataset this size.
LR_SCHEDULER_TYPE = "cosine"
# Justification: cosine decay avoids the sharp end-of-training LR cliff of linear
# decay, giving smoother convergence on a short training run.
WARMUP_RATIO = 0.03
# Justification: a short warmup (3% of steps) prevents the first few high-LR steps
# from producing large, destabilizing gradient updates on the randomly-initialized
# LoRA matrices.
NUM_EPOCHS = 3
# Justification: with only ~100-120 examples, 3 epochs gives enough passes over
# the data to learn the JSON schema and escalation logic reliably without
# overfitting the small training set (monitored via validation loss - see Task 2B).
PER_DEVICE_TRAIN_BATCH_SIZE = 2
# Justification: a batch size of 2 sequences at max_seq_length=1024 is close to
# the largest that reliably fits a 4-bit-quantized 7B model's activations on a
# free-tier T4 (15GB) without triggering CUDA OOM.
GRADIENT_ACCUMULATION_STEPS = 8
# Justification: combined with batch size 2, this yields an effective batch size
# of 16 - large enough for stable gradient estimates on this dataset size while
# keeping the per-step memory footprint small enough for a T4.
MAX_SEQ_LENGTH = 1024
# Justification: our policy excerpts + question + JSON answer comfortably fit
# within 1024 tokens (measured during dataset diversity analysis - see
# dataset_generation.py); going higher only wastes VRAM on padding.

SEED = 42

# ---------------------------------------------------------------------------
# Model saving / hub push
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HUB_MODEL_ID = os.environ.get("HF_HUB_MODEL_ID", "Chiransiriwardena/mistral-7b-compliance-assistant")
MERGED_MODEL_LOCAL_DIR = os.path.join(OUTPUT_DIR, "merged_model")
ADAPTER_LOCAL_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")

# ---------------------------------------------------------------------------
# Evaluation - Task 2C
# ---------------------------------------------------------------------------
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "groq")
JUDGE_MODEL_GROQ = "openai/gpt-oss-120b"
MIN_MANUAL_REVIEW_SAMPLES = 10
