"""
config.py
---------
Central configuration for the equity research pipeline.

All "magic numbers" referenced by the assessment (indicator windows,
minimum data history, minimum headline counts) live here so the rest
of the codebase reads them from one place instead of hardcoding values
inline.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Ticker / data window
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "AAPL"          # can be overridden at call time / CLI arg
LOOKBACK_YEARS = 2                # minimum 2 years of OHLCV required by spec

# ---------------------------------------------------------------------------
# Technical indicator parameters (Task 1A)
# ---------------------------------------------------------------------------
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2

# ---------------------------------------------------------------------------
# News retrieval
# ---------------------------------------------------------------------------
MIN_NEWS_HEADLINES = 10
YAHOO_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# ---------------------------------------------------------------------------
# LLM inference provider (Task 1B) - free tiers only, per assessment cost policy
# ---------------------------------------------------------------------------
# "groq" or "openrouter" - set via environment variable, defaults to groq
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "meta-llama/llama-3.1-70b-instruct:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

LLM_REQUEST_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2
LLM_TEMPERATURE = 0.2  # low temperature for structured/deterministic-ish output

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)