"""
config.py
---------
Central configuration for the Task 3 multi-agent financial research system.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
for _d in (LOG_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM provider - Groq's OpenAI-compatible tool-calling API, used both as the
# agent's reasoning model (via langchain_groq.ChatGroq) and directly via
# requests inside the llm_sentiment tool.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
AGENT_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-oss-20b")
AGENT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "600"))
STRUCTURING_MAX_TOKENS = int(os.environ.get("STRUCTURING_MAX_TOKENS", "400"))
AGENT_MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "0"))
AGENT_RECURSION_LIMIT = int(os.environ.get("AGENT_RECURSION_LIMIT", "12"))
LLM_REQUEST_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Tool defaults
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "AAPL"
NEWS_DEFAULT_N = 10
VOLATILITY_DEFAULT_WINDOW = 30       # trading days
TRADING_DAYS_PER_YEAR = 252
WEB_SEARCH_MAX_RESULTS = 5
YAHOO_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# ---------------------------------------------------------------------------
# Observability (Task 3C)
# ---------------------------------------------------------------------------
AGENT_TRACE_PATH = os.path.join(LOG_DIR, "agent_trace.jsonl")
MESSAGE_TRACE_PATH = os.path.join(LOG_DIR, "agent_message_trace.jsonl")
TRACE_OUTPUT_TRUNCATE_CHARS = 200

# ---------------------------------------------------------------------------
# Session / default thread id for LangGraph checkpointer (short-term memory)
# ---------------------------------------------------------------------------
DEFAULT_THREAD_ID = "session-1"
