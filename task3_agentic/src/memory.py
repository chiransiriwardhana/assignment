"""
memory.py
---------
Task 3C - Persistent Memory.

Saves the final research brief to a JSON file keyed by ticker + date. On a
subsequent run for the same ticker on the same day, the cached file is
detected and loaded instead of re-running the full tool/LLM pipeline.

(Short-term, within-session memory is handled separately in single_agent.py /
multi_agent.py by reusing the same LangGraph checkpointer `thread_id` across
calls, so the agent's own message history already contains prior tool
results and it can answer follow-ups without re-invoking a tool.)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any, Dict, Optional

from . import config

logger = logging.getLogger(__name__)


def _cache_path(ticker: str, on_date: Optional[str] = None) -> str:
    day = on_date or date.today().isoformat()
    return os.path.join(config.CACHE_DIR, f"{ticker.upper()}_{day}.json")


def load_persistent_brief(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the cached brief for `ticker` dated today, or None if not cached."""
    path = _cache_path(ticker)
    if os.path.exists(path):
        logger.info("Persistent cache HIT for %s at %s - skipping tool/LLM calls.", ticker, path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.info("Persistent cache MISS for %s - running full pipeline.", ticker)
    return None


def save_persistent_brief(ticker: str, data: Dict[str, Any]) -> str:
    """Persist the final brief/report for `ticker`, keyed by ticker + today's date."""
    path = _cache_path(ticker)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved persistent brief for %s to %s", ticker, path)
    return path


def clear_cache(ticker: Optional[str] = None) -> None:
    """Utility for demos/tests: clear today's cache for one ticker, or everything."""
    if ticker:
        path = _cache_path(ticker)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Cleared cache for %s", ticker)
    else:
        for fname in os.listdir(config.CACHE_DIR):
            os.remove(os.path.join(config.CACHE_DIR, fname))
        logger.info("Cleared all cached briefs")
