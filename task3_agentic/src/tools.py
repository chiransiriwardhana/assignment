"""
tools.py
--------
Task 3A - the five tools every agent in this system is built from:
  get_price_data, get_news, calculate_volatility, llm_sentiment, web_search.

Every tool is wrapped with `@traced(...)` (observability.py) so every call is
logged to agent_trace.jsonl, and every tool catches its own exceptions and
returns a structured {"error": ...} dict rather than raising - so a LangGraph
agent always has something to *observe* and can decide on a fallback approach
per the Task 3A error-handling requirement.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from langchain_core.tools import tool

from . import config
from .observability import traced

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# 1. get_price_data
# ---------------------------------------------------------------------------
def _compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))


@traced("get_price_data")
def _get_price_data_impl(ticker: str, period: str = "1y") -> Dict[str, Any]:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if df.empty:
        return {"error": f"No OHLCV data returned for ticker={ticker!r} (check the symbol is valid)."}

    close = df["Close"]
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi_14 = _compute_rsi(close, 14) if len(close) >= 15 else None

    return {
        "ticker": ticker,
        "period": period,
        "current_price": float(close.iloc[-1]),
        "fifty_two_week_high": float(df["High"].max()),
        "fifty_two_week_low": float(df["Low"].min()),
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi_14": rsi_14,
        "num_trading_days": len(df),
    }


@tool
def get_price_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Fetch OHLCV price data for `ticker` over `period` (e.g. "6mo", "1y", "2y")
    and return current price, 52-week high/low, SMA-50, SMA-200, and RSI-14.
    Returns {"error": ...} instead of raising if the ticker is invalid or data
    is unavailable - try a different ticker spelling or fall back to web_search
    if this happens."""
    return _get_price_data_impl(ticker, period)


# ---------------------------------------------------------------------------
# 2. get_news
# ---------------------------------------------------------------------------
def _fetch_news_yfinance(ticker: str, n: int) -> List[Dict[str, str]]:
    tk = yf.Ticker(ticker)
    raw_news = tk.news or []
    headlines = []
    for item in raw_news[:n]:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if title:
            headlines.append({"headline": title, "publisher": item.get("publisher", "unknown")})
    return headlines


def _fetch_news_rss_fallback(ticker: str, n: int) -> List[Dict[str, str]]:
    url = config.YAHOO_RSS_TEMPLATE.format(ticker=ticker)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    headlines = []
    for item in root.findall(".//item")[:n]:
        title_el = item.find("title")
        if title_el is not None and title_el.text:
            headlines.append({"headline": title_el.text, "publisher": "Yahoo RSS"})
    return headlines


@traced("get_news")
def _get_news_impl(ticker: str, n: int = config.NEWS_DEFAULT_N) -> Dict[str, Any]:
    try:
        headlines = _fetch_news_yfinance(ticker, n)
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        headlines = []

    if len(headlines) < n:
        try:
            headlines.extend(_fetch_news_rss_fallback(ticker, n - len(headlines)))
        except Exception as exc:
            logger.warning("RSS fallback failed for %s: %s", ticker, exc)

    if not headlines:
        return {"error": f"No news headlines could be retrieved for ticker={ticker!r}."}
    return {"ticker": ticker, "headlines": headlines[:n]}


@tool
def get_news(ticker: str, n: int = config.NEWS_DEFAULT_N) -> Dict[str, Any]:
    """Retrieve up to `n` recent news headlines for `ticker`. Returns a dict
    with a "headlines" list of {"headline", "publisher"}. Returns {"error": ...}
    if no headlines are available - try web_search for analyst commentary instead."""
    return _get_news_impl(ticker, n)


# ---------------------------------------------------------------------------
# 3. calculate_volatility
# ---------------------------------------------------------------------------
@traced("calculate_volatility")
def _calculate_volatility_impl(ticker: str, window: int = config.VOLATILITY_DEFAULT_WINDOW) -> Dict[str, Any]:
    # Pull a little extra history so the rolling return-window has enough data.
    df = yf.download(ticker, period=f"{max(window * 3, 90)}d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if df.empty or len(df) < window + 1:
        return {"error": f"Not enough price history to compute {window}-day volatility for {ticker!r}."}

    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent_returns = returns.tail(window)
    daily_std = float(recent_returns.std())
    annualized_pct = daily_std * np.sqrt(config.TRADING_DAYS_PER_YEAR) * 100

    return {
        "ticker": ticker,
        "window_days": window,
        "daily_std": daily_std,
        "annualized_volatility_pct": round(annualized_pct, 2),
    }


@tool
def calculate_volatility(ticker: str, window: int = config.VOLATILITY_DEFAULT_WINDOW) -> Dict[str, Any]:
    """Compute annualized historical volatility (%) for `ticker` over the last
    `window` trading days, from daily log returns. Returns {"error": ...} if
    there isn't enough price history - try a shorter window or a different ticker."""
    return _calculate_volatility_impl(ticker, window)


# ---------------------------------------------------------------------------
# 4. llm_sentiment
# ---------------------------------------------------------------------------
_SENTIMENT_SYSTEM_PROMPT = """You are a financial news sentiment scorer. Given a list of \
headlines, respond with ONLY a valid JSON object with exactly these keys: "overall_score" \
(a number from -1.0 very negative to 1.0 very positive), "overall_label" ("positive", \
"negative", or "neutral"), and "rationale" (one sentence explaining the aggregate score)."""


@traced("llm_sentiment")
def _llm_sentiment_impl(headlines: List[str]) -> Dict[str, Any]:
    if not headlines:
        return {"error": "No headlines provided to llm_sentiment."}
    if not config.GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set - cannot call the sentiment LLM."}

    user_prompt = "Headlines:\n" + "\n".join(f"- {h}" for h in headlines)
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": config.AGENT_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        resp = requests.post(
            config.GROQ_BASE_URL, headers=headers, json=payload, timeout=config.LLM_REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return {"error": f"llm_sentiment API call failed: {exc}"}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else None

    if parsed is None:
        return {"error": f"llm_sentiment returned unparseable output: {raw[:200]}"}

    return {"headline_count": len(headlines), **parsed}


@tool
def llm_sentiment(headlines: List[str]) -> Dict[str, Any]:
    """Score the aggregate sentiment of a list of news headlines using an LLM.
    Returns {"overall_score", "overall_label", "rationale"}, or {"error": ...}
    if scoring fails - fall back to a neutral assumption if this happens."""
    return _llm_sentiment_impl(headlines)


# ---------------------------------------------------------------------------
# 5. web_search
# ---------------------------------------------------------------------------
@traced("web_search")
def _web_search_impl(query: str, max_results: int = config.WEB_SEARCH_MAX_RESULTS) -> Dict[str, Any]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {"error": "duckduckgo-search is not installed. Run: pip install duckduckgo-search"}

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return {"error": f"web_search failed: {exc}"}

    if not results:
        return {"error": f"No web search results for query={query!r}."}

    return {
        "query": query,
        "results": [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ],
    }


@tool
def web_search(query: str, max_results: int = config.WEB_SEARCH_MAX_RESULTS) -> Dict[str, Any]:
    """Search the web (via DuckDuckGo) for analyst commentary or context not
    available from the other tools, e.g. "AAPL analyst price target 2026".
    Returns {"error": ...} on failure - try rephrasing the query if this happens."""
    return _web_search_impl(query, max_results)


ALL_TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
