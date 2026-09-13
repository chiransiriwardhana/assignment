"""
data_pipeline.py
-----------------
Task 1A - Financial Data Pipeline.

Responsibilities:
  * Fetch >= 2 years of daily OHLCV data via yfinance (no hardcoded date strings).
  * Compute SMA-50, SMA-200, RSI-14, MACD(12,26,9), Bollinger Bands(20, 2) from
    first principles (no TA-Lib).
  * Retrieve >= 10 recent news headlines with a graceful RSS fallback.
  * Produce a clean summary dictionary (current price, 52w high/low, P/E,
    YTD return, momentum signal).
  * Handle missing/null data without raising unhandled exceptions.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from . import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# OHLCV retrieval
# ---------------------------------------------------------------------------
def fetch_ohlcv(ticker: str, years: int = config.LOOKBACK_YEARS) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for `ticker` covering the last `years` years.

    Dates are computed relative to "now" at call time (never hardcoded),
    so re-running this a year from now still pulls the correct window.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(years * 365.25))

    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # network / rate limit / bad ticker
        logger.error("yfinance download failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df.empty:
        logger.warning("No OHLCV data returned for ticker=%s", ticker)
        return df

    # yfinance can return a MultiIndex column frame for a single ticker in
    # some versions - flatten it defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna(how="all")
    return df


# ---------------------------------------------------------------------------
# Indicators - all implemented from first principles (no TA-Lib)
# ---------------------------------------------------------------------------
def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """
    RSI with Wilder smoothing, computed manually:
      1. delta = price change
      2. separate gains / losses
      3. Wilder's smoothed moving average = EWM with alpha = 1/period
      4. RS = avg_gain / avg_loss ; RSI = 100 - 100 / (1 + RS)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 (no losses in window) RSI is defined as 100.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """
    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(signal) of the MACD line
    Histogram = MACD line - Signal line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def compute_bollinger_bands(
    series: pd.Series,
    window: int = config.BOLLINGER_WINDOW,
    num_std: float = config.BOLLINGER_NUM_STD,
) -> pd.DataFrame:
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all required indicators as columns to a copy of df."""
    if df.empty or "Close" not in df.columns:
        logger.warning("Cannot compute indicators - empty/invalid dataframe")
        return df

    out = df.copy()
    close = out["Close"]

    out["SMA_50"] = compute_sma(close, config.SMA_SHORT_WINDOW)
    out["SMA_200"] = compute_sma(close, config.SMA_LONG_WINDOW)
    out["RSI_14"] = compute_rsi(close, config.RSI_PERIOD)

    macd_df = compute_macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    out = out.join(macd_df)

    bb_df = compute_bollinger_bands(close, config.BOLLINGER_WINDOW, config.BOLLINGER_NUM_STD)
    out = out.join(bb_df)

    return out


# ---------------------------------------------------------------------------
# News retrieval - primary source yfinance, RSS fallback
# ---------------------------------------------------------------------------
def _fetch_news_yfinance(ticker: str, n: int) -> List[Dict[str, Any]]:
    try:
        tk = yf.Ticker(ticker)
        raw_news = tk.news or []
    except Exception as exc:
        logger.warning("yfinance news fetch failed for %s: %s", ticker, exc)
        return []

    headlines = []
    for item in raw_news[:n]:
        # yfinance news items may be nested under "content" in newer schema versions.
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else item.get("publisher")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else item.get("link")
        )
        if title:
            headlines.append(
                {"headline": title, "publisher": publisher or "unknown", "link": link or ""}
            )
    return headlines


def _fetch_news_rss_fallback(ticker: str, n: int) -> List[Dict[str, Any]]:
    """Fallback to Yahoo Finance RSS feed if the primary source is thin/unavailable."""
    url = config.YAHOO_RSS_TEMPLATE.format(ticker=ticker)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:
        logger.warning("RSS fallback failed for %s: %s", ticker, exc)
        return []

    headlines = []
    for item in root.findall(".//item")[:n]:
        title_el = item.find("title")
        link_el = item.find("link")
        if title_el is not None and title_el.text:
            headlines.append(
                {"headline": title_el.text, "publisher": "Yahoo RSS", "link": (link_el.text if link_el is not None else "")}
            )
    return headlines


def fetch_news(ticker: str, n: int = config.MIN_NEWS_HEADLINES) -> List[Dict[str, Any]]:
    """Retrieve at least `n` headlines, falling back to RSS if the primary source is short."""
    headlines = _fetch_news_yfinance(ticker, n)
    if len(headlines) < n:
        logger.info(
            "Only %d headlines from yfinance for %s, topping up via RSS fallback",
            len(headlines),
            ticker,
        )
        needed = n - len(headlines)
        headlines.extend(_fetch_news_rss_fallback(ticker, needed))
    return headlines[:n] if headlines else []


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
def get_fundamental_info(ticker: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("Could not fetch fundamentals for %s: %s", ticker, exc)
        info = {}
    return info


# ---------------------------------------------------------------------------
# Momentum signal - a rule-based combination of the computed indicators
# ---------------------------------------------------------------------------
def derive_momentum_signal(df_with_indicators: pd.DataFrame) -> str:
    """
    Combine SMA crossover, RSI zone, and MACD histogram sign into a single
    human-readable momentum label. This is a *rule-based* pre-signal that is
    later handed to the LLM in Task 1B for reasoned Buy/Hold/Sell justification
    - it is not itself the final trading recommendation.
    """
    if df_with_indicators.empty:
        return "unknown"

    last = df_with_indicators.iloc[-1]
    votes = []

    sma_50, sma_200 = last.get("SMA_50"), last.get("SMA_200")
    if pd.notna(sma_50) and pd.notna(sma_200):
        votes.append("bullish" if sma_50 > sma_200 else "bearish")

    rsi = last.get("RSI_14")
    if pd.notna(rsi):
        if rsi >= 70:
            votes.append("overbought")
        elif rsi <= 30:
            votes.append("oversold")
        else:
            votes.append("neutral")

    macd_hist = last.get("macd_hist")
    if pd.notna(macd_hist):
        votes.append("bullish" if macd_hist > 0 else "bearish")

    if not votes:
        return "insufficient_data"

    bullish_votes = votes.count("bullish")
    bearish_votes = votes.count("bearish")

    if "overbought" in votes:
        return "overbought_caution"
    if "oversold" in votes:
        return "oversold_watch"
    if bullish_votes > bearish_votes:
        return "bullish_momentum"
    if bearish_votes > bullish_votes:
        return "bearish_momentum"
    return "neutral"


# ---------------------------------------------------------------------------
# Summary dictionary
# ---------------------------------------------------------------------------
def build_summary_dict(
    ticker: str,
    df_with_indicators: pd.DataFrame,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the clean summary dictionary required by Task 1A, robust to missing data."""
    if df_with_indicators.empty:
        logger.warning("Building summary with empty dataframe for %s", ticker)
        return {
            "ticker": ticker,
            "current_price": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
            "pe_ratio": info.get("trailingPE"),
            "ytd_return_pct": None,
            "momentum_signal": "unknown",
        }

    close = df_with_indicators["Close"]
    current_price = float(close.iloc[-1])

    one_year_ago = df_with_indicators.index[-1] - timedelta(days=365)
    last_52w = df_with_indicators[df_with_indicators.index >= one_year_ago]
    fifty_two_week_high = float(last_52w["High"].max()) if not last_52w.empty else float(df_with_indicators["High"].max())
    fifty_two_week_low = float(last_52w["Low"].min()) if not last_52w.empty else float(df_with_indicators["Low"].min())

    # YTD return: from first trading day of the current calendar year.
    year_start = datetime(datetime.now().year, 1, 1)
    ytd_slice = df_with_indicators[df_with_indicators.index >= year_start]
    if not ytd_slice.empty:
        start_price = float(ytd_slice["Close"].iloc[0])
        ytd_return_pct = ((current_price - start_price) / start_price) * 100 if start_price else None
    else:
        ytd_return_pct = None

    last_row = df_with_indicators.iloc[-1]

    def _safe_float(val):
        return float(val) if pd.notna(val) else None

    summary = {
        "ticker": ticker,
        "current_price": current_price,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "pe_ratio": info.get("trailingPE"),
        "ytd_return_pct": ytd_return_pct,
        "momentum_signal": derive_momentum_signal(df_with_indicators),
        "sma_50": _safe_float(last_row.get("SMA_50")),
        "sma_200": _safe_float(last_row.get("SMA_200")),
        "rsi_14": _safe_float(last_row.get("RSI_14")),
        "macd": _safe_float(last_row.get("macd")),
        "macd_signal": _safe_float(last_row.get("macd_signal")),
        "bollinger_upper": _safe_float(last_row.get("bb_upper")),
        "bollinger_lower": _safe_float(last_row.get("bb_lower")),
    }
    return summary


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------
def run_data_pipeline(ticker: str = config.DEFAULT_TICKER) -> Dict[str, Any]:
    """Run the full Task 1A pipeline end-to-end and return everything downstream needs."""
    raw_df = fetch_ohlcv(ticker)
    df = add_technical_indicators(raw_df)
    info = get_fundamental_info(ticker)
    news = fetch_news(ticker, config.MIN_NEWS_HEADLINES)
    summary = build_summary_dict(ticker, df, info)

    return {
        "ticker": ticker,
        "ohlcv": df,
        "info": info,
        "news": news,
        "summary": summary,
    }


if __name__ == "__main__":
    result = run_data_pipeline(config.DEFAULT_TICKER)
    print("Summary:", result["summary"])
    print(f"Fetched {len(result['news'])} headlines")