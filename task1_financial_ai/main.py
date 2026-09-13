"""
main.py
-------
End-to-end orchestration for Task 1 - Financial AI LLM-Powered Equity
Research Assistant. Runs the full pipeline: data ingestion (1A) -> LLM
reasoning (1B) -> bonus report rendering.

Usage:
    python main.py --ticker MSFT
    GROQ_API_KEY=sk-... python main.py --ticker MSFT
"""

import argparse
import json
import logging

from src import config
from src.data_pipeline import run_data_pipeline
from src.llm_reasoning import (
    aggregate_sentiment,
    analyze_all_headlines,
    generate_trading_signal,
)
from src.report_generator import save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def run(ticker: str) -> None:
    logger.info("=== Task 1A: Financial Data Pipeline (%s) ===", ticker)
    pipeline_result = run_data_pipeline(ticker)
    summary = pipeline_result["summary"]
    news = pipeline_result["news"]
    df = pipeline_result["ohlcv"]

    logger.info("Summary dictionary:\n%s", json.dumps(summary, indent=2, default=str))
    logger.info("Retrieved %d headlines", len(news))

    logger.info("=== Task 1B: LLM Sentiment and Signal Reasoning ===")
    headlines_text = [n["headline"] for n in news]
    headline_sentiments = analyze_all_headlines(headlines_text)
    aggregated = aggregate_sentiment(headline_sentiments)
    logger.info("Aggregated sentiment: %s", aggregated.model_dump())

    trading_signal = generate_trading_signal(summary, aggregated)
    if trading_signal:
        logger.info("Trading signal: %s | %s", trading_signal.signal.value, trading_signal.justification)
    else:
        logger.warning("Trading signal generation failed - see logs above for the reason.")

    logger.info("=== Bonus: Report Rendering ===")
    paths = save_report(summary, aggregated, headline_sentiments, trading_signal, df)
    logger.info("Report saved: %s", paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 - Equity Research Assistant")
    parser.add_argument("--ticker", type=str, default=config.DEFAULT_TICKER, help="Ticker symbol, e.g. AAPL")
    args = parser.parse_args()
    run(args.ticker)