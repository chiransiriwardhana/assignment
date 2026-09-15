# Task 1 - Financial AI: LLM-Powered Equity Research Assistant

Implements the full spec from `Task 1 - Financial AI LLM-Powered Equity Research Assistant`
(100 points): Task 1A data pipeline, Task 1B LLM sentiment/signal reasoning, and the
bonus report renderer.

## Structure

```
task1_financial/
├── src/
│   ├── config.py            # all constants (windows, thresholds, API config) - single source of truth
│   ├── data_pipeline.py      # Task 1A: OHLCV fetch, indicators, news, summary dict
│   ├── schemas.py            # Pydantic models for every LLM-produced structured object
│   ├── prompts.py            # prompt templates, separated from business logic
│   ├── llm_reasoning.py       # Task 1B: sentiment + signal reasoning, with validation
│   └── report_generator.py   # Bonus: Markdown brief + styled HTML with embedded chart
├── notebooks/
│   └── Task1_Equity_Research.ipynb   # Colab-ready notebook, run cells top to bottom
├── outputs/                  # generated reports land here (.md / .html)
├── logs/
├── main.py                   # CLI entry point: `python main.py --ticker MSFT`
├── requirements.txt
├── .env.example
└── README.md
```

## How each requirement is satisfied

**Task 1A - Financial Data Pipeline**
- `fetch_ohlcv()` pulls >= 2 years of daily OHLCV via `yfinance`; the date window is
  computed from `datetime.now()` at call time — no hardcoded date strings.
- `compute_sma`, `compute_rsi` (Wilder smoothing via `ewm(alpha=1/period)`), `compute_macd`
  (12/26/9 EMA-based), and `compute_bollinger_bands` (20-period, 2 std) are all implemented
  from first principles on top of pandas — no TA-Lib dependency anywhere.
- `fetch_news()` pulls headlines from `yfinance`'s news endpoint and tops up from the
  Yahoo Finance RSS feed if fewer than 10 come back, guaranteeing >= 10 headlines when
  the network/ticker allow it.
- `build_summary_dict()` returns current price, 52-week high/low, P/E (when available),
  YTD return, and a rule-based momentum pre-signal derived from the indicators.
- Every network call is wrapped in `try/except` with logging; missing fields degrade to
  `None` rather than raising, and constants (windows, thresholds) live in `config.py`
  instead of being scattered as magic numbers.

**Task 1B - LLM Sentiment and Signal Reasoning**
- `analyze_headline_sentiment()` calls a free-tier LLM (Groq Llama-3.3-70B by default,
  or OpenRouter — switch via `LLM_PROVIDER` env var) and validates the JSON response
  against the `HeadlineSentiment` Pydantic model (`headline`, `sentiment`, `confidence`,
  `brief_reason`). `aggregate_sentiment()` combines all results into one confidence-
  weighted overall score.
- `generate_trading_signal()` feeds the LLM every computed indicator plus the aggregated
  sentiment and requires a 3-5 sentence justification that reasons over the *combination*
  of signals (the prompt explicitly forbids restating indicators in isolation), validated
  against the `TradingSignal` model.
- All JSON parsing failures and Pydantic `ValidationError`s are caught and logged; the
  pipeline continues (skipping the failed headline, or reporting signal generation as
  unavailable) rather than crashing.
- Prompts are defined as constants in `prompts.py`, fully separated from `llm_reasoning.py`,
  with explicit system/user roles.

**Bonus - Report Rendering**
- `report_generator.py` builds a one-page Markdown brief (company snapshot, technical
  outlook, news sentiment summary with top 3 headlines by confidence, LLM recommendation,
  and a mandatory risk disclaimer), then renders it to a styled HTML page with an embedded
  base64 matplotlib chart (price, SMA-50/200, Bollinger Bands over the last 12 months).

## Running it

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` (or set Colab secrets) and add a free Groq or OpenRouter
   API key.
3. Run the notebook `notebooks/Task1_Equity_Research.ipynb` top to bottom, or:
   ```bash
   GROQ_API_KEY=your_key python main.py --ticker AAPL
   ```
4. Outputs land in `outputs/`: `<TICKER>_equity_research_brief.md` and `.html`.

## Known limitations / what I'd improve with more time

- Sentiment classification is one LLM call per headline; batching into a single call
  with a JSON-array response would cut latency and API usage on the free tier.
- The rule-based momentum pre-signal is intentionally simple (a 3-vote heuristic) — it
  exists only to give the LLM a starting point, not as a standalone signal.
- No caching layer yet: re-running the notebook re-fetches OHLCV, news, and re-calls the
  LLM every time. A local cache keyed by ticker + date would reduce free-tier rate-limit
  pressure during iterative development.
