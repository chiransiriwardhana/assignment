"""
multi_agent.py
---------------
Task 3B - Multi-Agent Coordination.

Agent A (Data Analyst)  - tools: get_price_data, calculate_volatility, llm_sentiment
                          - no access to web_search
Agent B (Research Writer) - tools: web_search, get_news
                          - no direct access to price-data tools

Agent A produces a structured `DataBrief` (Pydantic), handed to Agent B.
Agent B may send ONE `ClarificationRequest` back to Agent A, which responds
with a `ClarificationResponse`; Agent B incorporates it into the final
`ResearchReport`. The whole pipeline runs end-to-end with no manual
intervention - `run_multi_agent_pipeline(ticker)` is the single entry point.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from . import config
from .observability import print_trace
from .schemas import ClarificationRequest, ClarificationResponse, DataBrief, ResearchReport, RiskItem
from .tools import calculate_volatility, get_news, get_price_data, llm_sentiment, web_search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _is_rate_limit_error(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429 or "rate limit" in str(error).lower()

# ---------------------------------------------------------------------------
# Tool access restriction - enforced structurally: each agent is only ever
# bound to its own tool list, so it is physically incapable of calling a tool
# outside its role (not merely instructed not to).
# ---------------------------------------------------------------------------
AGENT_A_TOOLS = [get_price_data, calculate_volatility, llm_sentiment]
AGENT_B_TOOLS = [web_search, get_news]

AGENT_A_SYSTEM_PROMPT = """You are Agent A, a quantitative Data Analyst. You ONLY have
access to get_price_data, calculate_volatility, and llm_sentiment - you have no web
search or general news access. Use your tools to gather quantitative facts about the
ticker you are asked about, then summarize them precisely and numerically. If a tool
returns an error, report the limitation and continue. Do not repeatedly retry a tool."""

AGENT_B_SYSTEM_PROMPT = """You are Agent B, a Research Writer. You ONLY have access to
web_search and get_news - you have NO direct access to price or volatility data; for any
quantitative figures you must rely entirely on what Agent A (the Data Analyst) hands you.
Use each relevant tool at most once to gather qualitative context. If a tool returns an
error or no useful result, report the limitation and continue without retrying."""


def _agent_config() -> Dict[str, Any]:
    return {"recursion_limit": config.AGENT_RECURSION_LIMIT}


def build_agent_a():
    model = ChatGroq(
        model=config.AGENT_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=config.AGENT_MAX_TOKENS,
        max_retries=config.AGENT_MAX_RETRIES,
    )
    return create_agent(model, AGENT_A_TOOLS, system_prompt=AGENT_A_SYSTEM_PROMPT)


def build_agent_b():
    model = ChatGroq(
        model=config.AGENT_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.3,
        max_tokens=config.AGENT_MAX_TOKENS,
        max_retries=config.AGENT_MAX_RETRIES,
    )
    return create_agent(model, AGENT_B_TOOLS, system_prompt=AGENT_B_SYSTEM_PROMPT)


def _structuring_model():
    """A plain (tool-less) model call used purely to coerce free text into a Pydantic schema."""
    return ChatGroq(
        model=config.AGENT_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_tokens=config.STRUCTURING_MAX_TOKENS,
        max_retries=config.AGENT_MAX_RETRIES,
    )


# ---------------------------------------------------------------------------
# Step 1: Agent A produces the initial structured data brief
# ---------------------------------------------------------------------------
def agent_a_produce_brief(agent_a, ticker: str) -> DataBrief:
    query = (
        f"Produce a quantitative data brief for {ticker}: current price and moving "
        "averages / RSI from get_price_data, and 30-day annualized volatility from "
        "calculate_volatility. You do not have news headlines yet, so leave sentiment "
        "as neutral (0.0) with a note that it is pending Agent B's headlines."
    )
    result = agent_a.invoke({"messages": [("user", query)]}, config=_agent_config())
    print_trace(f"Agent A -> initial brief ({ticker})", result["messages"])
    raw_text = result["messages"][-1].content

    structured = _structuring_model().with_structured_output(DataBrief)
    brief: DataBrief = structured.invoke(
        f"Extract a DataBrief for ticker {ticker} from this analyst note "
        f"(use sentiment_score=0.0, sentiment_label='neutral' if not discussed):\n\n{raw_text}"
    )
    return brief


# ---------------------------------------------------------------------------
# Step 2: Agent B reviews the brief and asks Agent A exactly one clarifying question
# ---------------------------------------------------------------------------
def agent_b_request_clarification(agent_b, ticker: str, brief: DataBrief) -> ClarificationRequest:
    prompt = f"""You are drafting a research report for {ticker} and have received this
quantitative brief from Agent A (the Data Analyst):

{brief.model_dump_json(indent=2)}

Before finalizing your report, identify exactly ONE additional piece of quantitative
information that would materially improve it (for example: sentiment scored against
the actual news headlines, a different volatility window, or a confidence check on the
technical indicators). Reply with ONLY that one question, addressed to Agent A."""
    result = agent_b.invoke({"messages": [("user", prompt)]}, config=_agent_config())
    print_trace(f"Agent B -> clarification request ({ticker})", result["messages"])
    question_text = result["messages"][-1].content.strip()
    return ClarificationRequest(question=question_text)


# ---------------------------------------------------------------------------
# Step 3: Agent A answers the clarification (this is where headlines/sentiment
# often get folded in, since Agent B is the one who actually has get_news)
# ---------------------------------------------------------------------------
def agent_a_answer_clarification(
    agent_a, ticker: str, brief: DataBrief, request: ClarificationRequest
) -> ClarificationResponse:
    query = f"""Agent B (the Research Writer) has a follow-up question about {ticker}:

"{request.question}"

Your original brief was:
{brief.model_dump_json(indent=2)}

Use your own tools (get_price_data, calculate_volatility, llm_sentiment) if needed to
answer precisely and numerically - you still have no news or web access, so if the
question requires headline-level sentiment, answer using your existing tools' best
numeric proxy (e.g. technical-indicator-implied momentum) and note explicitly that
headline sentiment is outside your tool access and Agent B should factor it in separately."""
    result = agent_a.invoke({"messages": [("user", query)]}, config=_agent_config())
    print_trace(f"Agent A -> clarification response ({ticker})", result["messages"])
    answer_text = result["messages"][-1].content
    return ClarificationResponse(answer=answer_text, supporting_data={"original_brief": brief.model_dump()})


# ---------------------------------------------------------------------------
# Step 4: Agent B finalizes the report using its own qualitative tools plus
# Agent A's brief and clarification
# ---------------------------------------------------------------------------
def agent_b_finalize_report(
    agent_b, ticker: str, brief: DataBrief, clarification: ClarificationResponse
) -> ResearchReport:
    prompt = f"""Using the quantitative brief and clarification below, plus your own
web_search / get_news tools for qualitative market context, produce the FINAL research
report for {ticker}.

Quantitative brief from Agent A:
{brief.model_dump_json(indent=2)}

Agent A's answer to your clarification question:
{clarification.answer}

Your final message must cover: a financial health summary, the top three risks to the
share price over the next 90 days (each with a specific piece of supporting evidence -
a number, indicator, or headline), and one data-driven hedge strategy recommendation."""
    try:
        result = agent_b.invoke({"messages": [("user", prompt)]}, config=_agent_config())
    except GraphRecursionError:
        logger.warning("Agent B reached the recursion limit; returning a data-only fallback report.")
        return ResearchReport(
            ticker=ticker,
            financial_health_summary=(
                f"{ticker} is trading at ${brief.current_price:.2f}; "
                f"30-day annualized volatility is {brief.annualized_volatility_pct:.2f}%. "
                "The qualitative search stage did not complete."
            ),
            top_risks=[
                RiskItem(risk="Market volatility", evidence=f"Annualized volatility: {brief.annualized_volatility_pct:.2f}%"),
                RiskItem(risk="Momentum reversal", evidence=f"RSI-14: {brief.rsi_14 if brief.rsi_14 is not None else 'unavailable'}"),
                RiskItem(risk="Qualitative information gap", evidence="News and analyst context were unavailable before the recursion limit."),
            ],
            hedge_strategy="Use a defined-risk hedge sized to the reported volatility, such as a protective put or put spread, and review it when new qualitative evidence is available.",
        )
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise
        logger.warning("Agent B hit the provider rate limit; returning a data-only fallback report.")
        return ResearchReport(
            ticker=ticker,
            financial_health_summary=(
                f"{ticker} is trading at ${brief.current_price:.2f}; "
                f"30-day annualized volatility is {brief.annualized_volatility_pct:.2f}%. "
                "The qualitative search stage was unavailable because the provider rate limit was reached."
            ),
            top_risks=[
                RiskItem(risk="Market volatility", evidence=f"Annualized volatility: {brief.annualized_volatility_pct:.2f}%"),
                RiskItem(risk="Momentum reversal", evidence=f"RSI-14: {brief.rsi_14 if brief.rsi_14 is not None else 'unavailable'}"),
                RiskItem(risk="Qualitative information gap", evidence="News and analyst context were unavailable due to provider throttling."),
            ],
            hedge_strategy="Use a defined-risk hedge sized to the reported volatility, such as a protective put or put spread, and review it when new qualitative evidence is available.",
        )
    print_trace(f"Agent B -> final report ({ticker})", result["messages"])
    raw_text = result["messages"][-1].content

    structured = _structuring_model().with_structured_output(ResearchReport)
    report: ResearchReport = structured.invoke(
        f"Extract a ResearchReport for ticker {ticker} (exactly 3 top_risks, each with "
        f"risk + evidence) from this analyst note:\n\n{raw_text}"
    )
    return report


# ---------------------------------------------------------------------------
# Orchestration - fully automated, no manual intervention between query and report
# ---------------------------------------------------------------------------
def run_multi_agent_pipeline(ticker: str) -> Dict[str, Any]:
    agent_a = build_agent_a()
    agent_b = build_agent_b()

    brief = agent_a_produce_brief(agent_a, ticker)
    clarification_request = agent_b_request_clarification(agent_b, ticker, brief)
    clarification_response = agent_a_answer_clarification(agent_a, ticker, brief, clarification_request)
    final_report = agent_b_finalize_report(agent_b, ticker, brief, clarification_response)

    return {
        "ticker": ticker,
        "data_brief": brief,
        "clarification_request": clarification_request,
        "clarification_response": clarification_response,
        "final_report": final_report,
    }


if __name__ == "__main__":
    result = run_multi_agent_pipeline(config.DEFAULT_TICKER)
    print(result["final_report"].model_dump_json(indent=2))
