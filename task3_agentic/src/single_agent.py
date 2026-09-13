"""
single_agent.py
-----------------
Task 3A - Tool-Using Research Agent (built with LangGraph).

A single ReAct-style agent with all five tools. LangGraph's `create_react_agent`
lets the underlying LLM decide autonomously which tool to call and in what
order based on what it has observed so far (no hardcoded sequence) - each
LLM turn either emits a tool call or a final answer, and tool results are fed
back in as observations before the next decision, which is exactly the
"observe -> decide next action" cycle the spec requires.

Also implements:
  * Task 3C short-term memory: reusing the same LangGraph checkpointer
    `thread_id` across calls means the agent's message history (including
    prior tool results) is available on follow-up questions, so it can answer
    without re-invoking a tool.
  * Task 3C persistent memory: `run_query()` checks a JSON cache keyed by
    ticker + date before doing any tool/LLM work at all.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from . import config, memory
from .observability import print_trace
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """You are a senior equity research agent. You have five tools:
get_price_data, get_news, calculate_volatility, llm_sentiment, and web_search.

Decide autonomously which tools to call and in what order, based on what you
observe from each result - do not follow a fixed sequence. A sensible (but not
mandatory) pattern is: retrieve price data and news, compute volatility, score
sentiment on the retrieved headlines, then use web_search only if you need
additional qualitative context the other tools didn't provide.

If any tool returns an {"error": ...} result or clearly incomplete data, do NOT
stop - try an alternative: a different argument (e.g. a shorter volatility
window), a different tool, or a web_search query to compensate for the gap.

If you already retrieved a piece of information earlier in this conversation,
reuse it from your own memory rather than calling the same tool again.

Once you have enough information, respond with a final answer containing
EXACTLY three sections, each with specific, evidence-backed content:

## Financial Health Summary
## Top Three Risks
(one bullet per risk; each bullet must cite a specific number, indicator, or
headline you actually observed as evidence)
## Hedge Strategy Recommendation
"""


def build_agent():
    """Build the LangGraph ReAct agent with a persistent in-memory checkpointer."""
    if not config.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not set - the agent will fail to call the LLM.")

    model = ChatGroq(model=config.AGENT_MODEL, api_key=config.GROQ_API_KEY, temperature=0.1)
    checkpointer = MemorySaver()
    agent = create_react_agent(model, ALL_TOOLS, checkpointer=checkpointer, prompt=SYSTEM_PROMPT)
    return agent


def run_query(agent, ticker: str, thread_id: str = config.DEFAULT_THREAD_ID, use_cache: bool = True) -> Dict[str, Any]:
    """Run the full research query for `ticker`, using the persistent cache if available."""
    if use_cache:
        cached = memory.load_persistent_brief(ticker)
        if cached is not None:
            return cached

    query = (
        f"Analyse the current financial health and market sentiment of {ticker}. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy."
    )
    result = agent.invoke(
        {"messages": [("user", query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print_trace(f"Task 3A single-agent run for {ticker}", result["messages"])

    final_report = result["messages"][-1].content
    output = {"ticker": ticker, "report": final_report}

    if use_cache:
        memory.save_persistent_brief(ticker, output)
    return output


def ask_followup(agent, question: str, thread_id: str = config.DEFAULT_THREAD_ID) -> str:
    """
    Ask a follow-up question on the SAME thread_id used by `run_query`. The
    checkpointer restores the full prior message history (including tool
    results already retrieved), so the agent can answer from that context
    without necessarily calling a tool again - this is the Task 3C
    short-term-memory demonstration.
    """
    result = agent.invoke(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print_trace("Task 3A follow-up (short-term memory)", result["messages"])
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_agent()
    output = run_query(agent, config.DEFAULT_TICKER)
    print(output["report"])
