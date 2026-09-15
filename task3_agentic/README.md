# Task 3 - Agentic Workflows: Multi-Agent Financial Research System

Built with **LangGraph** (on top of LangChain), using Groq's Llama-3.3-70B as the
tool-calling LLM for every agent.

## Structure

```
task3_agentic/
├── src/
│   ├── config.py         # constants: model, defaults, paths
│   ├── tools.py           # the 5 required tools, each traced + error-handled
│   ├── observability.py   # agent_trace.jsonl logger + message-trace printer
│   ├── memory.py           # persistent (JSON cache) memory helpers
│   ├── schemas.py          # Pydantic models for the Agent A -> Agent B handoff
│   ├── single_agent.py     # Task 3A: single ReAct agent, all 5 tools
│   ├── multi_agent.py      # Task 3B: two-agent pipeline with critique loop
│   └── dashboard.py        # Bonus: Streamlit observability dashboard
├── notebooks/
│   └── Task3_Agentic_Workflows.ipynb
├── logs/
│   ├── agent_trace.jsonl          # generated: every tool call, Task 3C requirement
│   └── agent_message_trace.jsonl  # generated: full agent message trace, Task 3B
├── cache/                  # generated: persistent per-ticker/date research briefs
├── requirements.txt
├── .env.example
└── README.md
```

## How each requirement is satisfied

**Task 3A - Tool-Using Research Agent**
- `tools.py` implements all five required tools (`get_price_data`, `get_news`,
  `calculate_volatility`, `llm_sentiment`, `web_search`), each wrapped with
  `@traced(...)` for observability and each catching its own exceptions to
  return `{"error": ...}` instead of raising, so the agent always has
  something concrete to observe and can try an alternative rather than
  crashing (Task 3A "Error Handling").
- `single_agent.py` builds the agent with **LangGraph's `create_react_agent`**,
  which lets the LLM decide autonomously, turn by turn, which tool to call
  based on what it has already observed - there is no hardcoded call
  sequence anywhere in the code (Task 3A "Autonomous Tool Selection").
- Every LangGraph turn is exactly one observe → decide cycle (tool result fed
  back as an observation before the next LLM decision); `print_trace()`
  prints the full sequence of messages/tool calls to the notebook output,
  satisfying "Observe and Replan Cycle" directly from the framework's own
  execution loop rather than anything staged.
- The system prompt requires the final answer to contain exactly the three
  required sections (Financial Health Summary / Top Three Risks / Hedge
  Strategy Recommendation), each evidence-backed.

**Task 3B - Multi-Agent Coordination**
- `multi_agent.py` defines Agent A (Data Analyst) and Agent B (Research
  Writer) as two **separate** LangGraph agents, each bound to its own tool
  list (`AGENT_A_TOOLS`, `AGENT_B_TOOLS`) — tool access is restricted
  structurally (the agent object is physically incapable of calling a tool
  it wasn't given), not just by instruction.
- `agent_a_produce_brief()` returns a validated `DataBrief` Pydantic object -
  the handoff to Agent B is a typed schema, never a raw string.
- `agent_b_request_clarification()` → `agent_a_answer_clarification()` →
  `agent_b_finalize_report()` implements the required critique loop exactly
  once: Agent B asks one specific question, Agent A answers it, and Agent B's
  final report generation explicitly incorporates that answer.
- `print_trace()` is called after every single agent turn in the pipeline, so
  the complete message trace (what each agent said, which tools it called,
  what was hand off) is visible in notebook output and also persisted to
  `logs/agent_message_trace.jsonl`.
- `run_multi_agent_pipeline(ticker)` is the single entry point - calling it
  runs the entire A → B → A → B flow end-to-end with no manual steps in between.

**Task 3C - Memory and Observability**
- **Short-term memory**: `single_agent.py` uses a LangGraph `MemorySaver`
  checkpointer keyed by `thread_id`. `run_query()` and `ask_followup()` reuse
  the same `thread_id`, so the agent's full prior message history (including
  tool results already retrieved) is available on a follow-up question - the
  notebook demonstrates asking a follow-up that's answered without a new tool
  call appearing in `agent_trace.jsonl`.
- **Persistent memory**: `memory.py`'s `load_persistent_brief()` /
  `save_persistent_brief()` cache the final brief to
  `cache/{TICKER}_{date}.json`. `run_query()` checks this cache *before*
  invoking the agent at all - a second run for the same ticker on the same
  day loads the cached file and performs zero new tool/LLM calls.
- **Observability**: `observability.py`'s `traced()` decorator wraps every
  tool call and appends a line to `logs/agent_trace.jsonl` with the tool
  name, input arguments, output (truncated to 200 characters), and
  wall-clock duration - present in the repo per the mandatory deliverable list.

**Bonus - Observability Platform Integration**
- `dashboard.py` is a Streamlit app that reads `agent_trace.jsonl` and
  `agent_message_trace.jsonl` and renders call counts, average duration per
  tool, an error table, and the raw searchable trace. Run with:
  `streamlit run src/dashboard.py`.

## Running it

1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` (or set `GROQ_API_KEY` as a Colab secret /
   shell env var). Free key from console.groq.com.
3. Run `notebooks/Task3_Agentic_Workflows.ipynb` top to bottom, or:
   ```bash
   python -m src.single_agent      # Task 3A demo
   python -m src.multi_agent       # Task 3B demo
   streamlit run src/dashboard.py  # bonus dashboard (after a run has produced logs)
   ```
4. `logs/agent_trace.jsonl` and `logs/agent_message_trace.jsonl` are created
   on first run - commit them (with real content) as part of the submission.
