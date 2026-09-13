# REFLECTION - Task 3

_Max 600 words combined across all tasks attempted — merge this with your Task 1
(and Task 2) reflection into a single root-level REFLECTION.md before submitting._

## Architectural decisions

- Chose LangGraph's `create_react_agent` over a hand-rolled loop or CrewAI because
  it gives autonomous tool selection "for free" — the LLM's own tool-calling output
  drives the loop, so there's no hardcoded call sequence to defend, and the
  message list it returns is already the exact observe/decide trace the spec asks for.
- Enforced Agent A / Agent B's restricted tool access structurally (each agent
  object is only ever constructed with its own tool list) rather than through
  prompt instructions alone, since a prompt-only restriction is not a real guarantee.
- Used a separate, tool-less `_structuring_model()` call (`with_structured_output`)
  to convert each agent's free-text reasoning into the Pydantic handoff schema,
  rather than asking the ReAct agent to emit strict JSON directly while also doing
  tool calls — mixing "reason freely + call tools" with "emit only valid JSON" in
  a single agent turn is a common source of malformed output.
- Made every tool catch its own exceptions and return `{"error": ...}` instead of
  raising, so a tool failure becomes something the agent can *observe and route
  around* (per the Task 3A requirement) instead of crashing the whole run.

_(Candidate: replace this with your own words once you've run the notebook — mention
what ticker you tested with and what the agent's actual tool-call order looked like.)_

## What I'd improve with more time

- Add retry/backoff around the Groq calls (both the agents' own reasoning calls and
  the extra `llm_sentiment` / structuring calls), since free-tier rate limits get hit
  quickly when running the notebook repeatedly during development.
- Extend the persistent cache to store the full message trace alongside the final
  report, so a cache hit could still show *why* the cached answer was correct instead
  of just the final text.
- Tighten Agent B's clarification question to a constrained schema instead of free
  text, to make the critique loop's behavior easier to test deterministically.

## Limitations encountered

- LangGraph's `create_react_agent` API has changed across versions; if you hit an
  import error, check your installed `langgraph` version against the `prompt=`
  argument's expected name (`prompt` vs `state_modifier` in older releases).
- `duckduckgo-search` occasionally rate-limits aggressive querying — the `web_search`
  tool's error handling returns `{"error": ...}` on failure rather than crashing, but
  a real deployment would want a second search provider as a fallback.

_(Candidate: add anything specific you hit while actually running this — e.g.
Groq rate limits, ticker-specific data gaps, etc.)_
