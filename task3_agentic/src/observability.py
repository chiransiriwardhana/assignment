"""
observability.py
------------------
Task 3C - Observability.

`traced()` wraps every tool function so that its name, input arguments, output
(truncated to 200 chars), duration, and any error are appended to
`logs/agent_trace.jsonl` — required to be present in the submitted repository.

`print_trace()` renders a full agent message trace (what each agent said,
which tools it called, what was handed off) for Task 3B's "Message Trace
Visible" requirement, and also persists it to `logs/agent_message_trace.jsonl`.
"""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from . import config


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _truncate(value: Any, limit: int = config.TRACE_OUTPUT_TRUNCATE_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def traced(tool_name: str) -> Callable:
    """
    Decorator applied to the *raw* implementation of every tool (before the
    @tool decorator turns it into a LangChain tool object). Ensures the tool
    NEVER raises - on failure it logs the error and returns an
    {"error": ...} dict so the agent can observe the failure and try an
    alternative approach instead of the whole run crashing.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            # Best-effort capture of inputs for logging (args + kwargs).
            try:
                arg_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
                bound_args = dict(zip(arg_names, args))
                bound_args.update(kwargs)
            except Exception:  # pragma: no cover - purely defensive
                bound_args = {"args": args, "kwargs": kwargs}

            try:
                result = fn(*args, **kwargs)
                duration = time.perf_counter() - start
                _append_jsonl(
                    config.AGENT_TRACE_PATH,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "tool": tool_name,
                        "inputs": bound_args,
                        "output": _truncate(result),
                        "error": None,
                        "duration_seconds": round(duration, 4),
                    },
                )
                return result
            except Exception as exc:  # noqa: BLE001 - tool must never raise
                duration = time.perf_counter() - start
                error_msg = f"{type(exc).__name__}: {exc}"
                _append_jsonl(
                    config.AGENT_TRACE_PATH,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "tool": tool_name,
                        "inputs": bound_args,
                        "output": None,
                        "error": error_msg,
                        "duration_seconds": round(duration, 4),
                    },
                )
                # Return a structured error instead of raising, so the calling
                # agent sees it as a tool observation and can decide on a
                # fallback (per Task 3A's error-handling requirement).
                return {"error": error_msg, "tool": tool_name}

        return wrapper

    return decorator


def print_trace(label: str, messages) -> None:
    """
    Print (and persist) a full agent message trace: each message's role,
    content, and any tool calls it made. Used by both single_agent.py and
    multi_agent.py so the required trace is visible in notebook output.
    """
    print(f"\n=== {label} ===")
    for m in messages:
        role = getattr(m, "type", m.__class__.__name__)
        content = getattr(m, "content", "") or ""
        tool_calls = getattr(m, "tool_calls", None)

        print(f"[{role}] {content[:300]}")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "role": role,
            "content": content[:500],
            "tool_calls": None,
        }
        if tool_calls:
            calls_summary = []
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                print(f"    -> tool_call: {name}({args})")
                calls_summary.append({"name": name, "args": args})
            record["tool_calls"] = calls_summary

        _append_jsonl(config.MESSAGE_TRACE_PATH, record)
