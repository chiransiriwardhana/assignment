"""
dashboard.py
------------
Bonus - Observability Platform Integration.

A simple Streamlit dashboard that reads logs/agent_trace.jsonl and displays
every tool call visually: counts, average duration per tool, error rate, and
a raw searchable table.

Run with:
    streamlit run src/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

TRACE_PATH = Path(__file__).resolve().parent.parent / "logs" / "agent_trace.jsonl"
MESSAGE_TRACE_PATH = Path(__file__).resolve().parent.parent / "logs" / "agent_message_trace.jsonl"

st.set_page_config(page_title="CDAZZDEV Task 3 - Agent Observability", layout="wide")
st.title("Agent Observability Dashboard")
st.caption(f"Reading: {TRACE_PATH}")


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return pd.DataFrame(records)


trace_df = load_jsonl(TRACE_PATH)

if trace_df.empty:
    st.warning("No trace data found yet. Run the notebook (Task 3A/3B) first to generate agent_trace.jsonl.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total tool calls", len(trace_df))
    n_errors = int(trace_df["error"].notna().sum()) if "error" in trace_df.columns else 0
    col2.metric("Errors", n_errors)
    col3.metric("Avg duration (s)", round(trace_df["duration_seconds"].mean(), 3))

    st.subheader("Calls per tool")
    st.bar_chart(trace_df["tool"].value_counts())

    st.subheader("Average duration per tool (seconds)")
    st.bar_chart(trace_df.groupby("tool")["duration_seconds"].mean())

    st.subheader("Raw tool-call trace")
    st.dataframe(trace_df, use_container_width=True)

    if n_errors:
        st.subheader("Errors")
        st.dataframe(trace_df[trace_df["error"].notna()], use_container_width=True)

st.divider()

message_df = load_jsonl(MESSAGE_TRACE_PATH)
if not message_df.empty:
    st.subheader("Agent message trace (Task 3B)")
    labels = message_df["label"].unique().tolist()
    selected_label = st.selectbox("Filter by run", ["(all)"] + labels)
    filtered = message_df if selected_label == "(all)" else message_df[message_df["label"] == selected_label]
    st.dataframe(filtered, use_container_width=True)
