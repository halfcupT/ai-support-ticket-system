"""
Minimal Streamlit UI for the AI Support Ticket Intelligence system.

Run with:
    streamlit run ui/streamlit_app.py

This talks to the FastAPI backend over HTTP (API_BASE_URL, default
http://localhost:8000) rather than importing app/ directly, so the UI and
API can be scaled/deployed independently, and so the REST API deliverable
is genuinely exercised (not bypassed) when you use the UI.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Ticket Intelligence", layout="wide")
st.title("🎫 AI Support Ticket Intelligence")
st.caption(f"Backend: {API_BASE_URL}")


def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_BASE_URL}{path}: {exc}")
        return None


def api_post(path: str, json_body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=json_body, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_BASE_URL}{path}: {exc}")
        return None


# --- Sidebar: health -----------------------------------------------------
with st.sidebar:
    st.header("System status")
    health = api_get("/health")
    if health:
        icon = "🟢" if health["status"] == "ok" else "🟠"
        st.write(f"{icon} **{health['status'].upper()}** — {health['rows_loaded']} tickets loaded")
        st.write(f"LLM provider: `{health['llm_provider']}`")
        st.write("LLM reachable: " + ("✅" if health["llm_reachable"] else "❌"))
        if health.get("detail"):
            st.caption(health["detail"])
    st.divider()
    st.caption(
        "Tip: start the API with `uvicorn app.main:app --reload` and set "
        "`LLM_PROVIDER` / `OLLAMA_MODEL` (or `GROQ_API_KEY`) in your .env file."
    )

tab_query, tab_anomalies, tab_dashboard, tab_tickets = st.tabs(
    ["💬 Ask a question", "🚨 Anomalies", "📊 Dashboard", "📋 Browse tickets"]
)

# --- Tab 1: NL query -------------------------------------------------------
with tab_query:
    st.subheader("Ask a natural-language question about the ticket data")
    example_questions = [
        "How many tickets are currently open?",
        "Which agent has the lowest average customer rating?",
        "What is the average customer rating for Technical category tickets?",
        "Show me all Critical tickets not resolved within 12 hours.",
        "How many critical tickets are unresolved?",
    ]
    chosen_example = st.selectbox("Example questions", ["(choose one)"] + example_questions)
    default_q = chosen_example if chosen_example != "(choose one)" else ""
    question = st.text_input("Your question", value=default_q, placeholder="e.g. How many open tickets are there?")
    phrase_with_llm = st.checkbox("Phrase the answer with the LLM (uncheck for a raw templated answer)", value=True)

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            result = api_post("/query", {"question": question, "phrase_with_llm": phrase_with_llm})
        if result:
            st.success(result["answer"])
            if result.get("used_llm_fallback"):
                st.warning("The LLM could not produce a valid query; a safe default was used instead.")
            for w in result.get("warnings", []):
                st.info(w)
            with st.expander("Show generated query spec + raw result"):
                st.json(result["query_spec"])
                st.write(result["result"])

# --- Tab 2: Anomalies ------------------------------------------------------
with tab_anomalies:
    st.subheader("Detected anomalies")
    type_options = {
        "Long resolution time": "long_resolution_time",
        "Slow first response": "slow_first_response",
        "Unresolved SLA breach": "unresolved_sla_breach",
        "Low satisfaction despite resolution": "low_satisfaction_despite_resolution",
    }
    picked_labels = st.multiselect("Filter by type", list(type_options.keys()), default=list(type_options.keys()))
    picked_types = [type_options[label] for label in picked_labels]

    params = [("types", t) for t in picked_types] if picked_types else None
    data = api_get("/anomalies", params=params)
    if data:
        summary = data["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total anomalies", summary["total"])
        c2.metric("High severity", summary["by_severity"].get("high", 0))
        c3.metric("Medium severity", summary["by_severity"].get("medium", 0))

        if data["anomalies"]:
            df_anom = pd.DataFrame(data["anomalies"])
            st.dataframe(df_anom, use_container_width=True, hide_index=True)
        else:
            st.info("No anomalies found for the selected type(s).")

# --- Tab 3: Dashboard -------------------------------------------------------
with tab_dashboard:
    st.subheader("Dataset overview")
    stats = api_get("/stats")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total tickets", stats["total_tickets"])
        c2.metric("Avg response (h)", stats["avg_response_time_hrs"])
        c3.metric("Avg resolution (h)", stats["avg_resolution_time_hrs"])
        c4.metric("Avg rating", stats["avg_customer_rating"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**By status**")
            st.bar_chart(pd.Series(stats["by_status"]))
        with col2:
            st.write("**By priority**")
            st.bar_chart(pd.Series(stats["by_priority"]))
        with col3:
            st.write("**By category**")
            st.bar_chart(pd.Series(stats["by_category"]))

# --- Tab 4: Browse tickets ---------------------------------------------------
with tab_tickets:
    st.subheader("Browse raw tickets")
    c1, c2, c3, c4 = st.columns(4)
    status_f = c1.selectbox("Status", ["(any)", "Open", "Resolved", "Escalated"])
    priority_f = c2.selectbox("Priority", ["(any)", "Low", "Medium", "High", "Critical"])
    category_f = c3.selectbox("Category", ["(any)", "Billing", "Technical", "General"])
    limit_f = c4.number_input("Rows", min_value=5, max_value=500, value=25, step=5)

    params = {"limit": int(limit_f)}
    if status_f != "(any)":
        params["status"] = status_f
    if priority_f != "(any)":
        params["priority"] = priority_f
    if category_f != "(any)":
        params["category"] = category_f

    result = api_get("/tickets", params=params)
    if result:
        st.caption(f"{result['total_matching']} matching ticket(s) — showing {len(result['tickets'])}")
        st.dataframe(pd.DataFrame(result["tickets"]), use_container_width=True, hide_index=True)
