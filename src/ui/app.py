"""
src/ui/app.py
Sentinel-10 Streamlit UI
Run with: streamlit run src/ui/app.py
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]

SAMPLE_QUESTIONS = [
    "What are the main cybersecurity risk factors?",
    "How does management describe AI as a competitive risk?",
    "What are the key supply chain risks?",
    "How does management describe revenue growth outlook?",
    "What regulatory risks does the company face?",
    "How does the company describe its cloud strategy?",
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel-10",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sentinel-10")
st.caption("Multi-agent SEC filings research copilot")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    ticker = st.selectbox("Company", TICKERS)

    mode = st.radio(
        "Mode",
        ["Single company", "Compare companies"],
        help="Single: deep dive one company. Compare: same question across multiple."
    )

    if mode == "Compare companies":
        compare_tickers = st.multiselect(
            "Companies to compare",
            TICKERS,
            default=["AAPL", "MSFT"]
        )

    use_agent = st.toggle(
        "Multi-agent mode",
        value=True,
        help="Uses the full LangGraph pipeline with tone scoring and verification"
    )

    st.divider()
    st.header("Sample questions")
    selected_sample = st.selectbox(
        "Pick one or write your own",
        [""] + SAMPLE_QUESTIONS
    )

# ── Main input ─────────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question about SEC filings",
    value=selected_sample,
    placeholder="What are Apple's main AI risk factors?"
)

ask_btn = st.button("Ask", type="primary", use_container_width=True)

# ── Query and display ──────────────────────────────────────────────────────────
if ask_btn and question:

    if mode == "Single company":
        endpoint = "/agent/ask" if use_agent else "/ask"

        with st.spinner(f"Researching {ticker} filings..."):
            try:
                resp = requests.post(
                    f"{API_URL}{endpoint}",
                    json={"question": question, "ticker": ticker},
                    timeout=120
                )
                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running on port 8000.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # ── Answer ─────────────────────────────────────────────────────────────
        st.subheader("Answer")
        st.markdown(data.get("answer", "No answer returned"))

        # ── Agent metadata (only in agent mode) ────────────────────────────────
        if use_agent:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Query Type",    data.get("query_type", "—").capitalize())
            with col2:
                tone  = data.get("tone_label", "—")
                score = data.get("tone_score")
                st.metric("Tone", f"{tone.capitalize()} ({score:.2f})" if score else tone)
            with col3:
                verified = data.get("verified", False)
                st.metric("Verified", "✓ Yes" if verified else "⚠ No")
            with col4:
                st.metric("Retries", data.get("retry_count", 0))

        # ── Latency + cost ──────────────────────────────────────────────────────
        usage = data.get("usage", {})
        if usage:
            with st.expander("Token usage"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Input tokens",      usage.get("input_tokens", 0))
                c2.metric("Output tokens",     usage.get("output_tokens", 0))
                c3.metric("Cache read tokens", usage.get("cache_read_tokens", 0))

        # ── Sources ─────────────────────────────────────────────────────────────
        sources = data.get("sources", [])
        if sources:
            with st.expander(f"Sources ({len(sources)} chunks retrieved)"):
                for s in sources:
                    st.markdown(
                        f"**[{s.get('citation_number', '?')}]** "
                        f"`{s.get('ticker')}` · "
                        f"{s.get('section', '').replace('_', ' ').title()} · "
                        f"Filed {s.get('filing_date', '—')}"
                    )
                    st.caption(s.get("text", "")[:300] + "...")
                    st.divider()

    else:
        # ── Compare mode ────────────────────────────────────────────────────────
        if len(compare_tickers) < 2:
            st.warning("Select at least 2 companies to compare.")
            st.stop()

        with st.spinner(f"Comparing {', '.join(compare_tickers)}..."):
            try:
                resp = requests.post(
                    f"{API_URL}/compare",
                    json={"question": question, "tickers": compare_tickers},
                    timeout=180
                )
                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.subheader(f"Comparison: {question}")
        results = data.get("results", {})

        for t, result in results.items():
            with st.expander(f"**{t}**", expanded=True):
                st.markdown(result.get("answer", "No answer"))