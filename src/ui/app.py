"""
src/ui/app.py
Sentinel-10 Streamlit UI
Run with: streamlit run src/ui/app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]

SAMPLE_QUESTIONS = {
    "10-K": [
        "What are the main cybersecurity risk factors?",
        "How does management describe AI as a competitive risk?",
        "What are the key supply chain risks?",
        "What regulatory risks does the company face?",
        "How does the company describe its cloud strategy?",
        "What are the company's main revenue streams?",
        "How does management describe long-term growth strategy?",
        "What are the key competitive risks mentioned?",
        "How does the company describe its international operations risk?",
        "What environmental and climate risks are disclosed?",
    ],
    "10-Q": [
        "What were the key financial highlights this quarter?",
        "How did revenue change compared to the same quarter last year?",
        "How did operating margins change this quarter?",
        "What new risks were disclosed this quarter?",
        "How does management describe the macroeconomic environment?",
        "Were there any material legal proceedings this quarter?",
        "What guidance did management give for next quarter?",
        "How did segment performance change quarter over quarter?",
        "Were there any significant changes to risk factors this quarter?",
        "What did management say about AI investment this quarter?",
    ],
    None: [
        "What are the main cybersecurity risk factors?",
        "How does management describe AI as a competitive risk?",
        "What were the key financial highlights this quarter?",
        "How did revenue change year over year?",
        "What are the key supply chain risks?",
        "What regulatory risks does the company face?",
        "How does management describe revenue growth outlook?",
        "What new risks were disclosed recently?",
        "How does the company describe its cloud strategy?",
        "What did management say about margins and profitability?",
    ],
}

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

    # ── Form type selector ─────────────────────────────────────────────────────
    form_type_option = st.radio(
        "Filing type",
        ["All filings", "10-K only (Annual)", "10-Q only (Quarterly)"],
        help=(
            "All filings: searches across 10-K and 10-Q\n\n"
            "10-K: annual report — strategy, full risk factors\n\n"
            "10-Q: quarterly report — recent changes, latest financials"
        )
    )

    # Map display label to API value
    FORM_TYPE_MAP = {
        "All filings":             None,
        "10-K only (Annual)":      "10-K",
        "10-Q only (Quarterly)":   "10-Q",
    }
    form_type = FORM_TYPE_MAP[form_type_option]

    # Show a hint about what 10-Q covers
    if form_type == "10-Q":
        st.info(
            "10-Q filings cover what **changed** since the last annual report — "
            "quarterly financials, updated risks, and recent events. "
            "Best for near-term investment decisions."
        )
    elif form_type == "10-K":
        st.info(
            "10-K filings are comprehensive annual reports covering business "
            "overview, full risk factors, and audited financials. "
            "Best for deep company research."
        )

    st.divider()

    mode = st.radio(
        "Mode",
        ["Single company", "Compare companies"],
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
        help="Uses full LangGraph pipeline with tone scoring and verification"
    )

    st.divider()

    st.header("Sample questions")
    current_questions = SAMPLE_QUESTIONS.get(form_type, SAMPLE_QUESTIONS[None])
    selected_sample = st.selectbox(
        "Pick one or write your own",
        [""] + current_questions
    )

# ── Main input ─────────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question about SEC filings",
    value=selected_sample,
    placeholder="What are Apple's main AI risk factors?"
)

# Show what's being searched
if form_type:
    st.caption(f"Searching {ticker} **{form_type}** filings")
else:
    st.caption(f"Searching {ticker} **all filings** (10-K + 10-Q)")

ask_btn = st.button("Ask", type="primary", use_container_width=True)

# ── Query and display ──────────────────────────────────────────────────────────
if ask_btn and question:

    if mode == "Single company":
        endpoint = "/agent/ask" if use_agent else "/ask"

        with st.spinner(f"Researching {ticker} {form_type or 'filings'}..."):
            # Show live status so user knows it's working
            status = st.empty()
            
            try:
                status.caption("🔍 Routing and retrieving...")
                resp = requests.post(
                    f"{API_URL}{endpoint}",
                    json={
                        "question":  question,
                        "ticker":    ticker,
                        "form_type": form_type,   # ← passed to API
                    },
                    timeout=300
                )
                resp.raise_for_status()
                data = resp.json()
                status.empty()
            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out after 5 minutes. "
                    "Try disabling multi-agent mode or using a shorter question."
                )
                st.stop()    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running on port 8000.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # ── Answer ─────────────────────────────────────────────────────────────
        st.subheader("Answer")
        st.markdown(data.get("answer", "No answer returned"))

        # ── Filing type badge ───────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Filing Type", form_type or "All")
        with col2:
            st.metric("Query Type", data.get("query_type", "—").capitalize()
                      if use_agent else "—")
        with col3:
            if use_agent:
                tone  = data.get("tone_label", "—")
                score = data.get("tone_score")
                st.metric("Tone",
                          f"{tone.capitalize()} ({score:.2f})" if score else tone)
            else:
                st.metric("Tone", "—")
        with col4:
            if use_agent:
                verified = data.get("verified", False)
                st.metric("Verified", "✓ Yes" if verified else "⚠ No")
            else:
                st.metric("Verified", "—")
        with col5:
            st.metric("Retries", data.get("retry_count", 0) if use_agent else "—")

        # ── Token usage ─────────────────────────────────────────────────────────
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
                    # Show form type badge per source
                    form_badge = s.get("form_type", "")
                    badge_color = "🟦" if form_badge == "10-K" else "🟩"

                    st.markdown(
                        f"**[{s.get('citation_number', '?')}]** "
                        f"{badge_color} `{form_badge}` · "
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
            # Show live status so user knows it's working
            status = st.empty()
            try:
                status.caption("🔍 Routing and retrieving...")
                resp = requests.post(
                    f"{API_URL}/compare",
                    json={
                        "question":  question,
                        "tickers":   compare_tickers,
                        "form_type": form_type,   # ← passed to API
                    },
                    timeout=300
                )
                resp.raise_for_status()
                data = resp.json()
                status.empty()
            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out after 5 minutes. "
                    "Try disabling multi-agent mode or using a shorter question."
                )
                st.stop() 
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.subheader(f"Comparison — {form_type or 'All filings'}: {question}")
        results = data.get("results", {})

        for t, result in results.items():
            with st.expander(f"**{t}**", expanded=True):
                st.markdown(result.get("answer", "No answer"))

                sources = result.get("sources", [])
                if sources:
                    st.caption(f"{len(sources)} sources retrieved")
                    for s in sources[:2]:
                        form_badge = s.get("form_type", "")
                        badge_color = "🟦" if form_badge == "10-K" else "🟩"
                        st.caption(
                            f"{badge_color} `{form_badge}` · "
                            f"{s.get('section','').replace('_',' ').title()} · "
                            f"Filed {s.get('filing_date','—')}"
                        )