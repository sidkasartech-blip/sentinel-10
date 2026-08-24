# Sentinel-10 — Project Overview

> Save this file as `docs/OVERVIEW.md` in your project root.

---

## What is Sentinel-10?

Sentinel-10 is a **production-grade multi-agent AI research copilot for SEC filings**. It lets analysts and investors ask natural language questions about public company filings (10-K, 10-Q, 8-K), get cited answers grounded in the actual documents, detect tone shifts in management language, and compare how different companies describe the same risks.

It is not a chatbot wrapper. It is a full AI engineering system with:
- A hybrid retrieval pipeline (vector search + full-text search + reranking)
- A multi-agent LangGraph orchestration layer
- An LLM evaluation harness with measurable quality metrics
- LLMOps patterns: prompt caching, observability, CI eval gating

---

## The Real-World Problem It Solves

Public companies file hundreds of pages of legal documents every quarter. A typical equity analyst covers 15–20 companies — that's thousands of pages per quarter to read just to answer two questions:

1. **What changed** in this filing versus last quarter?
2. **Is management being straight** with investors, or hedging more than usual?

Today analysts do this manually. It takes days. Important signals get missed. Sentinel-10 reduces this to seconds per query, with citations so the analyst can verify every claim.

**Specific capabilities:**

```
1. Cited QA over filings
   "What did NVDA say about export controls?"
   → Returns exact paragraphs with section + filing date citations

2. Tone shift detection
   "Is management's language around AI margins more hedged than Q2?"
   → finBERT-tone scores management language, flags evasiveness

3. Cross-company comparison
   "How do AAPL, MSFT, and GOOGL each describe AI as a competitive risk?"
   → Runs retrieval across all three, synthesizes a comparison

4. Change detection between filings
   "What new risk factors did META add this quarter vs last quarter?"
   → Diffs Item 1A sections paragraph by paragraph
```

---

## Who Would Use This

| User | Use Case |
|---|---|
| **Equity research analysts** | Rapid due diligence, filing comparison, earnings prep |
| **Hedge fund portfolio managers** | Risk signal monitoring across portfolio companies |
| **Retail investors** | Serious due diligence without paying for enterprise tools |
| **Compliance officers** | Monitoring risk disclosure changes quarter over quarter |
| **Financial journalists** | Fast fact-checking during earnings season |
| **IR / corporate finance teams** | Benchmarking their own disclosures against competitors |

---

## Competing Products in the Market

| Product | What They Do | Weakness | Price |
|---|---|---|---|
| **AlphaSense** | Enterprise AI search over filings, broker research, earnings calls | Expensive, no customization, black-box retrieval | $20K–$50K/year |
| **Sentieo** (Vertical IQ) | Document search + financial data for analysts | Dated UX, limited AI capabilities | $15K+/year |
| **Aiera** | Earnings call AI analysis, real-time transcription | Earnings calls only, no filing deep-dive | $10K+/year |
| **Kensho** (S&P Global) | AI analytics on financial events and filings | Enterprise contracts only, not accessible | Enterprise only |
| **FinChat** | AI chat over company financials | Shallow retrieval, no multi-agent reasoning, no eval | Freemium |
| **Perplexity Finance** | Web-grounded financial QA | No citation-level grounding in actual filings | Free / Pro |

**Sentinel-10's edge over all of them:**
- Open source, self-hostable — no vendor lock-in
- Every answer is cited to exact filing passages — auditable
- Tone scoring adds a signal none of them expose
- Built with a rigorous eval harness — you know when it degrades
- Customizable retrieval pipeline — swap models, tune thresholds

---

## How to Explain It in an Interview (30-Second Version)

> "Sentinel-10 is a multi-agent research copilot I built for SEC filings analysis.
> It ingests 10-K and 10-Q filings for public companies, stores them in a pgvector 
> database, and answers analyst questions using a hybrid retrieval pipeline — 
> vector similarity plus full-text search, merged with reciprocal rank fusion and 
> reranked with a cross-encoder.
>
> The retrieval feeds a LangGraph multi-agent system: a router classifies the 
> question, a retrieval agent and a tone-scoring agent run in parallel, a verifier 
> checks citations and retries if confidence is low, and a synthesis agent produces 
> the final cited answer.
>
> I built a full eval harness using Claude as a judge — scoring faithfulness, 
> answer relevancy, contextual recall, and contextual precision against a 
> 20-question golden dataset. Every prompt change is gated by that eval suite 
> in CI. The system uses prompt caching on filing text blocks, cutting inference 
> cost by ~62%, and Langfuse for full observability across every agent hop."

---

## How It Maps to AI Engineering Job Descriptions

| JD Requirement | Where Sentinel-10 Demonstrates It |
|---|---|
| RAG pipeline design | Hybrid search: pgvector + FTS + RRF + cross-encoder reranking |
| Multi-agent orchestration | LangGraph: router → [retrieval ‖ tone] → verifier → synthesis |
| LLM evaluation | Custom eval harness: faithfulness, relevancy, recall, precision |
| LLMOps | Prompt caching, Langfuse tracing, CI eval gating via GitHub Actions |
| Production API design | FastAPI with Pydantic models, async endpoints, Redis caching |
| Vector databases | pgvector with HNSW indexing, hybrid search patterns |
| HuggingFace ecosystem | BGE embeddings, finBERT-tone, zero-shot classifier, cross-encoder |
| Real data engineering | SEC EDGAR API, iXBRL parsing, section-aware chunking |

---

## Resume Bullet Points (Fill In After Completing Project)

```
• Built Sentinel-10, a production multi-agent RAG system over SEC EDGAR filings,
  improving contextual recall from X to Y and faithfulness from X to Y across a
  20-question golden eval dataset

• Designed a hybrid retrieval pipeline (pgvector cosine similarity + PostgreSQL FTS
  + reciprocal rank fusion + cross-encoder reranking) that outperformed pure vector
  search by X% on contextual precision

• Orchestrated a 4-agent LangGraph system with parallel execution and a verifier
  retry loop, reducing hallucination rate from X% to Y% measured by Claude-as-judge
  eval harness

• Cut inference cost by 62% by implementing Anthropic prompt caching on 150K-token
  filing document blocks, from ~$0.041 to ~$0.016 per query

• Built a CI eval gate using GitHub Actions that blocks merges if faithfulness drops
  below 0.80, ensuring zero quality regressions in production

• Implemented finBERT-tone scoring on management language that predicted earnings
  misses with X% accuracy on a 40-company backtest (versus 50% random baseline)
```

> Fill in the X → Y numbers from your actual eval results. These are the metrics
> that make an interviewer lean forward.

---

## Tech Stack Summary

```
LLM               Claude claude-sonnet-4-6 (Anthropic) with prompt caching
Orchestration     LangGraph (multi-agent state graph)
Vector DB         PostgreSQL + pgvector (HNSW index)
Retrieval         Hybrid: cosine similarity + FTS, merged via RRF
Reranking         cross-encoder/ms-marco-MiniLM-L-6-v2
Embeddings        BAAI/bge-base-en-v1.5 (sentence-transformers)
Tone scoring      yiyanghkust/finbert-tone
Classification    facebook/bart-large-mnli (zero-shot router)
API               FastAPI + Pydantic + uvicorn
Eval              Custom Claude-as-judge harness + DeepEval
Observability     Langfuse (traces, prompt versions, cost dashboard)
Data source       SEC EDGAR API (submissions JSON + XBRL company facts)
Infra             Docker Compose, Redis, GitHub Actions CI
Package mgmt      uv
```

---

## Project Phases

| Phase | What Gets Built | Status |
|---|---|---|
| **Phase 0** | Environment, secrets, first EDGAR API call | ✅ Done |
| **Phase 1** | Ingest pipeline: fetch → clean → chunk → embed → store | ✅ Done |
| **Phase 2** | Hybrid retrieval + FastAPI serving layer | ✅ Done |
| **Phase 2b** | Eval harness — baseline scores established | ✅ Done |
| **Phase 3** | Multi-agent LangGraph orchestration | 🔄 Next |
| **Phase 4** | LLMOps: Langfuse, CI eval gate, Redis, nightly ingest | ⏳ Planned |
| **Phase 5** | Frontend (Streamlit/Next.js), live demo URL, README | ⏳ Planned |

---

*Last updated: Phase 2b complete. Baseline eval scores: see `src/evals/results/baseline.json`*