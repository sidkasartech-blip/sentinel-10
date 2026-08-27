# Sentinel-10

> Multi-agent SEC filings research copilot — production-grade AI system for financial document analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-green)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange)
![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is Sentinel-10?

Sentinel-10 is a production-grade multi-agent AI system that ingests SEC 10-K filings for major public companies and answers analyst-grade research questions with **cited, verified answers** grounded in the actual filing text.

It is not a chatbot wrapper around an LLM. It is a full AI engineering system with:

- **Hybrid retrieval pipeline** — vector search + full-text search + reciprocal rank fusion + cross-encoder reranking
- **Multi-agent LangGraph orchestration** — router, retrieval, tone scoring, verifier, and synthesis agents
- **LLM evaluation harness** — Claude-as-judge scoring faithfulness, relevancy, recall, and precision
- **LLMOps layer** — Langfuse tracing, prompt caching, cost tracking per query
- **Production API** — FastAPI with Pydantic validation, async endpoints, structured responses
- **Interactive UI** — Streamlit interface with company selector, comparison mode, and citation panel

---

## The Problem It Solves

Public companies file hundreds of pages of legal documents every quarter. A typical equity analyst covers 15–20 companies — thousands of pages per quarter — just to answer:

1. **What changed** in this filing versus last quarter?
2. **Is management being straightforward** with investors, or hedging more than usual?

Sentinel-10 reduces this from hours to seconds, with every answer cited to exact filing passages so the analyst can verify any claim.

---

## Current Coverage

| Company | Ticker | Filing Type | Source |
|---|---|---|---|
| Apple Inc. | AAPL | 10-K | SEC EDGAR |
| Microsoft Corp. | MSFT | 10-K | SEC EDGAR |
| Alphabet Inc. | GOOGL | 10-K | SEC EDGAR |
| Meta Platforms | META | 10-K | SEC EDGAR |
| NVIDIA Corp. | NVDA | 10-K | SEC EDGAR |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
│              Streamlit UI  ·  FastAPI REST  ·  /docs            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    MULTI-AGENT CORE (LangGraph)                 │
│                                                                 │
│   ┌──────────┐    ┌───────────┐  ┌──────────┐                  │
│   │  Router  │───▶│ Retrieval │  │  Tone    │  (parallel)      │
│   │  Agent   │    │  Agent    │  │  Agent   │                  │
│   └──────────┘    └─────┬─────┘  └────┬─────┘                  │
│                         └──────┬───────┘                        │
│                    ┌──────────▼──────────┐                     │
│                    │  Synthesis Agent    │                     │
│                    └──────────┬──────────┘                     │
│                    ┌──────────▼──────────┐                     │
│                    │  Verifier Agent     │◀── retry loop       │
│                    └─────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    HYBRID RETRIEVAL PIPELINE                    │
│                                                                 │
│   BGE Embeddings ──▶ pgvector cosine search ──┐                │
│                                               ├──▶ RRF Merge   │
│   PostgreSQL FTS ──▶ ts_rank search ──────────┘       │        │
│                                               Cross-encoder    │
│                                               Reranker ────▶   │
│                                               Top-K Chunks     │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    DATA LAYER                                   │
│                                                                 │
│   PostgreSQL + pgvector  ·  HNSW Index  ·  GIN FTS Index       │
│   filing_chunks table: id, ticker, section, text, embedding     │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    INGEST PIPELINE                              │
│                                                                 │
│   SEC EDGAR API ──▶ iXBRL Parser ──▶ Section Detector          │
│   ──▶ Chunker ──▶ BGE Embedder ──▶ pgvector Store              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### LLM and AI

| Component | Model / Library | Purpose |
|---|---|---|
| **Primary LLM** | `claude-sonnet-4-6` (Anthropic) | Answer generation, router, verifier |
| **Embeddings** | `BAAI/bge-base-en-v1.5` | Dense vector embeddings (768-dim) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking |
| **Tone scoring** | `ProsusAI/finbert` | Financial sentiment classification |
| **Orchestration** | `LangGraph` | Multi-agent state machine |
| **Prompt caching** | Anthropic API cache_control | 62% cost reduction on repeated queries |

### Retrieval

| Technique | Implementation | Purpose |
|---|---|---|
| **Vector search** | pgvector `<=>` cosine distance | Semantic similarity search |
| **Full-text search** | PostgreSQL `tsvector` + GIN index | Exact term matching |
| **RRF fusion** | Custom implementation (k=60) | Merges vector + FTS ranked lists |
| **Cross-encoder rerank** | `ms-marco-MiniLM-L-6-v2` | Re-scores query-chunk pairs |
| **HNSW index** | pgvector HNSW | Fast approximate nearest-neighbour |

### Infrastructure

| Component | Technology | Purpose |
|---|---|---|
| **Vector database** | PostgreSQL 16 + pgvector | Chunk storage + vector search |
| **API framework** | FastAPI + Pydantic + uvicorn | REST API serving |
| **UI** | Streamlit | Interactive front end |
| **Observability** | Langfuse Cloud | Traces, cost, prompt versioning |
| **Containerisation** | Docker Compose | Local dev orchestration |
| **Package management** | uv | Fast Python dependency resolution |

### Python Libraries

| Library | Version | Purpose |
|---|---|---|
| `anthropic` | >=0.122.0 | Claude API client with prompt caching |
| `langgraph` | latest | Multi-agent graph orchestration |
| `fastapi` | >=0.141.1 | REST API framework |
| `uvicorn` | >=0.52.3 | ASGI server |
| `sentence-transformers` | >=5.7.0 | BGE embeddings + cross-encoder |
| `transformers` | latest | finBERT tone model |
| `psycopg2-binary` | >=2.9.12 | PostgreSQL adapter |
| `pgvector` | >=0.5.0 | pgvector Python client |
| `langfuse` | 2.60.2 | LLM observability |
| `streamlit` | latest | Interactive UI |
| `beautifulsoup4` | >=4.15.0 | iXBRL HTML parsing |
| `lxml` | >=6.1.1 | HTML parser backend |
| `tiktoken` | >=0.13.0 | Token counting for chunking |
| `langchain-text-splitters` | latest | Recursive character text splitter |
| `requests` | >=2.34.2 | SEC EDGAR API calls |
| `python-dotenv` | >=1.2.2 | Environment variable management |
| `deepeval` | latest | LLM evaluation framework |

### External APIs

| API | Authentication | Purpose |
|---|---|---|
| **SEC EDGAR Submissions API** | User-Agent header only (free) | Company filings metadata |
| **SEC EDGAR XBRL API** | User-Agent header only (free) | Structured financial facts |
| **SEC EDGAR Archives** | User-Agent header only (free) | Raw filing document fetch |
| **Anthropic API** | API key | Claude LLM calls |
| **Langfuse Cloud** | Public + secret key pair | Observability dashboard |

---

## Project Structure

```
sentinel-10/
├── src/
│   ├── ingest/
│   │   ├── ingest_filings.py      # Main ingest pipeline
│   │   └── migrate.py             # Database schema migrations
│   ├── retrieval/
│   │   └── retriever.py           # Hybrid search pipeline
│   ├── agents/
│   │   ├── state.py               # LangGraph shared state schema
│   │   ├── nodes.py               # Agent node implementations
│   │   └── graph.py               # LangGraph graph definition
│   ├── api/
│   │   ├── main.py                # FastAPI application
│   │   └── rag_chain.py           # Prompt engineering + LLM calls
│   ├── evals/
│   │   ├── eval.py                # Claude-as-judge eval harness
│   │   └── results/
│   │       └── baseline.json      # Baseline eval scores
│   ├── observability/
│   │   └── tracer.py              # Langfuse tracer initialisation
│   └── ui/
│       └── app.py                 # Streamlit interface
├── notebooks/
│   ├── 01_hello_filing.ipynb      # Phase 0: first EDGAR API call
│   ├── 02_phase1_ingest.ipynb     # Phase 1: ingest exploration
│   ├── 03_retrieval_inspection.ipynb  # Phase 2: retrieval debugging
│   └── 04_debug_sections.ipynb    # iXBRL section parsing debug
├── docs/
│   └── OVERVIEW.md                # Product overview + interview guide
├── docker-compose.yml             # PostgreSQL + pgAdmin + Langfuse
├── pyproject.toml                 # Project dependencies (uv)
├── .env                           # Secrets (never commit)
├── .gitignore
└── README.md
```

---

## Database Schema

```sql
-- Main table: one row per chunk
CREATE TABLE filing_chunks (
    id             SERIAL PRIMARY KEY,
    ticker         TEXT    NOT NULL,          -- e.g. 'AAPL'
    filing_date    TEXT    NOT NULL,          -- e.g. '2024-09-28'
    section        TEXT    NOT NULL,          -- e.g. 'item_1a'
    section_title  TEXT    DEFAULT '',        -- e.g. 'Risk Factors'
    chunk_index    INTEGER NOT NULL,          -- position within section
    text           TEXT    NOT NULL,          -- raw chunk text
    token_count    INTEGER,                   -- tiktoken count
    embedding      vector(768),              -- BGE dense embedding
    fts            tsvector GENERATED ALWAYS  -- auto full-text index
                   AS (to_tsvector('english', text)) STORED,
    created_at     TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast vector search
CREATE INDEX chunks_embedding_idx
ON filing_chunks
USING hnsw (embedding vector_cosine_ops);

-- GIN index for fast full-text search
CREATE INDEX chunks_fts_idx
ON filing_chunks USING gin(fts);
```

---

## API Endpoints

### `GET /health`
Liveness check.
```json
{"status": "ok", "version": "0.1.0"}
```

### `GET /tickers`
Lists ingested companies.
```json
{"tickers": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]}
```

### `POST /ask`
Basic RAG endpoint — single retrieval + generation pass.
```json
// Request
{
  "question": "What are Apple's main cybersecurity risks?",
  "ticker": "AAPL",
  "top_k": 5
}

// Response
{
  "question": "...",
  "answer": "Apple faces... [1]. The company also... [2]",
  "sources": [
    {
      "citation_number": 1,
      "ticker": "AAPL",
      "section": "item_1a",
      "filing_date": "2024-09-28",
      "text": "...",
      "rerank_score": 4.21,
      "rrf_score": 0.032
    }
  ],
  "ticker": "AAPL",
  "latency_ms": 2341,
  "usage": {
    "input_tokens": 4821,
    "output_tokens": 312,
    "cache_read_tokens": 8041,
    "cache_creation_tokens": 0
  }
}
```

### `POST /agent/ask`
Multi-agent endpoint — full LangGraph pipeline with tone scoring and verification.
```json
// Request
{
  "question": "How does Apple describe AI as a risk?",
  "ticker": "AAPL"
}

// Response
{
  "question": "...",
  "ticker": "AAPL",
  "query_type": "qualitative",
  "answer": "...[1]...[2]\n\n**Management Tone:** Neutral (score: 0.54)",
  "sources": [...],
  "tone_score": 0.54,
  "tone_label": "neutral",
  "verified": true,
  "retry_count": 0,
  "usage": {...}
}
```

### `POST /compare`
Same question across multiple companies.
```json
// Request
{
  "question": "How does management describe AI as a risk factor?",
  "tickers": ["AAPL", "MSFT", "NVDA"]
}
```

---

## Multi-Agent Pipeline

### Agent 1 — Router
Classifies the question into `quantitative`, `qualitative`, or `comparison`. Adjusts downstream retrieval strategy (quantitative queries fetch more chunks).

### Agent 2 — Retrieval Agent (parallel)
Runs the hybrid search pipeline:
1. Embed query with BGE `bge-base-en-v1.5`
2. Vector cosine search via pgvector
3. Full-text search via PostgreSQL `tsvector`
4. Merge with Reciprocal Rank Fusion (k=60)
5. Cross-encoder rerank with `ms-marco-MiniLM-L-6-v2`

### Agent 3 — Tone Agent (parallel with Retrieval)
Runs retrieved chunk text through `ProsusAI/finbert` to score management language sentiment. Returns `positive`, `neutral`, or `negative` with a normalised 0–1 score.

### Agent 4 — Synthesis Agent
Builds a numbered context block from retrieved chunks and calls Claude with a system prompt enforcing citation rules. Appends tone score to the final answer.

### Agent 5 — Verifier Agent
Checks whether every factual claim in the generated answer is supported by the retrieved context. If confidence < 0.6 and retries < 2, triggers a retry through the retrieval agent. Makes the system self-correcting.

---

## Retrieval Pipeline — Technical Detail

### Why Hybrid Search?

| Method | Strength | Weakness |
|---|---|---|
| Vector only | Finds semantically similar text | Misses exact terms ("SOX compliance") |
| FTS only | Finds exact terms | Misses paraphrased content |
| Hybrid + RRF | Both | Slightly more complex |

### Reciprocal Rank Fusion Formula

```
RRF(chunk) = Σ 1 / (k + rank_in_list)

Where k = 60 (standard smoothing constant)

Example:
  Chunk A: rank 1 in vector, rank 3 in FTS
  RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

  Chunk B: rank 1 in FTS only
  RRF = 0 + 1/(60+1) = 0.01639

  Chunk A wins — appeared in both lists.
```

### Cross-Encoder Scores

The cross-encoder reads the query and chunk together — unlike bi-encoders (BGE) which embed them separately. Scores are raw logits (can be negative):

```
score > 0    → chunk is relevant
score < 0    → chunk is less relevant (still ranked by relative order)
score = 4.2  → very relevant
score = -4.7 → not relevant
```

---

## Evaluation

### Metrics (Claude-as-judge)

| Metric | Measures | Target |
|---|---|---|
| **Faithfulness** | Are claims grounded in retrieved context? | > 0.80 |
| **Answer Relevancy** | Does the answer address the question? | > 0.80 |
| **Contextual Recall** | Did retrieval surface all needed info? | > 0.70 |
| **Contextual Precision** | Are retrieved chunks relevant? | > 0.70 |

### Baseline Scores (2-question sample)

```json
{
  "faithfulness":        0.9688,
  "answer_relevancy":    0.9750,
  "contextual_recall":   0.5000,
  "contextual_precision":0.3083,
  "overall":             0.6880
}
```

Faithfulness and answer relevancy are strong. Contextual recall and precision are targets for improvement via query expansion, HyDE, and contextual chunking.

---

## Key Engineering Decisions

### Why pgvector over Pinecone?
pgvector keeps vectors in the same database as metadata — no sync layer, no second service, no additional cost. Handles tens of millions of vectors with HNSW indexing. For a project already using PostgreSQL, it's the right call.

### Why Claude claude-sonnet-4-6 with prompt caching?
10-K filings are 150K+ tokens. Without caching, every question about the same company re-processes the full document. Prompt caching cuts input token cost by ~62% on repeated queries to the same ticker.

### Why a custom eval harness over RAGAS/DeepEval?
Dependency conflicts between RAGAS, LangChain, and the Anthropic SDK made both frameworks unworkable in this environment. A custom Claude-as-judge harness with a structured JSON prompt is more transparent, fully controllable, and easier to explain in an interview.

### Why LangGraph over a linear chain?
The verifier retry loop requires a cyclic graph — impossible in a linear chain. LangGraph's stateful graph model handles conditional branching and retry logic cleanly. The explicit state schema also makes debugging straightforward.

### SEC EDGAR URL construction
Modern 10-K filings (post-2020) use company-specific filenames (`aapl-20250927.htm`) not the legacy `{accession}.htm` pattern. The `primaryDocument` field from the EDGAR submissions API is the most reliable source — no index fetch required.

### iXBRL parsing
Modern 10-K filings embed XBRL taxonomy URLs in hidden HTML elements. `BeautifulSoup.get_text()` extracts these as noise. Fix: strip `ix:header`, `ix:hidden`, and `display:none` elements before text extraction. Additionally, filings contain two sets of Item markers — the table of contents (close together, no content between them) and the real section headers (500+ chars of content after each). Distinguish them by content length between consecutive markers.

---

## LLMOps

### Prompt Caching
```python
# Cache the filing context block — expensive to re-process
{
    "type": "text",
    "text": f"CONTEXT FROM SEC FILINGS:\n\n{context_block}",
    "cache_control": {"type": "ephemeral"}  # 5-minute cache window
}
```

Check cache hit in response:
```python
response.usage.cache_read_input_tokens    # > 0 means cache hit
response.usage.cache_creation_input_tokens # > 0 means cache miss (first call)
```

### Langfuse Tracing
Every `/agent/ask` request creates a Langfuse trace with 5 child spans — one per agent node. Each span records:
- Input and output
- Latency
- Token usage (synthesis node)
- Verification result (verifier node)

Dashboard: `https://cloud.langfuse.com`

---

## Setup and Running

### Prerequisites
- Python 3.10+
- Docker Desktop
- `uv` package manager

### 1. Clone and install
```bash
git clone https://github.com/yourname/sentinel-10
cd sentinel-10
uv sync
```

### 2. Environment variables
```bash
cp .env.example .env
# Fill in:
# ANTHROPIC_API_KEY=sk-ant-...
# SEC_USER_AGENT=YourName yourname@email.com
# DATABASE_URL=postgresql://postgres:sentinel123@localhost:5432/sentinel10
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Start database
```bash
docker compose up -d
```

### 4. Ingest filings
```bash
python -m src.ingest.ingest_filings
# Takes ~10 minutes, ingests 5 companies
```

### 5. Run API
```bash
uvicorn src.api.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### 6. Run UI
```bash
streamlit run src/ui/app.py
# Opens at http://localhost:8501
```

### 7. Run eval
```bash
python -m src.evals.eval
# Results saved to src/evals/results/baseline.json
```

---

## Hard-Won Technical Lessons

| Problem | Root Cause | Fix |
|---|---|---|
| Filing URL 404 | Legacy `{accession}.htm` pattern doesn't work for modern filings | Use `primaryDocument` field from EDGAR submissions API directly |
| XBRL noise in chunks | `get_text()` extracts hidden iXBRL taxonomy URLs | Strip `ix:header`, `ix:hidden`, `display:none` elements before extraction |
| Table of contents duplication | Two sets of Item markers — ToC and real headers | Filter by content length between consecutive markers (>500 chars = real header) |
| Docker data loss | Container deleted without named volume | Always define named volumes in docker-compose.yml |
| FTS column missing | Schema added after initial ingest | Migration script `src/ingest/migrate.py` adds column without data loss |
| libnccl.so.2 error | CUDA-linked PyTorch installed on CPU machine | Reinstall with `--index-url https://download.pytorch.org/whl/cpu` |
| Verifier never passes | Tone note appended to answer not grounded in context | Strip tone note before verification |
| JSON parse error in verifier | `max_tokens=100` truncates response mid-JSON | Increase to 512 |
| LangChain import errors | `langchain.text_splitter` moved to separate package | Use `langchain_text_splitters` |
| Jupyter module caching | Python caches modules on first import | Use `importlib.reload()` or clear `sys.modules` |

---

## Roadmap

### Next — Retrieval improvements
- [ ] Query expansion (4 variations per query)
- [ ] HyDE — Hypothetical Document Embeddings
- [ ] MMR — Maximum Marginal Relevance diversity reranking
- [ ] Contextual chunking (Anthropic technique)
- [ ] Parent-child chunk hierarchy

### Data expansion
- [ ] 10-Q filings (quarterly)
- [ ] 8-K filings (material events)
- [ ] Historical filings (8 quarters per company)
- [ ] S&P 500 coverage via EDGAR bulk download
- [ ] Earnings call transcripts

### Signal development
- [ ] Quarter-over-quarter tone score tracking
- [ ] Risk factor change detection (semantic diff)
- [ ] XBRL quantitative layer (revenue, margins, EPS)
- [ ] Backtested tone → price signal study

### Enterprise features
- [ ] On-premise deployment with local LLM (Llama 3)
- [ ] User accounts + saved queries
- [ ] Email alerts on new filings
- [ ] Audit trail per answer
- [ ] SOC 2 compliance preparation

---

## Competing Products

| Product | Price | Sentinel-10 Advantage |
|---|---|---|
| AlphaSense | $20K–$50K/year | Open, auditable, cited answers |
| Sentieo | $15K+/year | Modern AI stack, tone signals |
| Aiera | $10K+/year | Filing depth, not just calls |
| FinChat | Freemium | Multi-agent verification, eval harness |

---

## Author

**Siddhesh Kasar**
Software Engineer → AI Engineer
Building Sentinel-10 as a production AI engineering portfolio project.

---

*Built with Claude claude-sonnet-4-6, LangGraph, pgvector, and SEC EDGAR public data.*