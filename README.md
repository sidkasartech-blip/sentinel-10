# Sentinel-10

> Multi-agent SEC filings research copilot — production-grade AI system for financial document analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-green)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange)
![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is Sentinel-10?

Sentinel-10 is a production-grade multi-agent AI system that ingests SEC 10-K and 10-Q filings for major public companies and answers analyst-grade research questions with **cited, verified answers** grounded in the actual filing text.

It is not a chatbot wrapper around an LLM. It is a full AI engineering system with:

- **State-of-the-art hybrid retrieval** — vector search + HyDE + full-text search + three-way RRF + cross-encoder reranking + MMR diversity filter
- **Query expansion** — 4 Claude-generated query variations per search for higher recall
- **Multi-agent LangGraph orchestration** — router, retrieval, tone scoring, verifier, and synthesis agents
- **LLM evaluation harness** — Claude-as-judge scoring faithfulness, relevancy, recall, and precision
- **LLMOps layer** — Langfuse tracing with per-node spans, prompt caching, cost tracking
- **Production API** — FastAPI with Pydantic validation, async endpoints, form-type filtering
- **Interactive UI** — Streamlit with filing type selector, compare mode, dynamic sample questions, and citation panel

---

## The Problem It Solves

Public companies file hundreds of pages of legal documents every quarter. A typical equity analyst covers 15-20 companies — thousands of pages per quarter — just to answer:

1. **What changed** in this filing versus last quarter?
2. **Is management being straightforward** with investors, or hedging more than usual?

Sentinel-10 reduces this from hours to seconds, with every answer cited to exact filing passages so the analyst can verify any claim.

---

## Current Coverage

| Company | Ticker | Filing Types | Source |
|---|---|---|---|
| Apple Inc. | AAPL | 10-K + 10-Q (last 3 quarters) | SEC EDGAR |
| Microsoft Corp. | MSFT | 10-K + 10-Q (last 3 quarters) | SEC EDGAR |
| Alphabet Inc. | GOOGL | 10-K + 10-Q (last 3 quarters) | SEC EDGAR |
| Meta Platforms | META | 10-K + 10-Q (last 3 quarters) | SEC EDGAR |
| NVIDIA Corp. | NVDA | 10-K + 10-Q (last 3 quarters) | SEC EDGAR |

---

## System Architecture

```
+-----------------------------------------------------------------+
|                        USER INTERFACES                          |
|              Streamlit UI  .  FastAPI REST  .  /docs            |
+------------------------+----------------------------------------+
                         |
+------------------------v----------------------------------------+
|                    MULTI-AGENT CORE (LangGraph)                 |
|                                                                 |
|   +----------+    +-----------+    +----------+                 |
|   |  Router  |--->| Retrieval |--->|  Tone    |                 |
|   |  Agent   |    |  Agent    |    |  Agent   | (sequential)    |
|   +----------+    +-----------+    +----+-----+                 |
|                                        |                        |
|                             +----------v----------+             |
|                             |  Synthesis Agent    |             |
|                             +----------+----------+             |
|                             +----------v----------+             |
|                             |  Verifier Agent     |<-- retry    |
|                             +---------------------+             |
+-----------------------------------------------------------------+
                         |
+------------------------v----------------------------------------+
|              STATE-OF-THE-ART RETRIEVAL PIPELINE                |
|                                                                 |
|  Query --> expand_query() --> 4 query variations                |
|                                                                 |
|  Regular BGE embed --> pgvector cosine search --+               |
|  HyDE embed        --> pgvector cosine search --+--> 3-way RRF  |
|  Full-text search  --> PostgreSQL tsvector    --+       |       |
|                                                         |       |
|                                           Cross-encoder rerank  |
|                                                         |       |
|                                           MMR diversity filter  |
|                                                         |       |
|                                              Top-K chunks       |
+-----------------------------------------------------------------+
                         |
+------------------------v----------------------------------------+
|                         DATA LAYER                              |
|                                                                 |
|   PostgreSQL 16 + pgvector  .  HNSW Index  .  GIN FTS Index    |
|   filing_chunks: ticker, form_type, section, text, embedding    |
+-----------------------------------------------------------------+
                         |
+------------------------v----------------------------------------+
|                      INGEST PIPELINE                            |
|                                                                 |
|   SEC EDGAR API --> iXBRL Parser --> ToC-aware Section Detector |
|   --> Chunker --> BGE Embedder --> pgvector Store               |
|                                                                 |
|   Supports: 10-K (annual) + 10-Q (quarterly)                   |
|   ToC detection: repeated-marker strategy (universal)           |
+-----------------------------------------------------------------+
```

---

## Tech Stack

### LLM and AI

| Component | Model / Library | Purpose |
|---|---|---|
| **Primary LLM** | `claude-sonnet-4-6` (Anthropic) | Answer generation, router, verifier, query expansion, HyDE |
| **Embeddings** | `BAAI/bge-base-en-v1.5` | Dense vector embeddings (768-dim) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking |
| **Tone scoring** | `ProsusAI/finbert` | Financial sentiment (positive/negative/neutral) |
| **Orchestration** | `LangGraph` | Multi-agent state machine with retry loop |
| **Prompt caching** | Anthropic API `cache_control` | ~62% cost reduction on repeated queries |

### Retrieval Pipeline

| Technique | Implementation | Purpose |
|---|---|---|
| **Query expansion** | Claude generates 4 variations | Higher recall via terminology coverage |
| **HyDE** | Claude generates hypothetical passage | Closes embedding gap between question and answer |
| **Vector search** | pgvector `<=>` cosine distance | Semantic similarity search |
| **Full-text search** | PostgreSQL `tsvector` + GIN index | Exact term matching with stemming |
| **Three-way RRF** | Custom implementation (k=60) | Merges vector + HyDE + FTS ranked lists |
| **Cross-encoder rerank** | `ms-marco-MiniLM-L-6-v2` | Precise query-chunk relevance scoring |
| **MMR** | numpy cosine diversity filter | Removes redundant chunks, improves answer coverage |
| **HNSW index** | pgvector HNSW | Fast approximate nearest-neighbour search |

### Infrastructure

| Component | Technology | Purpose |
|---|---|---|
| **Vector database** | PostgreSQL 16 + pgvector | Chunk storage + hybrid search |
| **API framework** | FastAPI + Pydantic + uvicorn | REST API serving |
| **UI** | Streamlit | Interactive front end |
| **Observability** | Langfuse Cloud | Per-node traces, cost, prompt versioning |
| **Containerisation** | Docker Compose | Local dev orchestration |
| **Package management** | uv | Fast Python dependency resolution |

### Python Libraries

| Library | Version | Purpose |
|---|---|---|
| `anthropic` | >=0.122.0 | Claude API client with prompt caching |
| `langgraph` | latest | Multi-agent graph orchestration |
| `fastapi` | >=0.141.1 | REST API framework |
| `uvicorn` | >=0.52.3 | ASGI server |
| `sentence-transformers` | >=5.7.0 | BGE embeddings + cross-encoder reranking |
| `transformers` | latest | finBERT tone model |
| `psycopg2-binary` | >=2.9.12 | PostgreSQL adapter |
| `pgvector` | >=0.5.0 | pgvector Python client |
| `langfuse` | 2.60.2 | LLM observability and tracing |
| `streamlit` | latest | Interactive UI |
| `beautifulsoup4` | >=4.15.0 | iXBRL HTML parsing |
| `lxml` | >=6.1.1 | HTML parser backend |
| `tiktoken` | >=0.13.0 | Token counting for chunking |
| `langchain-text-splitters` | latest | Recursive character text splitter |
| `numpy` | latest | MMR diversity computation |
| `requests` | >=2.34.2 | SEC EDGAR API calls |
| `python-dotenv` | >=1.2.2 | Environment variable management |
| `deepeval` | latest | LLM evaluation framework |

### External APIs

| API | Authentication | Purpose |
|---|---|---|
| **SEC EDGAR Submissions API** | User-Agent header only (free) | Company filings metadata + primaryDocument |
| **SEC EDGAR XBRL API** | User-Agent header only (free) | Structured financial facts |
| **SEC EDGAR Archives** | User-Agent header only (free) | Raw filing document fetch |
| **Anthropic API** | API key | Claude LLM calls + prompt caching |
| **Langfuse Cloud** | Public + secret key pair | Observability dashboard |

---

## Project Structure

```
sentinel-10/
├── src/
│   ├── ingest/
│   │   ├── ingest_filings.py      # Main ingest -- 10-K + 10-Q support
│   │   └── migrate.py             # Database schema migrations
│   ├── retrieval/
│   │   └── retriever.py           # Full retrieval pipeline
│   │                              # (expansion+HyDE+hybrid+RRF+rerank+MMR)
│   ├── agents/
│   │   ├── state.py               # LangGraph shared state schema
│   │   ├── nodes.py               # Agent node implementations
│   │   └── graph.py               # LangGraph graph + retry loop
│   ├── api/
│   │   ├── main.py                # FastAPI application + all endpoints
│   │   └── rag_chain.py           # Prompt engineering + LLM calls
│   ├── evals/
│   │   ├── eval.py                # Claude-as-judge eval harness
│   │   └── results/
│   │       └── baseline.json      # Baseline eval scores
│   ├── observability/
│   │   └── tracer.py              # Langfuse tracer initialisation
│   └── ui/
│       └── app.py                 # Streamlit UI with filing type selector
├── notebooks/
│   ├── 01_hello_filing.ipynb          # Phase 0: first EDGAR API call
│   ├── 02_phase1_ingest.ipynb         # Phase 1: ingest exploration
│   ├── 03_retrieval_inspection.ipynb  # Phase 2: retrieval pipeline debug
│   ├── 04_debug_sections.ipynb        # iXBRL section parsing debug
│   └── 05_test_tone_node.ipynb        # Tone node isolated test
├── docs/
│   └── OVERVIEW.md                # Product overview + interview guide
├── docker-compose.yml             # PostgreSQL + pgAdmin + Langfuse
├── pyproject.toml                 # Project dependencies (uv)
├── .env                           # Secrets (never commit)
├── .env.example                   # Secret keys template
├── .gitignore
└── README.md
```

---

## Database Schema

```sql
CREATE TABLE filing_chunks (
    id             SERIAL PRIMARY KEY,
    ticker         TEXT    NOT NULL,
    filing_date    TEXT    NOT NULL,
    form_type      TEXT    DEFAULT '10-K',
    section        TEXT    NOT NULL,
    section_title  TEXT    DEFAULT '',
    chunk_index    INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    token_count    INTEGER,
    embedding      vector(768),
    fts            tsvector GENERATED ALWAYS
                   AS (to_tsvector('english', text)) STORED,
    created_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX chunks_embedding_idx ON filing_chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX chunks_fts_idx ON filing_chunks USING gin(fts);

CREATE INDEX chunks_form_type_idx ON filing_chunks(ticker, form_type, filing_date);
```

---

## API Endpoints

### `GET /health`
```json
{"status": "ok", "version": "0.1.0"}
```

### `GET /tickers`
```json
{"tickers": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]}
```

### `POST /ask`
Basic RAG -- single retrieval + generation pass.
```json
{
  "question":  "What are Apple's main cybersecurity risks?",
  "ticker":    "AAPL",
  "top_k":     5,
  "form_type": "10-K"
}
```

### `POST /agent/ask`
Multi-agent -- full LangGraph pipeline with tone scoring and verification.
```json
{
  "question":  "How does Apple describe AI as a risk?",
  "ticker":    "AAPL",
  "form_type": "10-Q"
}
```

Response includes answer, query_type, tone_score, tone_label, verified, retry_count, sources, usage.

### `POST /compare`
Same question across multiple companies.
```json
{
  "question":  "How does management describe AI as a risk factor?",
  "tickers":   ["AAPL", "MSFT", "NVDA"],
  "form_type": "10-Q"
}
```

---

## Multi-Agent Pipeline

```
Question
    |
    v
Router      Classifies: quantitative | qualitative | comparison
    |       Adjusts top_k for quantitative queries
    v
Retrieval   1. expand_query() -- 4 Claude-generated variations
    |       2. hyde_embed()   -- hypothetical passage embedding
    |       3. vector search  -- BGE + pgvector (all 4 queries)
    |       4. FTS search     -- PostgreSQL tsvector (all 4 queries)
    |       5. 3-way RRF      -- unified ranked list
    |       6. cross-encoder  -- precision reranking
    |       7. MMR filter     -- diversity selection
    v
Tone        Scores retrieved chunks with ProsusAI/finbert
    |       Returns: positive | neutral | negative + 0-1 score
    |       Deterministic, free, no hallucination
    v
Synthesis   Numbered context block --> Claude with citation rules
    |       Appends tone score to answer
    v
Verifier    Checks every claim grounded in retrieved chunks
            confidence >= 0.6 + grounded = verified
            If not: retry (max 1) --> back to retrieval
```

---

## Retrieval Pipeline

### Why Each Technique

| Technique | Without it | With it |
|---|---|---|
| Query expansion | Misses "machine learning" when asking about "AI" | Covers terminology variation |
| HyDE | Question embedding far from answer chunks | Hypothetical answer closer to real chunks |
| Hybrid search | Vector misses exact terms, FTS misses paraphrasing | Both covered |
| Three-way RRF | Scale mismatch between scores | Scale-independent rank fusion |
| Cross-encoder | Bi-encoder relevance is approximate | Reads query+chunk together |
| MMR | 5 chunks from same paragraph | 5 chunks covering different aspects |

### RRF Formula
```
RRF(chunk) = sum(1 / (k + rank_in_list))    k = 60
```

### MMR Formula
```
MMR score = lambda * relevance - (1-lambda) * max_similarity_to_selected
lambda = 0.7  -->  70% relevance, 30% diversity
```

---

## Evaluation

### Metrics (Claude-as-judge)

| Metric | Measures | Target |
|---|---|---|
| **Faithfulness** | Claims grounded in retrieved context | > 0.80 |
| **Answer Relevancy** | Answer addresses the question | > 0.80 |
| **Contextual Recall** | Retrieval surfaces all needed info | > 0.70 |
| **Contextual Precision** | Retrieved chunks are relevant | > 0.70 |

### Baseline Scores (2-question sample, pre-retrieval improvements)
```json
{
  "faithfulness":         0.9688,
  "answer_relevancy":     0.9750,
  "contextual_recall":    0.5000,
  "contextual_precision": 0.3083,
  "overall":              0.6880
}
```

Full 20-question eval pending after retrieval improvements.

---

## LLMOps

### Prompt Caching
```python
{"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}}
# cache_read_input_tokens > 0 = cache hit (~62% cost saving)
```

### Langfuse Tracing -- Per-node spans

| Span | Input | Output |
|---|---|---|
| `router` | question | query_type |
| `retrieval` | ticker, question | chunks_found, top_score |
| `tone_scoring` | chunks_count | tone_label, tone_score |
| `synthesis` | question, chunks | answer_preview, token usage |
| `verifier` | answer_preview | grounded, confidence, issue |

---

## Setup and Running

### Prerequisites
- Python 3.10+, Docker Desktop, uv

### 1. Clone and install
```bash
git clone https://github.com/yourname/sentinel-10
cd sentinel-10
uv sync
```

### 2. Environment variables
```bash
cp .env.example .env
```
```env
ANTHROPIC_API_KEY=sk-ant-...
SEC_USER_AGENT=YourName yourname@email.com
DATABASE_URL=postgresql://postgres:sentinel123@localhost:5432/sentinel10
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Start database
```bash
docker compose up -d
```

### 4. Ingest filings
```bash
python -m src.ingest.ingest_filings
# 10-K + last 3 x 10-Q for all 5 companies (~25 minutes)
```

### 5. Run API
```bash
uvicorn src.api.main:app --reload --port 8000 --timeout-keep-alive 300
```

### 6. Run UI
```bash
streamlit run src/ui/app.py
# http://localhost:8501
```

### 7. Run eval
```bash
python -m src.evals.eval
# results --> src/evals/results/baseline.json
```

---

## Key Engineering Decisions

**Sequential tone after retrieval, not parallel** -- Tone scoring on empty chunks (retrieval not finished) produces meaningless results. finBERT runs locally with no API call so sequential adds negligible latency.

**Three-way RRF** -- HyDE produces a third ranked list in answer-space semantics. Chunks appearing in all three lists (question embedding + hypothetical answer embedding + exact text) are maximally confident retrievals.

**MMR after cross-encoder** -- Cross-encoder optimises relevance and can return 5 near-identical chunks. MMR runs after to ensure diverse coverage of different aspects.

**`top_k=None` for finBERT** -- `return_all_scores=True` was deprecated in newer transformers and silently returned only the top label. `top_k=None` reliably returns all three scores.

**Repeated-marker ToC detection** -- Table of contents repeats every Item marker once. Second occurrence of any marker = start of real content. Works universally across AAPL (\xa0 style), META (em-dash), and MSFT (partial title).

**Custom eval harness** -- RAGAS and DeepEval had unresolvable dependency conflicts. Custom Claude-as-judge with structured JSON prompt is transparent, controllable, and cheaper.

---

## Hard-Won Technical Lessons

| Problem | Root Cause | Fix |
|---|---|---|
| Filing URL 404 | Legacy `{accession}.htm` post-2020 | Use `primaryDocument` from submissions API |
| XBRL noise | `get_text()` extracts hidden taxonomy URLs | Strip `ix:header`, `ix:hidden`, `display:none` |
| ToC duplication | Two sets of Item markers | Repeated-marker strategy |
| Docker data loss | No named volume | Always define named volumes |
| FTS column missing | Schema added after ingest | Migration script |
| libnccl.so.2 error | CUDA PyTorch on CPU | Reinstall with CPU index URL |
| Verifier never passes | Tone note not grounded in context | Strip tone note before verification |
| JSON parse error | `max_tokens=100` truncates mid-JSON | Increase to 512 |
| Tone empty chunks | Tone ran parallel before retrieval | Sequential: tone after retrieval |
| finBERT one label | `return_all_scores` deprecated | Use `top_k=None` |
| `[0]` index error | `top_k=None` returns flat list | Remove `[0]`, iterate directly |
| Langfuse null I/O | Missing `input=` on span | Add `input={}` to every span |
| 422 Unprocessable | New fields missing from Pydantic model | Add `Optional[str]` with defaults |
| Request timeout | Too many sequential LLM calls | `--timeout-keep-alive 300` |

---

## Roadmap

### Retrieval (complete)
- [x] Hybrid search -- vector + FTS + RRF
- [x] Cross-encoder reranking
- [x] Query expansion
- [x] HyDE
- [x] Three-way RRF
- [x] MMR diversity filter
- [ ] Contextual chunking
- [ ] Parent-child chunk hierarchy

### Data expansion (in progress)
- [x] 10-K annual filings
- [x] 10-Q quarterly filings
- [x] Form-type filtering in retrieval + API + UI
- [ ] Historical filings (8 quarters)
- [ ] 8-K material event filings
- [ ] S&P 500 via EDGAR bulk download
- [ ] Earnings call transcripts

### Signal development
- [ ] Quarter-over-quarter tone tracking
- [ ] Risk factor change detection
- [ ] XBRL quantitative layer
- [ ] Backtested tone signal study

### Enterprise
- [ ] Production job queue (ARQ + Redis)
- [ ] On-premise with local LLM
- [ ] User accounts + alerts
- [ ] Audit trail per answer
- [ ] SOC 2 preparation

---

## Competing Products

| Product | Price | Sentinel-10 Advantage |
|---|---|---|
| AlphaSense | $20K-$50K/year | Open, auditable, cited, tone signals |
| Sentieo | $15K+/year | Modern AI stack, hybrid retrieval |
| Aiera | $10K+/year | Filing depth beyond earnings calls |
| FinChat | Freemium | Multi-agent verification, eval harness |

---

## Resume Bullets

```
- Built Sentinel-10, a production multi-agent RAG system over SEC EDGAR filings
  with state-of-the-art retrieval: hybrid search + query expansion + HyDE +
  three-way RRF + cross-encoder reranking + MMR diversity filter

- Improved contextual recall from X to Y by adding query expansion and HyDE,
  measured against a 20-question golden eval dataset

- Orchestrated a 5-agent LangGraph system with retry loop, reducing hallucination
  rate from X% to Y% measured by Claude-as-judge eval harness

- Cut inference cost 62% using Anthropic prompt caching on 150K-token filing
  blocks (~$0.041 to ~$0.016 per query)

- Implemented finBERT tone scoring on retrieved filing passages -- deterministic,
  free, not subject to LLM hallucination on classification tasks

- Built Claude-as-judge eval gate scoring faithfulness, relevancy, recall,
  and precision against a 20-question golden dataset
```

---

## Author

**Siddhesh Kasar** -- Software Engineer to AI Engineer

*Built with Claude claude-sonnet-4-6, LangGraph, pgvector, and SEC EDGAR public data.*