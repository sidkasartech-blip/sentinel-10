"""
src/api/main.py

FastAPI application exposing the RAG pipeline as a REST API.
Run with: uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.retrieval.retriever import retrieve
from src.api.rag_chain import generate_answer
import time
from src.agents.graph import run_agent

app = FastAPI(
    title="Sentinel-10",
    description="Multi-agent SEC filings research copilot",
    version="0.1.0"
)

# ── Request / Response models ─────────────────────────────────────────────────
# Pydantic models do two things: validate incoming JSON and
# document your API automatically at http://localhost:8000/docs

SUPPORTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]

class AskRequest(BaseModel):
    question: str  = Field(..., min_length=10,
                           example="What are Apple's main risk factors?")
    ticker:   str  = Field(..., example="AAPL")
    top_k:    int  = Field(3, ge=1, le=10,
                           description="Number of chunks to retrieve")

class CompareRequest(BaseModel):
    question: str        = Field(..., min_length=10,
                                 example="How does management describe AI as a risk factor?")
    tickers:  list[str] = Field(..., min_items=2, max_items=5,
                                 example=["AAPL", "MSFT", "NVDA"])
    top_k:    int        = Field(3, ge=1, le=10,
                                 description="Chunks per company")

class SourceChunk(BaseModel):
    citation_number: int
    ticker:          str
    section:         str
    filing_date:     str
    text:            str
    rerank_score:    float | None
    rrf_score:       float | None

class AskResponse(BaseModel):
    question:      str
    answer:        str
    sources:       list[SourceChunk]
    ticker:        str
    latency_ms:    int
    usage:         dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness check — useful for Docker health checks later."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/tickers")
def list_tickers():
    """Returns the list of ingested companies."""
    return {"tickers": SUPPORTED_TICKERS}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Main RAG endpoint.
    Retrieves relevant chunks from the filing database and
    generates a cited answer using Claude.
    """
    ticker = req.ticker.upper()

    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker {ticker} not ingested. "
                   f"Supported: {SUPPORTED_TICKERS}"
        )

    start = time.time()

    # Step 1 — retrieve relevant chunks
    chunks = retrieve(
        query=req.question,
        ticker=ticker,
        top_k=req.top_k
    )

    # Step 2 — generate cited answer
    result = generate_answer(req.question, chunks)

    latency_ms = int((time.time() - start) * 1000)

    return AskResponse(
        question=req.question,
        answer=result["answer"],
        sources=result["sources"],
        ticker=ticker,
        latency_ms=latency_ms,
        usage=result["usage"]
    )

@app.post("/compare")
def compare(req: CompareRequest):
    """
    Ask the same question across multiple companies.
    Useful for competitive analysis — preview of multi-agent work.
    """
    tickers = [t.upper() for t in req.tickers]
    invalid = [t for t in tickers if t not in SUPPORTED_TICKERS]
    if invalid:
        raise HTTPException(400, detail=f"Unsupported tickers: {invalid}")

    results = {}
    for ticker in tickers:
        chunks = retrieve(req.question, ticker, top_k=req.top_k)
        result = generate_answer(req.question, chunks)
        results[ticker] = {
            "answer":  result["answer"],
            "sources": result["sources"]
        }

    return {"question": req.question, "results": results}

@app.post("/agent/ask")
def agent_ask(req: AskRequest):
    """
    Multi-agent endpoint — routes through the full LangGraph pipeline.
    Slower than /ask but self-verifying and tone-aware.
    """
    ticker = req.ticker.upper()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(400, detail=f"Unsupported ticker: {ticker}")

    start  = time.time()
    result = run_agent(req.question, ticker)
    result["latency_ms"] = int((time.time() - start) * 1000)

    return result