"""
src/retrieval/retriever.py

Hybrid search: vector similarity + full-text search, merged
with Reciprocal Rank Fusion, reranked with a cross-encoder.

Why hybrid?
  - Vector search finds semantically similar text even if
    the exact words differ ("cybersecurity" finds "data breach")
  - Full-text search finds exact terms vector search can miss
    ("SOX compliance" — a specific term that needs exact matching)
  - Together they beat either alone by ~15% on recall
"""

import os
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Load models once at module level — expensive to reload per request
print("Loading retrieval models...")
EMBED_MODEL   = SentenceTransformer("BAAI/bge-base-en-v1.5")
RERANK_MODEL  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Retrieval models ready.")

def _vector_search(query_vec: list[float], ticker: str, top_k: int, conn) -> list[dict]:
    """Pure vector similarity search using pgvector cosine distance."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id,
            ticker,
            filing_date,
            section,
            chunk_index,
            text,
            1 - (embedding <=> %s::vector) AS score
        FROM filing_chunks
        WHERE ticker = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_vec, ticker, query_vec, top_k))

    rows = cur.fetchall()
    cur.close()

    return [
        {
            "id": r[0], "ticker": r[1], "filing_date": r[2],
            "section": r[3], "chunk_index": r[4],
            "text": r[5], "vector_score": float(r[6])
        }
        for r in rows
    ]

def _fulltext_search(query: str, ticker: str,top_k: int, conn) -> list[dict]:
    """
    PostgreSQL full-text search using the fts column we created at ingest.
    ts_rank gives a relevance score based on term frequency.
    """
    cur = conn.cursor()
    # Quick count check before the full query
    cur.execute("""
        SELECT
            id,
            ticker,
            filing_date,
            section,
            chunk_index,
            text,
            ts_rank(fts, plainto_tsquery('english', %s)) AS score
        FROM filing_chunks
        WHERE ticker = %s
          AND fts @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
    """, (query, ticker, query, top_k))

    rows = cur.fetchall()
    cur.close()

    return [
        {
            "id": r[0], "ticker": r[1], "filing_date": r[2],
            "section": r[3], "chunk_index": r[4],
            "text": r[5], "fts_score": float(r[6])
        }
        for r in rows
    ]

def _reciprocal_rank_fusion(
    vector_results: list[dict],
    fts_results: list[dict],
    k: int = 60                 # RRF constant — 60 is the standard default
) -> list[dict]:
    """
    Merges two ranked lists into one using Reciprocal Rank Fusion.

    RRF score = 1/(k + rank_in_list_1) + 1/(k + rank_in_list_2)

    Why RRF and not just average the scores?
    Vector scores and FTS scores are on completely different scales
    (0.85 cosine vs 0.003 ts_rank). You can't average them directly.
    RRF uses rank position instead of raw score — scale-independent.
    """

    scores = {}   # id → rrf_score
    chunks = {}   # id → chunk dict

    for rank, chunk in enumerate(vector_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunks[cid] = chunk

    for rank, chunk in enumerate(fts_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in chunks:
            chunks[cid] = chunk

    # Sort by combined RRF score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    merged = []
    for cid in sorted_ids:
        chunk = chunks[cid].copy()
        chunk["rrf_score"] = round(scores[cid], 6)
        merged.append(chunk)

    return merged

def _rerank(query: str, candidates: list[dict],
            top_n: int) -> list[dict]:
    """
    Cross-encoder reranking: takes the top ~10 RRF candidates and
    scores each (query, chunk) pair more carefully.

    Why rerank?
    The cross-encoder reads query AND chunk together — it understands
    relevance in context, not just similarity in isolation.
    It's slower (can't be pre-indexed) but much more accurate.
    We run it on only 10 candidates to keep latency low.
    """
    if not candidates:
        return []

    pairs  = [(query, c["text"]) for c in candidates]
    scores = RERANK_MODEL.predict(pairs)   # returns a float per pair

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidates,
                      key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]

def retrieve(
    query:   str,
    ticker:  str,
    top_k:   int = 5,           # final chunks returned to LLM
    fetch_k: int = 10           # candidates before reranking
) -> list[dict]:
    """
    Main retrieval function. Called by the API layer.

    Full pipeline:
      1. Embed query with BGE
      2. Vector search → top fetch_k chunks
      3. Full-text search → top fetch_k chunks
      4. RRF merge → unified ranked list
      5. Cross-encoder rerank → top_k final chunks
    """

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    # Step 1 — embed the query
    query_vec = EMBED_MODEL.encode(
        query, normalize_embeddings=True
    ).tolist()

    # Steps 2 & 3 — parallel searches
    vector_results = _vector_search(query_vec, ticker, fetch_k, conn)
    fts_results    = _fulltext_search(query, ticker, fetch_k, conn)

    conn.close()

    # Step 4 — merge with RRF
    merged = _reciprocal_rank_fusion(vector_results, fts_results)

    # Step 5 — rerank top candidates
    final = _rerank(query, merged[:fetch_k], top_k)

    return final


