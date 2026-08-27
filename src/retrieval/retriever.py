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
from transformers import pipeline
from src.retrieval.query_expander import expand_query
from src.retrieval.hyde import hyde_embed
import numpy as np

load_dotenv()

# Load models once at module level — expensive to reload per request
print("Loading retrieval models...")
EMBED_MODEL   = SentenceTransformer("BAAI/bge-base-en-v1.5")
RERANK_MODEL  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Retrieval models ready.")
print("Loading finBERT tone model...")
TONE_MODEL = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    return_all_scores=True,
)
print("finBERT ready.")

def _vector_search(query_vec: list[float], ticker: str, top_k: int, conn, form_type=None) -> list[dict]:
    """Pure vector similarity search using pgvector cosine distance."""
    where_clause = "WHERE ticker = %s"
    params       = [query_vec, ticker]

    if form_type:
        where_clause += " AND form_type = %s"
        params.append(form_type)

    params.extend([query_vec, top_k])
    
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            id,
            ticker,
            filing_date,
            form_type,
            section,
            chunk_index,
            text,
            1 - (embedding <=> %s::vector) AS score
        FROM filing_chunks
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, params)

    rows = cur.fetchall()
    cur.close()

    return [
        {
            "id": r[0], "ticker": r[1], "filing_date": r[2],"form_type": r[3],
            "section": r[4], "chunk_index": r[5],
            "text": r[6], "vector_score": float(r[7])
        }
        for r in rows
    ]

def _fulltext_search(query: str, ticker: str,top_k: int, conn, form_type: str= None) -> list[dict]:
    """
    PostgreSQL full-text search using the fts column we created at ingest.
    ts_rank gives a relevance score based on term frequency.
    """
    if form_type:
        where_clause = "WHERE ticker = %s AND form_type = %s AND fts @@ plainto_tsquery('english', %s)"
        params = (query, ticker, form_type, query, top_k)
    else:
        where_clause = "WHERE ticker = %s AND fts @@ plainto_tsquery('english', %s)"
        params = (query, ticker, query, top_k)
    cur = conn.cursor()
    # Quick count check before the full query
    cur.execute(f"""
        SELECT
            id,
            ticker,
            filing_date,
            section,
            chunk_index,
            text,
            ts_rank(fts, plainto_tsquery('english', %s)) AS score
        FROM filing_chunks
        {where_clause}
        ORDER BY score DESC
        LIMIT %s
    """, params)

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
# ── Three-list RRF ─────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion_three(
    vector_results: list[dict],
    hyde_results:   list[dict],
    fts_results:    list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merges THREE ranked lists using Reciprocal Rank Fusion.

    Same formula as two-list RRF but sums contributions from:
      1. Regular BGE vector search
      2. HyDE vector search (hypothetical document embedding)
      3. PostgreSQL full-text search

    A chunk appearing in all three lists gets maximum score.
    Chunks only in one list get a lower score.

    Score range: 0 to ~0.049 (3 × 1/(60+1))
    """
    scores = {}
    chunks = {}

    # Process all three lists with the same formula
    for result_list in [vector_results, hyde_results, fts_results]:
        for rank, chunk in enumerate(result_list):
            cid         = chunk["id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    merged = []
    for cid in sorted_ids:
        chunk              = chunks[cid].copy()
        chunk["rrf_score"] = round(scores[cid], 6)
        merged.append(chunk)

    return merged

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

# ── MMR — Maximum Marginal Relevance ──────────────────────────────────────────

def _mmr_rerank(
    query_vec:     list[float],
    chunks:        list[dict],
    top_k:         int,
    lambda_param:  float = 0.7
) -> list[dict]:
    """
    Maximum Marginal Relevance reranking.

    Balances two competing goals:
      Relevance  — how similar is the chunk to the query?
      Diversity  — how different is this chunk from already-selected chunks?

    MMR score = λ × relevance - (1-λ) × max_similarity_to_selected

    lambda_param:
      1.0 = pure relevance (same as regular rerank)
      0.0 = pure diversity (maximally different chunks)
      0.7 = good balance for financial QA — relevant but not redundant

    Why this matters:
      Without MMR: top 5 chunks might all be slight variations of the
                   same paragraph. Wastes context window, hurts answer quality.
      With MMR:    top 5 chunks cover different aspects of the answer.
                   Better recall, better synthesis.
    """
    if not chunks:
        return []

    # Embed all candidate chunks
    candidate_texts = [c["text"] for c in chunks]
    candidate_vecs  = EMBED_MODEL.encode(
        candidate_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    query_arr = np.array(query_vec)

    selected_indices = []
    remaining        = list(range(len(chunks)))

    for _ in range(min(top_k, len(chunks))):
        best_score = float("-inf")
        best_idx   = None

        for i in remaining:
            # Relevance: cosine similarity to query
            relevance = float(np.dot(query_arr, candidate_vecs[i]))

            # Redundancy: max similarity to any already-selected chunk
            if not selected_indices:
                redundancy = 0.0
            else:
                selected_vecs = candidate_vecs[selected_indices]
                similarities  = np.dot(selected_vecs, candidate_vecs[i])
                redundancy    = float(np.max(similarities))

            # MMR score — balance relevance against redundancy
            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx   = i

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    result = [chunks[i] for i in selected_indices]
    print(f"[MMR] Selected {len(result)} diverse chunks from {len(chunks)} candidates")
    return result

def retrieve(
    query:   str,
    ticker:  str,
    top_k:   int = 5,           # final chunks returned to LLM
    fetch_k: int = 15,          # candidates before reranking
    form_type=None,
    use_hyde: bool = True,
    use_expansion:bool  = True,
    use_mmr:      bool  = True,
    lambda_mmr:   float = 0.7,
) -> list[dict]:
    """
    Main retrieval function. Full pipeline:

      1. Query expansion    → 4 query variations
      2. Regular embedding  → vector search
      3. HyDE embedding     → vector search (if use_hyde=True)
      4. Full-text search   → FTS across all query variations
      5. Three-way RRF      → unified ranked list
      6. Cross-encoder      → precision reranking
      7. MMR                → diversity filter (if use_mmr=True)

    Returns top_k chunks ready to send to the LLM.
    """
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    
    # Step 1: Expand query into 4 variations
    queries = expand_query(query) if use_expansion else [query]
    
    all_vector_results = []
    all_fts_results    = []
    

    # Step 2: embed each query
    for q in queries:
        qvec = EMBED_MODEL.encode(q, normalize_embeddings=True).tolist()
        all_vector_results.extend(_vector_search(qvec, ticker, fetch_k, conn, form_type))
        all_fts_results.extend(_fulltext_search(q, ticker, fetch_k, conn, form_type))
    
    # Deduplicate by chunk id before RRF
    seen = set()
    unique_vector = []
    for r in all_vector_results:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique_vector.append(r)
    
    # Deduplicate FTS results
    seen       = set()
    unique_fts = []
    for r in all_fts_results:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique_fts.append(r)
    
    # ── Step 3: HyDE search ────────────────────────────────────────────────────
    hyde_results = []
    if use_hyde:
        hyde_vec = hyde_embed(query)
        if hyde_vec:
            hyde_results = _vector_search(hyde_vec, ticker, fetch_k, conn, form_type)
    
    conn.close()

    # Step 4 — merge with RRF
    if hyde_results:
        merged = _reciprocal_rank_fusion_three(
            unique_vector, hyde_results, unique_fts
        )
        print(f"[Retrieve] Three-way RRF: "
              f"{len(unique_vector)} vec + {len(hyde_results)} hyde "
              f"+ {len(unique_fts)} fts → {len(merged)} merged")
    else:
        merged = _reciprocal_rank_fusion(unique_vector, unique_fts)
        print(f"[Retrieve] Two-way RRF: "
              f"{len(unique_vector)} vec + {len(unique_fts)} fts "
              f"→ {len(merged)} merged")

    # ── Step 5: Cross-encoder rerank ──────────────────────────────────────────
    # Rerank top fetch_k candidates for precision
    reranked = _rerank(query, merged[:fetch_k], top_n=fetch_k)
    print(f"[Retrieve] After rerank: top score = "
          f"{reranked[0]['rerank_score'] if reranked else 'N/A'}")
    
    # ── Step 6: MMR diversity filter ──────────────────────────────────────────
    if use_mmr and len(reranked) > top_k:
        query_vec = EMBED_MODEL.encode(query, normalize_embeddings=True).tolist()
        final     = _mmr_rerank(query_vec, reranked, top_k, lambda_mmr)
    else:
        final = reranked[:top_k]

    print(f"[Retrieve] Final: {len(final)} chunks returned\n")

    return final


