"""
src/ingest/ingest_filings.py

Ingests 10-K filings from SEC EDGAR for a list of tickers.
Run with: python -m src.ingest.ingest_filings

This is the production version of notebooks/02_phase1_ingest.ipynb
"""

import os
import re
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

HEADERS     = {"User-Agent": os.getenv("SEC_USER_AGENT")}
CONN_STR    = os.getenv("DATABASE_URL")
TICKERS     = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
FORM_TYPE   = "10-K"
CHUNK_SIZE  = 500   # tokens
CHUNK_OVERLAP = 50  # tokens

# ── Database setup ────────────────────────────────────────────────────────────

def setup_database():
    conn = psycopg2.connect(CONN_STR)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS filing_chunks (
            id           SERIAL PRIMARY KEY,
            ticker       TEXT    NOT NULL,
            filing_date  TEXT    NOT NULL,
            section      TEXT    NOT NULL,
            chunk_index  INTEGER NOT NULL,
            text         TEXT    NOT NULL,
            token_count  INTEGER,
            embedding    vector(768),
            created_at   TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON filing_chunks
        USING hnsw (embedding vector_cosine_ops);
    """)

    # Full-text search index — needed for hybrid search in Phase 2
    cur.execute("""
        ALTER TABLE filing_chunks
        ADD COLUMN IF NOT EXISTS fts tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_fts_idx
        ON filing_chunks USING gin(fts);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✓ Database ready")

def clear_ticker(ticker: str):
    """Remove existing chunks for a ticker before re-ingesting."""
    conn = psycopg2.connect(CONN_STR)
    cur  = conn.cursor()
    cur.execute("DELETE FROM filing_chunks WHERE ticker = %s", (ticker,))
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    if deleted:
        print(f"  Cleared {deleted} existing chunks for {ticker}")

# ── EDGAR fetching ────────────────────────────────────────────────────────────

def get_cik(ticker: str) -> tuple[str, str]:
    """Returns (zero-padded CIK, company name) for a ticker."""
    url  = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS).json()

    for entry in data.values():
        if entry["ticker"] == ticker.upper():
            return str(entry["cik_str"]).zfill(10), entry["title"]

    raise ValueError(f"Ticker {ticker} not found in EDGAR")

def get_latest_accession(cik: str, form_type: str) -> tuple[str, str]:
    """Returns (raw accession number, filing date) for the latest filing."""
    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()

    forms      = data["filings"]["recent"]["form"]
    dates      = data["filings"]["recent"]["filingDate"]
    accessions = data["filings"]["recent"]["accessionNumber"]

    for form, date, acc in zip(forms, dates, accessions):
        if form == form_type:
            return acc, date

    raise ValueError(f"No {form_type} found for CIK {cik}")

def get_primary_doc_url(cik: str, accession_raw: str, form_type: str) -> str:
    """Resolves the primary .htm document URL from the filing index."""
    acc_nodash = accession_raw.replace("-", "")
    cik_int    = int(cik)

    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{accession_raw}-index.json"
    )
    files = requests.get(index_url, headers=HEADERS).json()["directory"]["item"]

    def score(f):
        name  = f["name"].lower()
        ftype = f.get("type", "").upper()
        if ftype == form_type and name.endswith(".htm"):   return 0
        if name.endswith(".htm") and not name.startswith("ex") and "-" in name: return 1
        if name.endswith(".htm") and not name.startswith("ex"):                 return 2
        return 99

    candidates = sorted([f for f in files if f["name"].endswith(".htm")], key=score)

    if not candidates:
        raise ValueError(f"No .htm found in filing {accession_raw}")

    primary = candidates[0]["name"]
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{primary}"
    )

def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text

# ── Text processing ───────────────────────────────────────────────────────────

def clean_and_section(raw_html: str) -> list[dict]:
    soup  = BeautifulSoup(raw_html, "lxml")

    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    full_text = soup.get_text(separator="\n")
    lines     = [l.strip() for l in full_text.splitlines()]
    lines     = [l for l in lines if len(l) > 20]
    full_text = "\n".join(lines)

    pattern = re.compile(r'(ITEM\s+\d+[A-Z]?\.|Item\s+\d+[A-Z]?\.)', re.IGNORECASE)
    parts   = pattern.split(full_text)

    sections      = []
    current_label = "preamble"

    for part in parts:
        if pattern.match(part.strip()):
            current_label = part.strip().lower().replace(" ", "_").replace(".", "")
        elif len(part.strip()) > 100:
            sections.append({"section": current_label, "text": part.strip()})

    return sections

def chunk_sections(sections: list[dict], ticker: str, filing_date: str) -> list[dict]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda t: len(tokenizer.encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for section in sections:
        for i, text in enumerate(splitter.split_text(section["text"])):
            chunks.append({
                "ticker":       ticker,
                "filing_date":  filing_date,
                "section":      section["section"],
                "chunk_index":  i,
                "text":         text,
                "token_count":  len(tokenizer.encode(text)),
            })
    return chunks

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32,
                              show_progress_bar=True,
                              normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks

def store_chunks(chunks: list[dict]):
    conn = psycopg2.connect(CONN_STR)
    cur  = conn.cursor()

    rows = [
        (c["ticker"], c["filing_date"], c["section"],
         c["chunk_index"], c["text"], c["token_count"], c["embedding"])
        for c in chunks
    ]

    execute_values(
        cur,
        """INSERT INTO filing_chunks
           (ticker, filing_date, section, chunk_index, text, token_count, embedding)
           VALUES %s""",
        rows,
        template="(%s,%s,%s,%s,%s,%s,%s::vector)"
    )

    conn.commit()
    cur.close()
    conn.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading BGE embedding model...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    print("\nSetting up database...")
    setup_database()

    for ticker in TICKERS:
        print(f"\n{'─'*50}")
        print(f"Processing {ticker}...")

        try:
            clear_ticker(ticker)

            cik, company   = get_cik(ticker)
            print(f"  Company : {company}")

            accession, date = get_latest_accession(cik, FORM_TYPE)
            print(f"  Filing  : {FORM_TYPE} on {date}")

            doc_url        = get_primary_doc_url(cik, accession)
            raw_html       = fetch_html(doc_url)
            print(f"  Fetched : {len(raw_html):,} chars")

            sections       = clean_and_section(raw_html)
            print(f"  Sections: {len(sections)}")

            chunks         = chunk_sections(sections, ticker, date)
            print(f"  Chunks  : {len(chunks)}")

            chunks         = embed_chunks(chunks, model)
            store_chunks(chunks)
            print(f"  ✓ Stored {len(chunks)} chunks")

            # Be polite to SEC servers — don't hammer them
            time.sleep(1)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    print(f"\n{'─'*50}")
    print("Ingest complete. Running sanity check...\n")
    sanity_check()

def sanity_check():
    conn = psycopg2.connect(CONN_STR)
    cur  = conn.cursor()

    cur.execute("SELECT ticker, COUNT(*) FROM filing_chunks GROUP BY ticker ORDER BY ticker")
    rows = cur.fetchall()

    print("Chunks per company:")
    for ticker, count in rows:
        status = "✓" if count > 100 else "⚠ low"
        print(f"  {ticker}: {count:,} chunks {status}")

    cur.execute("SELECT COUNT(*) FROM filing_chunks WHERE embedding IS NULL")
    nulls = cur.fetchone()[0]
    print(f"\nMissing embeddings: {nulls} {'✓' if nulls == 0 else '⚠ re-run needed'}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()

