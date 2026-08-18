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
            id             SERIAL PRIMARY KEY,
            ticker         TEXT    NOT NULL,
            filing_date    TEXT    NOT NULL,
            section        TEXT    NOT NULL,
            section_title  TEXT    DEFAULT '',
            chunk_index    INTEGER NOT NULL,
            text           TEXT    NOT NULL,
            token_count    INTEGER,
            embedding      vector(768),
            created_at     TIMESTAMP DEFAULT NOW()
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

def get_latest_accession(cik: str, form_type: str) -> tuple[str, str, str]:
    """Returns (raw accession number, filing date) for the latest filing."""
    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()

    forms      = data["filings"]["recent"]["form"]
    dates      = data["filings"]["recent"]["filingDate"]
    accessions = data["filings"]["recent"]["accessionNumber"]
    primary_document = data["filings"]["recent"]["primaryDocument"]

    for form, date, acc, document in zip(forms, dates, accessions, primary_document):
        if form == form_type:
            return acc, date, document

    raise ValueError(f"No {form_type} found for CIK {cik}")

def get_primary_doc_url(cik: str, accession_raw: str, form_type: str, document_name: str) -> str:
    """Resolves the primary .htm document URL from the filing index."""
    acc_nodash = accession_raw.replace("-", "")
    cik_int    = int(cik)

    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{document_name}"
    print(f"Trying document URL: {doc_url}")
    return doc_url

def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text

# ── Text processing ───────────────────────────────────────────────────────────

def clean_and_section(raw_html: str) -> list[dict]:
    """
    Extracts human-readable sections from an iXBRL 10-K filing.

    Key insight from diagnostic: modern 10-Ks have TWO sets of Item markers:
    1. Table of contents (lines 136-200) — "Item 1.", "Item 1A." etc.
       These are close together (3 lines apart) with NO content between them.
    2. Real section headers (line 470+) — "Item 1C.\xa0\xa0\xa0\xa0Cybersecurity"
       These have non-breaking spaces (\xa0) before the section title,
       and are followed by thousands of words of real content.

    Strategy: find ALL Item markers, then skip the ToC cluster by only
    keeping markers that have substantial content (>500 chars) after them.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # ── Strip noise ────────────────────────────────────────────────────────────
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["ix:header", "header"]):
        tag.decompose()
    for tag in soup.find_all("ix:hidden"):
        tag.decompose()
    for tag in soup.find_all(style=True):
        style = tag.get("style", "").lower().replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()

    # ── Extract and clean text ─────────────────────────────────────────────────
    full_text = soup.get_text(separator="\n")

    lines = []
    for line in full_text.splitlines():
        line = line.strip()
        if not line or (len(line) < 25 and not re.match(r'Item\s+\d+', line, re.IGNORECASE)): continue
        if line.startswith("http"):                   continue
        if re.match(r'^[\w\-]+:[\w\-]+', line):      continue  # xbrl:tags
        if line.count(":") > 5 and len(line) < 300:  continue  # tag lists
        lines.append(line)

    full_text = "\n".join(lines)

    # ── Find ALL Item markers ──────────────────────────────────────────────────
    # Matches both:
    #   "Item 1A."                  (ToC style — no title)
    #   "Item 1C.\xa0\xa0Cybersecurity"  (real header — has title after \xa0)
    # \xa0 is the non-breaking space character you saw in the diagnostic
    ITEM_PATTERN = re.compile(
        r'(?:^|\n)(Item\s+\d+[A-Za-z]?\.?\s*(?:[\xa0\s]+\w+.*)?)',
        re.MULTILINE
    )

    all_matches = list(ITEM_PATTERN.finditer(full_text))
    print(f"  Total Item markers found: {len(all_matches)}")

    # ── Separate ToC entries from real headers ─────────────────────────────────
    # Key observation: ToC entries are clustered close together.
    # Real headers have at least 500 characters of content before the next header.
    # We find the "jump" — where the gap between consecutive markers gets large.

    real_matches = []
    for i, match in enumerate(all_matches):
        current_pos = match.start()
        next_pos    = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(full_text)
        content_len = next_pos - current_pos

        if content_len > 500:
            # This marker has substantial content after it — it's a real header
            real_matches.append(match)

    print(f"  Real section headers    : {len(real_matches)}")

    if len(real_matches) < 3:
        print("  ⚠ Too few real headers found — using paragraph fallback")
        return _paragraph_fallback(full_text)

    # ── Build sections from real headers ──────────────────────────────────────
    sections = []

    for i, match in enumerate(real_matches):
        # Parse the item label — normalize \xa0 to regular space first
        header_text = match.group(1).replace("\xa0", " ").strip()

        # Extract item number: "Item 1A. Risk Factors" → "item_1a"
        item_match = re.match(r'Item\s+(\d+[A-Za-z]?)\.?', header_text, re.IGNORECASE)
        if item_match:
            item_num = item_match.group(1).lower()
            label    = f"item_{item_num}"
        else:
            label    = f"section_{i}"

        # Extract title if present: "Item 1A.\xa0\xa0Risk Factors" → "Risk Factors"
        title_match = re.match(
            r'Item\s+\d+[A-Za-z]?\.?[\xa0\s]+(.+)', header_text, re.IGNORECASE
        )
        section_title = title_match.group(1).strip() if title_match else ""

        # Content = everything from end of this header to start of next
        content_start = match.end()
        content_end   = real_matches[i + 1].start() if i + 1 < len(real_matches) else len(full_text)
        section_text  = full_text[content_start:content_end].strip()

        # Skip XBRL-heavy sections
        xbrl_hits = section_text.count("us-gaap") + section_text.count("fasb.org")
        if xbrl_hits > 10 and len(section_text.split()) < 500:
            print(f"  Skipping {label} — XBRL heavy")
            continue

        if len(section_text) < 200:
            continue

        sections.append({
            "section": label,
            "title":   section_title,
            "text":    section_text
        })

        print(f"  [{label}] {section_title[:40]:<40} {len(section_text):>8,} chars")

    return sections

def _paragraph_fallback(full_text: str) -> list[dict]:
    """
    Fallback for heavily styled filings where Item markers aren't detectable.
    Groups paragraphs into chunks of ~3 paragraphs each.
    """
    paragraphs = [
        p.strip() for p in full_text.split("\n\n")
        if len(p.strip()) > 150
    ]
    sections = []
    group_size = 3
    for i in range(0, len(paragraphs), group_size):
        group = paragraphs[i:i + group_size]
        sections.append({
            "section": f"para_group_{i // group_size}",
            "title":   "",
            "text":    "\n\n".join(group)
        })
    print(f"  Fallback: {len(sections)} groups from {len(paragraphs)} paragraphs")
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
                "ticker":        ticker,
                "filing_date":   filing_date,
                "section":       section["section"],
                "section_title": section.get("title", ""),
                "chunk_index":   i,
                "text":          text,
                "token_count":   len(tokenizer.encode(text)),
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
         c["chunk_index"], c["text"], c["token_count"], c["embedding"], c["section_title"])
        for c in chunks
    ]

    execute_values(
        cur,
        """INSERT INTO filing_chunks
           (ticker, filing_date, section, chunk_index, text, token_count, embedding,section_title)
           VALUES %s""",
        rows,
        template="(%s,%s,%s,%s,%s,%s,%s::vector,%s)"
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

            accession, date, document_name = get_latest_accession(cik, FORM_TYPE)
            print(f"  Filing  : {FORM_TYPE} on {date}")

            doc_url        = get_primary_doc_url(cik, accession, FORM_TYPE, document_name)
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

