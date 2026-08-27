"""
src/ingest/migrate.py

Adds missing columns/indexes to an existing database without
touching the data. This is called a "migration" — the production
pattern for evolving your schema safely.

Run with: python -m src.ingest.migrate
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def run_migration():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur  = conn.cursor()

    print("Running migrations on filing_chunks...\n")

    # ── Migration 1: Add the fts column ───────────────────────────────────────
    # GENERATED ALWAYS AS (...) STORED means Postgres computes and stores
    # this value automatically whenever a row is inserted or updated.
    # You never write to it manually — it stays in sync with `text` for free.
    print("Step 1/2 — Adding fts column (this may take 1-2 min on large tables)...")
    cur.execute("""
        ALTER TABLE filing_chunks
        ADD COLUMN IF NOT EXISTS fts tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
    """)
    print("  ✓ fts column added")

    # ── Migration 2: Add the GIN index for fast FTS queries ───────────────────
    # GIN = Generalized Inverted Index — the right index type for tsvector.
    # Without this, FTS falls back to a full table scan (very slow at scale).
    print("Step 2/2 — Creating GIN index on fts column...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_fts_idx
        ON filing_chunks USING gin(fts);
    """)
    print("  ✓ GIN index created")

    conn.commit()

    # ── Verify ────────────────────────────────────────────────────────────────
    print("\nVerifying migration...")

    cur.execute("""
        SELECT COUNT(*) FROM filing_chunks WHERE fts IS NOT NULL;
    """)
    populated = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM filing_chunks;")
    total = cur.fetchone()[0]

    print(f"  Rows total        : {total:,}")
    print(f"  Rows with fts     : {populated:,}")

    if populated == total:
        print("  ✓ Migration complete — all rows have fts populated")
    else:
        print(f"  ⚠ {total - populated} rows missing fts — re-run if needed")

    # Quick FTS sanity check
    cur.execute("""
        SELECT ticker, LEFT(text, 80)
        FROM filing_chunks
        WHERE fts @@ plainto_tsquery('english', 'cybersecurity risk')
        LIMIT 3;
    """)
    rows = cur.fetchall()
    print(f"\n  FTS smoke test — 'cybersecurity risk' matches: {len(rows)} rows")
    for row in rows:
        print(f"    [{row[0]}] {row[1]}...")

    cur.close()
    conn.close()
    print("\n✓ All done. Your API should work now.")


if __name__ == "__main__":
    run_migration()