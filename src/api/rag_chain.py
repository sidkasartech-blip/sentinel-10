"""
src/api/rag_chain.py

Takes retrieved chunks and generates a cited answer using Claude.
This is deliberately kept separate from retrieval — clean separation
of concerns makes each piece independently testable in Phase 2b.
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# ── Prompt engineering ────────────────────────────────────────────────────────
# This system prompt is the single most important text in your project.
# Every word is intentional — read the comments carefully.

SYSTEM_PROMPT = """You are a financial research analyst assistant specializing \
in SEC filings analysis.

Your job is to answer questions strictly based on the provided context \
from SEC filings. Follow these rules without exception:

1. CITATIONS: Every factual claim must cite its source as [1], [2], etc.
   matching the context numbers provided.

2. GROUNDING: Only use information from the provided context. If the \
   context does not contain enough information to answer, say:
   "The provided filings do not contain sufficient information to answer \
   this question."

3. ACCURACY: Never invent numbers, dates, or facts. If you are uncertain, \
   say so explicitly.

4. FORMAT: Structure your answer as:
   - A direct answer in 2-3 sentences
   - Supporting detail with citations
   - Any important caveats or limitations

5. TONE: Professional, precise, and objective. This is financial research."""

def build_context_block(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a numbered context block.
    The numbers [1], [2] etc. map directly to citations in the answer.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] SOURCE: {chunk['ticker']} | "
            f"Section: {chunk['section']} | "
            f"Filed: {chunk['filing_date']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(lines)

def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Sends question + retrieved context to Claude and returns
    a structured response with the answer and source metadata.
    """
    if not chunks:
        return {
            "answer": "No relevant information found in the filings.",
            "sources": [],
            "usage": {}
        }

    context_block = build_context_block(chunks)

    # Why cache_control on the context?
    # A 10-K has ~150K tokens. If you ask 5 questions about AAPL in a session,
    # you'd pay to process those 150K tokens 5 times without caching.
    # With prompt caching, Claude caches the context after the first call —
    # subsequent calls with the same context cost ~90% less on input tokens.
    # This is a real LLMOps concern — discuss it in interviews.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONTEXT FROM SEC FILINGS:\n\n{context_block}",
                        # Cache the context block — expensive to re-process
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": f"QUESTION: {question}"
                    }
                ]
            }
        ]
    )

    answer = response.content[0].text

    # Attach rerank scores to sources for transparency
    sources = [
        {
            "citation_number": i + 1,
            "ticker":          c["ticker"],
            "section":         c["section"],
            "filing_date":     c["filing_date"],
            "text":            c["text"],
            "rerank_score":    c.get("rerank_score"),
            "rrf_score":       c.get("rrf_score"),
        }
        for i, c in enumerate(chunks)
    ]

    return {
        "answer":  answer,
        "sources": sources,
        "usage": {
            "input_tokens":          response.usage.input_tokens,
            "output_tokens":         response.usage.output_tokens,
            "cache_read_tokens":     getattr(response.usage, "cache_read_input_tokens", 0),
            "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        }
    }