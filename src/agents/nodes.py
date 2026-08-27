"""
src/agents/nodes.py

Each function here is a LangGraph node — it receives the full state,
does one job, and returns a dict of the fields it wants to update.
LangGraph merges the returned dict back into the shared state.
"""

import os
import anthropic
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve, TONE_MODEL
from src.api.rag_chain import generate_answer, build_context_block
from src.observability.tracer import get_tracer

load_dotenv()
client = anthropic.Anthropic()
langfuse = get_tracer()

# ── Node 1: Router ────────────────────────────────────────────────────────────
def router_node(state: dict) -> dict:
    """
    Classifies the question so downstream agents can specialize.

    Why: a question about revenue numbers needs different handling
    than a question about management tone. Routing lets each agent
    focus on what it does best.
    """
    print(f"\n[Router] Classifying: '{state['question'][:60]}...'")
    span = langfuse.span(
        trace_id=state.get("trace_id"),
        name="router",
        input={"question": state["question"]}
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""Classify this financial research question into exactly one Reply with just the category word, nothing else.
            
            Categories:
            - quantitative: asks about numbers, revenue, margins, growth rates
            - qualitative: asks about risks, strategy, management language, narrative
            - comparison: asks to compare multiple companies or time periods
            
            Question: {state['question']}

            Category:"""
        }]
    )

    query_type = response.content[0].text.strip().lower()

    # Sanitize — default to qualitative if unexpected response
    if query_type not in ["quantitative", "qualitative", "comparison"]:
        query_type = "qualitative"
    
    span.end(output={"query_type": query_type})
    print(f"[Router] Type: {query_type}")
    return {"query_type": query_type}

# ── Node 2: Retrieval agent ───────────────────────────────────────────────────
def retrieval_node(state: dict) -> dict:
    """
    Runs hybrid search and returns the top chunks.
    Uses query_type to adjust retrieval strategy.
    """
    print(f"[Retrieval] Fetching chunks for {state['ticker']}...")
    span = langfuse.span(
        trace_id=state.get("trace_id"),
        name="retrieval",
        input={"question": state["question"], "ticker": state["ticker"]}
    )
    # Quantitative questions benefit from more chunks
    # (numbers often appear across multiple sections)
    top_k = 8 if state["query_type"] == "quantitative" else 5

    chunks = retrieve(
        query=state["question"],
        ticker=state["ticker"],
        top_k=top_k,
        form_type=state.get("form_type")
    )

    span.end(output={
        "chunks_found": len(chunks),
        "top_score":    chunks[0].get("rerank_score") if chunks else None
    })
    
    print(f"[Retrieval] Found {len(chunks)} chunks | "
          f"Top rerank score: {chunks[0].get('rerank_score', 'N/A') if chunks else 'none'}")

    return {"chunks": chunks}

# ── Node 3: Tone agent ────────────────────────────────────────────────────────
def tone_node(state: dict) -> dict:
    """
    Scores management language using finBERT-tone.
    A HuggingFace text-classification model trained specifically
    on financial text — more reliable than asking an LLM to score tone
    because it's deterministic, cheap, and doesn't hallucinate.

    Output labels: positive | negative | neutral
    Each with a probability score 0.0 → 1.0
    """
    print("[Tone] Scoring with finBERT...")
    
    span = langfuse.span(
        trace_id=state.get("trace_id"),
        name="tone_scoring",
        input={"chunks_count": len(state["chunks"])}
    )
    
    if not state["chunks"]:
        span.end(output={"tone_label": "neutral", "tone_score": 0.5})
        return {"tone_score": 0.5, "tone_label": "neutral"}

    # Score top 3 chunks individually then average
    # finBERT has a 512 token limit — truncate each chunk
    scores = {"positive": [], "negative": [], "neutral": []}

    for chunk in state["chunks"][:3]:
        text = chunk["text"][:512]
        results = TONE_MODEL(text)   # list of {label, score}
        for r in results:
            label = r["label"].lower()
            if label in scores:
                scores[label].append(r["score"])

    # Average across chunks
    avg = {
        label: sum(vals) / len(vals)
        for label, vals in scores.items() if vals
    }

    # Dominant label
    dominant_label = max(avg, key=avg.get)
    # Normalize to 0-1 where 1=positive, 0=negative
    tone_score = avg.get("positive", 0.5) - avg.get("negative", 0.0) + 0.5
    tone_score = round(min(max(tone_score, 0.0), 1.0), 4)

    span.end(output={"tone_label": dominant_label, "tone_score": tone_score})
    
    print(f"[Tone] {dominant_label} | score: {tone_score} | "
          f"pos={avg.get('positive',0):.2f} "
          f"neg={avg.get('negative',0):.2f} "
          f"neu={avg.get('neutral',0):.2f}")

    return {
        "tone_score": tone_score,
        "tone_label": dominant_label
    }

# ── Node 4: Synthesis agent ───────────────────────────────────────────────────

def synthesis_node(state: dict) -> dict:
    """
    Generates the final cited answer using retrieved chunks.
    Appends tone context to the answer.
    """
    print("[Synthesis] Generating answer...")

    span = langfuse.span(
        trace_id=state.get("trace_id"),
        name="synthesis",
        input={"question": state["question"], "chunks": len(state["chunks"])}
    )
    
    result = generate_answer(state["question"], state["chunks"])

    # Append tone signal to the answer
    tone_note = ""
    if state.get("tone_score") is not None:
        tone_note = (
            f"\n\n**Management Tone Analysis:** {state['tone_label'].capitalize()} "
            f"(score: {state['tone_score']:.2f}) — language in retrieved sections "
            f"appears {'confident and clear' if state['tone_score'] > 0.6 else 'hedged or uncertain'}."
        )
    span.end(output={
        "answer_preview":  result["answer"][:200],
        "input_tokens":    result["usage"].get("input_tokens"),
        "output_tokens":   result["usage"].get("output_tokens"),
        "cache_hit_tokens":result["usage"].get("cache_read_tokens"),
    })
    
    return {
        "answer":  result["answer"] + tone_note,
        "sources": result["sources"],
        "usage":   result["usage"],
        "verified": False   # verifier will check this next
    }

# ── Node 5: Verifier agent ────────────────────────────────────────────────────
def verifier_node(state: dict) -> dict:
    """
    Checks that the answer is grounded in the retrieved chunks.
    If confidence is low and retries remain, signals a retry.

    This is the node that makes your system self-correcting —
    the key differentiator from a basic RAG pipeline.
    """
    print("[Verifier] Checking answer grounding...")
    
    span = langfuse.span(
        trace_id=state.get("trace_id"),
        name="verifier",
        input={"question": state["question"], "chunks": len(state["chunks"])}
    )
    
    if not state.get("answer") or not state.get("chunks"):
        return {"verified": False}
    
    answer_to_verify = state['answer'].split("**Management Tone:**")[0].strip()

    context_preview = "\n".join([c["text"][:400] for c in state["chunks"][:5]])

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Is this answer fully supported by the provided context?
            Check if every factual claim in the answer can be found in the context.

            Context:
            {context_preview}

            Answer:
            {answer_to_verify[:500]}

            Reply with JSON only:
            {{"grounded": true/false, "confidence": 0.0-1.0, "issue": "one line or null"}}"""
        }]
    )

    import json
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    print(f"Verifier raw response {repr(raw)}")
    check = json.loads(raw.strip())
    
    print(f"[Verifier] Grounded: {check['grounded']} | "
          f"Confidence: {check['confidence']:.2f} | "
          f"Issue: {check['issue']}")

    print(f"[Verifier] Answer preview: {state['answer'][:300]}")
    print(f"[Verifier] Context preview: {context_preview[:300]}")
    print(f"[Verifier] Check result: {check}")
    # Mark verified if grounded and confidence is high enough
    verified = check["grounded"] and check["confidence"] >= 0.6
    span.end(output=check)
    
    return {"verified": verified }