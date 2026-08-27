"""
src/agents/graph.py

Wires the agents into a LangGraph state machine.

Graph topology:
  router → [retrieval, tone] (parallel) → synthesis → verifier
                                               ↑
retry loop ──────┘ (if not verified and retries < 2)
"""

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from src.agents.state import AgentState
from src.agents.nodes import (
    router_node,
    retrieval_node,
    tone_node,
    synthesis_node,
    verifier_node,
)
from src.observability.tracer import get_tracer
import uuid

def dispatch_parallel(state: dict) -> list:
    """
    Called after router. Returns a list of Send objects —
    each one launches a node in parallel with its own state slice.

    Send(node_name, state) means: run this node right now
    with this state, don't wait for the other Send to finish.
    """
    return [
        Send("retrieval", state),
        Send("tone",      state),
    ]

def should_retry(state: dict) -> str:
    """
    Conditional edge — decides what happens after verification.

    If answer is verified → END
    If not verified and retries < 2 → go back to retrieval for another attempt
    If out of retries → END anyway (best effort answer)

    This is what makes the graph cyclic — the retry loop.
    Without this, it's just a linear chain.
    """
    if state.get("verified"):
        print("[Graph] Verified ✓ — sending answer")
        return "end"

    retry_count = state.get("retry_count", 0)
    if retry_count < 1:
        print(f"[Graph] Not verified — retry {retry_count + 1}/2")
        return "retry"

    print("[Graph] Max retries reached — sending best-effort answer")
    return "end"


def build_graph():
    """
    Constructs and compiles the LangGraph state machine.
    Call this once at startup — compiled graph is reusable.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    graph.add_node("router",    router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("tone",      tone_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("verifier",  verifier_node)

    # ── Add a retry-aware retrieval wrapper ────────────────────────────────────
    # On retry, we bump the retry count before going back to retrieval
    def increment_retry(state: dict) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}

    graph.add_node("increment_retry", increment_retry)

    # ── Define edges ───────────────────────────────────────────────────────────
    # Entry point
    graph.set_entry_point("router")

    # Router fans out to retrieval + tone in parallel
    # graph.add_conditional_edges(
    #     "router",
    #     dispatch_parallel,
    #     # Tell LangGraph which nodes dispatch_parallel can Send to
    #     ["retrieval", "tone"]
    # )
    
    # Both parallel nodes feed into synthesis
    # LangGraph waits for ALL incoming edges before running synthesis
    graph.add_edge("router",    "retrieval")
    graph.add_edge("retrieval", "tone")       # ← tone gets chunks now
    graph.add_edge("tone",      "synthesis")
    graph.add_edge("synthesis", "verifier")


    # Verifier → conditional: end or retry
    graph.add_conditional_edges(
        "verifier",
        should_retry,
        {
            "end":   END,
            "retry": "increment_retry"
        }
    )

    # Retry loop → back to retrieval
    graph.add_edge("increment_retry", "retrieval")

    # ── Compile ────────────────────────────────────────────────────────────────
    return graph.compile()


# Compile once at module level — reused across all requests
compiled_graph = build_graph()


def run_agent(question: str, ticker: str, form_type: str = None) -> dict:
    """
    Main entry point. Runs the full agent graph and returns the result.
    Called by the FastAPI endpoint.
    """
    langfuse  = get_tracer()
    trace_id  = str(uuid.uuid4())

    # Create top-level trace for this query
    trace = langfuse.trace(
        id=trace_id,
        name="sentinel-10-query",
        input={"question": question, "ticker": ticker}
    )
    initial_state = {
        "question":   question,
        "ticker":     ticker,
        "form_type": form_type, 
        "trace_id": trace_id,
        "query_type": None,
        "chunks":     [],
        "tone_score": None,
        "tone_label": None,
        "answer":     None,
        "sources":    [],
        "usage":      {},
        "verified":   False,
        "retry_count": 0,
    }

    print(f"\n{'='*55}")
    print(f"SENTINEL-10 AGENT | {ticker} | {question[:50]}...")
    print(f"{'='*55}")

    final_state = compiled_graph.invoke(initial_state)

    # Close the top-level trace with the final answer
    trace.update(
        output={
            "answer":      final_state["answer"][:300],
            "verified":    final_state["verified"],
            "retry_count": final_state["retry_count"],
            "tone_label":  final_state["tone_label"],
        }
    )
    langfuse.flush()   # ensure trace is sent before response returns
    
    print(f"\n{'='*55}")
    print(f"DONE | Verified: {final_state['verified']} | "
          f"Retries: {final_state['retry_count']}")
    print(f"{'='*55}\n")

    return {
        "question":   final_state["question"],
        "ticker":     final_state["ticker"],
        "query_type": final_state["query_type"],
        "answer":     final_state["answer"],
        "sources":    final_state["sources"],
        "tone_score": final_state["tone_score"],
        "tone_label": final_state["tone_label"],
        "verified":   final_state["verified"],
        "retry_count":final_state["retry_count"],
        "usage":      final_state["usage"],
    }