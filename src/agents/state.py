"""
src/agents/state.py

The shared state object that flows through every agent in the graph.
Think of it as a baton passed between runners in a relay race —
each agent picks it up, adds their contribution, passes it on.
"""

from typing import TypedDict, Optional

class AgentState(TypedDict):
    #Input
    question: str
    ticker: str

    #Router output
    query_type: Optional[str] # "quantitative" | "qualitative" | "comparison"

    # Retrieval agent output
    chunks:      list

    # Tone agent output
    tone_score:  Optional[float]  # 0.0 (negative) to 1.0 (positive)
    tone_label:  Optional[str]    # "positive" | "neutral" | "negative"

    # Verifier output
    verified:    bool
    retry_count: int

    # Synthesis output
    answer:      Optional[str]
    sources:     list
    usage:       dict