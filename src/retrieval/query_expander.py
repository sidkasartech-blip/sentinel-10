# src/retrieval/query_expander.py
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

def expand_query(question: str) -> list[str]:
    """
    Generates multiple query variations to improve recall.
    
    Why: "What are Apple's AI risks?" might miss chunks that say
    "machine learning vulnerabilities" or "algorithmic bias exposure"
    — same concept, different words.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": f"""Generate 3 alternative search queries 
            for this financial research question. Each should capture the same intent 
            but use different terminology an SEC filing might use.

            Question: {question}

            Return JSON only:
            {{"queries": ["query1", "query2", "query3"]}}"""}]
        )

    import json, re
    raw = response.content[0].text.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    data  = json.loads(match.group())

    # Original + 3 expansions
    return [question] + data["queries"]