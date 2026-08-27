from sentence_transformers import SentenceTransformer
import anthropic

EMBED_MODEL   = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = anthropic.Anthropic()

def hyde_embed(question: str) -> list[float]:
    """
    HyDE: generate a fake answer, embed it, use that for vector search.
    
    Why: "What are Apple's cybersecurity risks?" embeds as a question.
    But your chunks contain answers. The embedding distance between
    a question and an answer is naturally larger than answer-to-answer.
    HyDE closes that gap.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": f"""Write a 2-sentence passage that would 
appear in an SEC 10-K filing answering this question. 
Use formal SEC filing language.

Question: {question}

Passage:"""}]
    )

    hypothetical_answer = response.content[0].text.strip()
    print(f"[HyDE] Hypothetical: {hypothetical_answer[:100]}...")

    # Embed the hypothetical answer, not the question
    return EMBED_MODEL.encode(
        hypothetical_answer,
        normalize_embeddings=True
    ).tolist()