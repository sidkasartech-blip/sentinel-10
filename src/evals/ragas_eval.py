"""
src/evals/ragas_eval.py

Baseline RAGAS evaluation of your RAG pipeline.
Run with: python -m src.evals.ragas_eval

RAGAS measures 4 things:
  faithfulness       — does the answer stick to the retrieved context?
                       (catches hallucination)
  answer_relevancy   — does the answer actually address the question?
  context_precision  — are the retrieved chunks relevant to the question?
  context_recall     — did retrieval find all the info needed to answer?
"""

import json
import sys
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper

load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.retrieval.retriever import retrieve
from src.api.rag_chain import generate_answer


# ── Golden dataset ─────────────────────────────────────────────────────────────
# These are hand-written question/answer pairs you know are correct.
# "ground_truth" = the answer you'd expect from reading the actual filing.
# Start with 20 — enough for a meaningful baseline score.
# This is the most important artifact in your eval system.

GOLDEN_DATASET = [
    {
        "question":     "What are Apple's primary cybersecurity risk factors?",
        "ticker":       "AAPL",
        "ground_truth": "Apple faces cybersecurity risks including unauthorized access to its systems, theft of proprietary information, and disruption of services. The company stores sensitive customer and employee data that could be compromised. Third-party vendors also introduce cybersecurity vulnerabilities."
    },
    {
        "question":     "How does Apple describe its AI and machine learning strategy?",
        "ticker":       "AAPL",
        "ground_truth": "Apple integrates AI and machine learning across its products and services including Siri, camera systems, and the App Store. The company develops its own chips optimized for on-device machine learning to protect user privacy."
    },
    {
        "question":     "What does Microsoft identify as its key cloud computing risks?",
        "ticker":       "MSFT",
        "ground_truth": "Microsoft identifies risks including security breaches, service outages, and competition in cloud services. Data center capacity constraints and dependency on third-party infrastructure providers are also noted risks for Azure."
    },
    {
        "question":     "How does NVIDIA describe its dependence on Taiwan for chip manufacturing?",
        "ticker":       "NVDA",
        "ground_truth": "NVIDIA relies heavily on TSMC in Taiwan for manufacturing its GPUs. Geopolitical risks, natural disasters, or supply chain disruptions in Taiwan could materially impact NVIDIA's ability to deliver products."
    },
    {
        "question":     "What competition risks does Meta identify in its social media business?",
        "ticker":       "META",
        "ground_truth": "Meta faces competition from TikTok, YouTube, Snapchat, and other platforms for user attention and advertiser spending. The company notes that users may reduce engagement if competitors offer more compelling features."
    },
    {
        "question":     "How does Google describe its advertising revenue concentration risk?",
        "ticker":       "GOOGL",
        "ground_truth": "Google generates the majority of its revenue from advertising and acknowledges risks from advertiser budget reductions, competition from other ad platforms, and changes in user behavior that could reduce ad impressions."
    },
    {
        "question":     "What does Apple say about its supplier concentration risk?",
        "ticker":       "AAPL",
        "ground_truth": "Apple sources components from a limited number of suppliers and manufacturers, primarily in Asia. Disruption to any key supplier relationship could delay product launches and negatively impact revenue."
    },
    {
        "question":     "How does Microsoft describe its regulatory and compliance risks?",
        "ticker":       "MSFT",
        "ground_truth": "Microsoft faces regulatory scrutiny globally including antitrust investigations, data privacy regulations like GDPR, and government requirements that may restrict certain business practices or require changes to its products."
    },
    {
        "question":     "What are NVIDIA's risks related to export controls?",
        "ticker":       "NVDA",
        "ground_truth": "NVIDIA faces restrictions on exporting certain high-performance chips to China and other countries due to US export control regulations. These restrictions could significantly reduce revenue from affected markets."
    },
    {
        "question":     "How does Meta describe its risks from Apple's privacy changes?",
        "ticker":       "META",
        "ground_truth": "Meta describes Apple's App Tracking Transparency framework as a significant headwind that reduced its ability to target ads on iOS devices, negatively impacting advertising revenue and making it harder to measure campaign effectiveness."
    },
]


def run_evaluation(dataset: list[dict] = None, verbose: bool = True) -> dict:
    """
    Runs the full RAGAS evaluation pipeline.

    For each question in the dataset:
      1. Retrieve top chunks (your hybrid search pipeline)
      2. Generate answer (Claude with citations)
      3. Package into RAGAS format
      4. Score all 4 metrics
    """
    if dataset is None:
        dataset = GOLDEN_DATASET

    print(f"Running RAGAS eval on {len(dataset)} questions...\n")

    questions        = []
    answers          = []
    contexts         = []
    ground_truths    = []

    for i, item in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {item['ticker']}: {item['question'][:60]}...")

        # Step 1 — retrieve
        chunks = retrieve(
            query=item["question"],
            ticker=item["ticker"],
            top_k=5
        )

        # Step 2 — generate
        result = generate_answer(item["question"], chunks)

        # Step 3 — package for RAGAS
        # contexts must be a list of strings (the retrieved chunk texts)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in chunks])
        ground_truths.append(item["ground_truth"])

        if verbose:
            print(f"      Answer preview: {result['answer'][:100]}...")

    # ── Build HuggingFace Dataset (RAGAS expects this format) ─────────────────
    ragas_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # ── Run RAGAS scoring ──────────────────────────────────────────────────────
    print("\nScoring with RAGAS (this calls Claude for each metric)...")

    # Tell RAGAS to use Claude instead of OpenAI
    llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6", temperature=0))

    results = evaluate(
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("RAGAS BASELINE SCORES")
    print("="*50)

    scores = results.to_pandas()

    metrics = {
        "faithfulness":      scores["faithfulness"].mean(),
        "answer_relevancy":  scores["answer_relevancy"].mean(),
        "context_precision": scores["context_precision"].mean(),
        "context_recall":    scores["context_recall"].mean(),
    }

    for metric, score in metrics.items():
        bar    = "█" * int(score * 20)
        status = "✓" if score >= 0.7 else "⚠"
        print(f"  {status} {metric:<22} {score:.4f}  {bar}")

    print("="*50)
    print(f"  Overall average: {sum(metrics.values())/len(metrics):.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs("src/evals/results", exist_ok=True)

    output = {
        "scores":    metrics,
        "per_question": scores.to_dict(orient="records")
    }

    with open("src/evals/results/baseline.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n  Results saved to src/evals/results/baseline.json")
    print("\nWhat these scores mean:")
    print("  faithfulness      — are answers grounded in retrieved chunks? (anti-hallucination)")
    print("  answer_relevancy  — does the answer address the question asked?")
    print("  context_precision — are retrieved chunks actually relevant?")
    print("  context_recall    — did retrieval find all info needed to answer?")
    print("\nTarget: all metrics above 0.70 before moving to Phase 3.")

    return metrics


if __name__ == "__main__":
    run_evaluation()