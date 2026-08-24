"""
src/evals/eval.py

Evaluation of Sentinel-10 RAG pipeline using DeepEval.
Run with: python -m src.evals.eval

Metrics:
  Faithfulness       — is the answer grounded in retrieved context? (no hallucination)
  Answer Relevancy   — does the answer actually address the question?
  Contextual Recall  — did retrieval surface all info needed to answer?
  Contextual Precision — are retrieved chunks relevant to the question?
"""

import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

from src.retrieval.retriever import retrieve
from src.api.rag_chain import generate_answer
from src.classes.claudeEvalMode import ClaudeEvalModel

# ── Golden dataset ─────────────────────────────────────────────────────────────

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
    {
        "question":     "What are Apple's risks related to international operations?",
        "ticker":       "AAPL",
        "ground_truth": "Apple faces risks from foreign currency fluctuations, trade tariffs, geopolitical tensions, and differing regulatory requirements across international markets that could impact revenue and operations."
    },
    {
        "question":     "How does NVIDIA describe risks from customer concentration?",
        "ticker":       "NVDA",
        "ground_truth": "NVIDIA derives significant revenue from a small number of large customers. Loss of any key customer or reduction in orders could materially impact revenue and financial results."
    },
    {
        "question":     "What does Microsoft say about its AI investment risks?",
        "ticker":       "MSFT",
        "ground_truth": "Microsoft acknowledges risks from heavy AI infrastructure investment including uncertain returns, high capital expenditure for data centers, and the possibility that AI products may not achieve expected adoption or revenue."
    },
    {
        "question":     "How does Google describe competition in the AI search market?",
        "ticker":       "GOOGL",
        "ground_truth": "Google faces increasing competition in search from AI-powered alternatives and acknowledges that its dominant search market position could be challenged by new AI-native search experiences from competitors."
    },
    {
        "question":     "What liquidity and capital resources does Apple report?",
        "ticker":       "AAPL",
        "ground_truth": "Apple maintains significant cash and marketable securities and generates strong operating cash flow. The company uses cash for share repurchases, dividends, and capital expenditures."
    },
    {
        "question":     "How does Meta describe its Reality Labs losses?",
        "ticker":       "META",
        "ground_truth": "Meta's Reality Labs segment continues to generate significant operating losses as the company invests heavily in metaverse and augmented reality development with uncertain timelines to profitability."
    },
    {
        "question":     "What does NVIDIA say about data center revenue growth?",
        "ticker":       "NVDA",
        "ground_truth": "NVIDIA reports strong data center revenue growth driven by demand for its GPUs for AI training and inference workloads from cloud providers and enterprises."
    },
    {
        "question":     "How does Microsoft describe its gaming business risks?",
        "ticker":       "MSFT",
        "ground_truth": "Microsoft faces risks in gaming including competition from Sony and Nintendo, dependency on successful game launches, and integration risks from its Activision Blizzard acquisition."
    },
    {
        "question":     "What environmental and climate risks does Apple identify?",
        "ticker":       "AAPL",
        "ground_truth": "Apple identifies climate change as a risk that could disrupt its supply chain and operations. The company has committed to carbon neutrality and faces risks if it fails to meet sustainability targets."
    },
    {
        "question":     "How does Google describe its cloud segment growth strategy?",
        "ticker":       "GOOGL",
        "ground_truth": "Google Cloud focuses on enterprise customers and AI-powered services to grow market share against AWS and Azure. The segment has reached profitability and continues to invest in infrastructure and AI capabilities."
    },
]


# ── Build test cases ───────────────────────────────────────────────────────────

def build_test_cases(dataset: list[dict]) -> list[LLMTestCase]:
    test_cases = []

    for i, item in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {item['ticker']}: {item['question'][:55]}...")

        # Run your pipeline
        chunks = retrieve(item["question"], item["ticker"], top_k=5)
        result = generate_answer(item["question"], chunks)

        # DeepEval LLMTestCase — maps directly to your pipeline outputs
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item["ground_truth"],
            retrieval_context=[c["text"] for c in chunks],
        )
        test_cases.append(test_case)

    return test_cases


# ── Run evaluation ─────────────────────────────────────────────────────────────

def run_evaluation():
    print("Building test cases — running full RAG pipeline...\n")
    test_cases = build_test_cases(GOLDEN_DATASET[:2])
    claude_eval = ClaudeEvalModel()

    # Define metrics — threshold=0.5 means score must be >= 0.5 to pass
    # Start at 0.5 for baseline — raise to 0.7 once you improve the pipeline
    metrics = [
        FaithfulnessMetric(threshold=0.5,       model=claude_eval, include_reason=True),
        AnswerRelevancyMetric(threshold=0.5,     model=claude_eval, include_reason=True),
        ContextualRecallMetric(threshold=0.5,    model=claude_eval, include_reason=True),
        ContextualPrecisionMetric(threshold=0.5, model=claude_eval, include_reason=True),
    ]

    print("\nScoring with DeepEval...\n")
    results = evaluate(test_cases=test_cases, metrics=metrics)

    # ── Aggregate scores ───────────────────────────────────────────────────────
    scores = {
        "faithfulness":        [],
        "answer_relevancy":    [],
        "contextual_recall":   [],
        "contextual_precision":[],
    }

    for tc in results.test_results:
        for metric in tc.metrics_data:
            name = metric.name.lower().replace(" ", "_")
            if name in scores:
                scores[name].append(metric.score)

    print("\n" + "=" * 50)
    print("DEEPEVAL BASELINE SCORES")
    print("=" * 50)

    summary = {}
    for metric_name, metric_scores in scores.items():
        if not metric_scores:
            continue
        avg    = sum(metric_scores) / len(metric_scores)
        bar    = "█" * int(avg * 20)
        status = "✓" if avg >= 0.5 else "⚠"
        print(f"  {status} {metric_name:<26} {avg:.4f}  {bar}")
        summary[metric_name] = round(avg, 4)

    overall = sum(summary.values()) / len(summary) if summary else 0
    print("=" * 50)
    print(f"  Overall average: {overall:.4f}")
    print("\n  Target: all metrics above 0.70 before Phase 3.")

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs("src/evals/results", exist_ok=True)
    with open("src/evals/results/baseline.json", "w") as f:
        json.dump({"scores": summary, "overall": overall}, f, indent=2)

    print("  Saved → src/evals/results/baseline.json")
    return summary


if __name__ == "__main__":
    run_evaluation()