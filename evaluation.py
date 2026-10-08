"""Evaluation module for Adaptive Agentic RAG.

Provides:
1. Ground-truth evaluation dataset loading (data/evaluation.json).
2. Creation and indexing of standard sample benchmark PDF.
3. Metric calculations:
   - Retrieval: Precision@K, Recall@K, Hit Rate@K
   - Generation: Faithfulness, Answer Relevance, Ground Truth Correctness
4. Side-by-side Baseline RAG vs Agentic RAG comparison runner.
5. Overall score and letter grading (A+, A, B, C, D, F).
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
import re
import fitz  # PyMuPDF

from config import EVAL_DATASET_PATH, DATA_DIR, TOP_K
from vector_store import VectorStore, RetrievedChunk
from ingestion import ingest_pdf, DocumentChunk
from embeddings import get_embedding
from agent import run_baseline_rag, run_agentic_rag

logger = logging.getLogger(__name__)

SAMPLE_PDF_PATH = DATA_DIR / "sample_corporate_report.pdf"


# -----------------------------------------------------------------------------
# 1. Benchmark Document Generator
# -----------------------------------------------------------------------------

def generate_sample_benchmark_pdf(file_path: Path = SAMPLE_PDF_PATH) -> Path:
    """Generate a realistic 4-page corporate report PDF for evaluation.

    Ensures the ground-truth evaluation questions in data/evaluation.json have
    verified, authentic source content.
    """
    if file_path.exists() and file_path.stat().st_size > 0:
        return file_path

    doc = fitz.open()

    # Page 1: Financial Overview & Leadership
    p1 = doc.new_page(width=595, height=842)
    t1 = (
        "ACME ENTERPRISE SOLUTIONS - ANNUAL OPERATIONAL REPORT 2024\n\n"
        "Executive Leadership & Financial Highlights:\n"
        "Elena Rostova has served as Chief Executive Officer since January 2022, leading global expansion.\n"
        "The company's total revenue in 2024 was $120 million, representing a 15% increase year-over-year.\n"
        "The operating margin reached 22% in 2024, improving from 18% in the previous year.\n\n"
        "Regional Revenue Performance:\n"
        "North American revenue surged 28% driven by enterprise software adoptions and customer renewals.\n"
        "European profitability contracted by 6% due to port logistics congestion and supplier delays.\n"
        "The 2023 restructuring plan reduced real estate overhead by $14M, which expanded 2024 margins."
    )
    p1.insert_text(fitz.Point(50, 72), t1, fontsize=11)

    # Page 2: Operations, R&D, and Human Resources
    p2 = doc.new_page(width=595, height=842)
    t2 = (
        "HUMAN CAPITAL AND RESEARCH INVESTMENTS\n\n"
        "Workforce Overview:\n"
        "As of December 31, 2024, the company employed 1,450 full-time team members across 8 global offices.\n"
        "Employee retention remained stable at 91% following flexible workplace initiatives.\n\n"
        "Research and Innovation:\n"
        "Research and development expenditure was $28 million, accounting for 23.3% of total revenue.\n"
        "Net revenue retention rose from 104% to 118% following the AI automation rollout, which reduced\n"
        "customer support ticket resolution costs by 40%.\n"
        "Enterprise customer acquisition cost (CAC) declined 12% to $14,000 with a payback period of 9 months."
    )
    p2.insert_text(fitz.Point(50, 72), t2, fontsize=11)

    # Page 3: Technology, Sustainability, and Infrastructure
    p3 = doc.new_page(width=595, height=842)
    t3 = (
        "TECHNOLOGY INFRASTRUCTURE & SUSTAINABILITY TARGETS\n\n"
        "Cloud Architecture:\n"
        "The primary cloud infrastructure is hosted on Google Cloud Platform with secondary disaster\n"
        "recovery on AWS, maintaining 99.99% uptime across all software services.\n"
        "Free cash flow increased to $35 million in fiscal year 2024.\n\n"
        "Divisional Growth Comparison:\n"
        "Cloud Software grew at 34% due to SaaS subscriptions, whereas Hardware declined by 4% due to\n"
        "semiconductor shortages and component inflation.\n"
        "The company committed to achieving net-zero carbon operations by 2030."
    )
    p3.insert_text(fitz.Point(50, 72), t3, fontsize=11)

    # Page 4: Risk Factors, Capital Structure, and Governance
    p4 = doc.new_page(width=595, height=842)
    t4 = (
        "RISK MANAGEMENT, DEBT OBLIGATIONS & GOVERNANCE\n\n"
        "Identified Enterprise Risks:\n"
        "The core identified risks include foreign exchange volatility, persistent semiconductor supply\n"
        "bottlenecks, regulatory AI compliance shifts, and rising talent acquisition costs.\n"
        "Competitors have increased price discounting in the small-business segment, prompting the company\n"
        "to focus on high-margin enterprise accounts with custom SLA guarantees.\n\n"
        "Capital Structure & Distributions:\n"
        "The company has $15 million in senior secured revolving credit notes maturing in November 2025.\n"
        "The Board declared a quarterly dividend of $0.15 per share, representing a 25% payout ratio.\n"
        "Invested $6M in Zero-Trust architecture and SOC 2 Type II audits, directly satisfying EU GDPR."
    )
    p4.insert_text(fitz.Point(50, 72), t4, fontsize=11)

    doc.save(str(file_path))
    doc.close()
    logger.info(f"Generated sample benchmark PDF at: {file_path}")
    return file_path


def load_evaluation_dataset(dataset_path: Path = EVAL_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load the ground-truth benchmark questions from JSON."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# 2. Metric Computation Functions
# -----------------------------------------------------------------------------

def calculate_precision_at_k(
    retrieved_chunks: List[RetrievedChunk],
    relevant_pages: List[int],
    k: int = TOP_K,
) -> float:
    """Calculate Precision@K: fraction of retrieved chunks belonging to ground-truth relevant pages.

    Precision@K = (# of retrieved chunks from relevant pages) / min(k, # retrieved)
    """
    if not relevant_pages:
        # Out-of-domain questions have 0 relevant pages in the document
        return 1.0 if not retrieved_chunks else 0.0

    if not retrieved_chunks:
        return 0.0

    eval_chunks = retrieved_chunks[:k]
    hits = sum(1 for c in eval_chunks if c.page_number in relevant_pages)
    return hits / len(eval_chunks)


def calculate_recall_at_k(
    retrieved_chunks: List[RetrievedChunk],
    relevant_pages: List[int],
    k: int = TOP_K,
) -> float:
    """Calculate Recall@K: fraction of expected relevant pages represented in top-K chunks.

    Recall@K = (# of distinct relevant pages retrieved) / (# total relevant pages)
    """
    if not relevant_pages:
        return 1.0 if not retrieved_chunks else 0.0

    eval_chunks = retrieved_chunks[:k]
    retrieved_page_set = set(c.page_number for c in eval_chunks)
    found_pages = set(relevant_pages).intersection(retrieved_page_set)

    return len(found_pages) / len(relevant_pages)


def calculate_hit_rate_at_k(
    retrieved_chunks: List[RetrievedChunk],
    relevant_pages: List[int],
    k: int = TOP_K,
) -> float:
    """Calculate Hit Rate@K: 1.0 if at least one relevant page was retrieved, else 0.0."""
    if not relevant_pages:
        return 1.0

    eval_chunks = retrieved_chunks[:k]
    for c in eval_chunks:
        if c.page_number in relevant_pages:
            return 1.0
    return 0.0


def calculate_faithfulness(
    answer: str,
    retrieved_chunks: List[RetrievedChunk],
    is_not_present: bool = False,
) -> float:
    """Evaluate faithfulness: whether the answer is strictly supported by context without hallucination.

    - For NOT_PRESENT questions: 1.0 if answer correctly declares insufficient evidence, 0.0 if hallucinated.
    - For factual questions: evaluates whether key claims/entities match retrieved context.
    """
    ans_lower = answer.lower()

    if is_not_present:
        if (
            "sufficient evidence" in ans_lower
            or "insufficient" in ans_lower
            or "not enough information" in ans_lower
            or "not present" in ans_lower
        ):
            return 1.0
        return 0.0

    if not retrieved_chunks:
        return 0.0

    # Aggregate retrieved context text
    context_text = " ".join([c.text.lower() for c in retrieved_chunks])

    # Extract non-trivial words (4+ characters) from the answer
    answer_words = re.findall(r"\b[a-z0-9]{4,}\b", ans_lower)
    if not answer_words:
        return 0.5

    # Check fraction of answer words present in the retrieved context
    grounded_words = sum(1 for w in answer_words if w in context_text)
    faithfulness_ratio = grounded_words / len(answer_words)

    # Scale gracefully
    return min(1.0, max(0.0, faithfulness_ratio * 1.15))


def calculate_answer_correctness(
    answer: str,
    ground_truth: str,
    relevant_keywords: List[str],
    is_not_present: bool = False,
) -> float:
    """Evaluate answer correctness against ground truth.

    - For NOT_PRESENT questions: checks for insufficient evidence declaration.
    - For normal questions: calculates keyword recall and token overlap against ground truth.
    """
    ans_lower = answer.lower()
    gt_lower = ground_truth.lower()

    if is_not_present:
        if "sufficient evidence" in ans_lower or "insufficient" in ans_lower or "not enough information" in ans_lower:
            return 1.0
        return 0.0

    # Check keyword recall
    if relevant_keywords:
        kw_hits = sum(1 for kw in relevant_keywords if kw.lower() in ans_lower)
        kw_score = kw_hits / len(relevant_keywords)
    else:
        kw_score = 1.0

    # Token overlap (Jaccard on words)
    words_ans = set(re.findall(r"\w+", ans_lower))
    words_gt = set(re.findall(r"\w+", gt_lower))
    if not words_gt:
        return kw_score

    intersection = words_ans.intersection(words_gt)
    token_score = len(intersection) / len(words_gt)

    return round((0.6 * kw_score) + (0.4 * token_score), 4)


def calculate_answer_relevance(
    query: str,
    answer: str,
) -> float:
    """Evaluate answer relevance: how well the answer addresses the question."""
    if not answer or not answer.strip():
        return 0.0

    ans_lower = answer.lower()
    q_words = [w for w in re.findall(r"\b\w{3,}\b", query.lower()) if w not in ["what", "when", "where", "which", "how", "the", "and"]]

    if not q_words:
        return 0.8

    matches = sum(1 for w in q_words if w in ans_lower)
    relevance = matches / len(q_words)
    return min(1.0, max(0.2, relevance + 0.3))


def compute_overall_grade(score: float) -> str:
    """Convert a 0-100 numerical score into a standard academic grade badge.

    A+ >= 95
    A  >= 90
    B  >= 80
    C  >= 70
    D  >= 60
    F  < 60
    """
    if score >= 95.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


# -----------------------------------------------------------------------------
# 3. Full Comparison Evaluation Runner
# -----------------------------------------------------------------------------

def run_evaluation_benchmark(
    vector_store: VectorStore,
    dataset: Optional[List[Dict[str, Any]]] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run both Baseline RAG and Agentic RAG against the ground-truth benchmark.

    Computes actual, reproducible metrics for both modes side-by-side.

    Args:
        vector_store: VectorStore with documents indexed.
        dataset: List of ground-truth test items. Defaults to loading evaluation.json.
        provider: 'ollama', 'builtin', or 'gemini'.
        api_key: Optional Gemini API key.
        model_name: Optional model name (e.g. 'qwen2.5').
        limit: Optional cap on number of test questions to run.

    Returns:
        Structured dictionary containing summary comparison table, scores, and per-item details.
    """
    if dataset is None:
        dataset = load_evaluation_dataset()

    test_items = dataset[:limit] if limit else dataset

    baseline_metrics = {
        "precision": [],
        "recall": [],
        "hit_rate": [],
        "faithfulness": [],
        "relevance": [],
        "correctness": [],
    }

    agentic_metrics = {
        "precision": [],
        "recall": [],
        "hit_rate": [],
        "faithfulness": [],
        "relevance": [],
        "correctness": [],
    }

    detailed_results = []

    for item in test_items:
        q = item["question"]
        gt = item["ground_truth"]
        rel_pages = item.get("relevant_pages", [])
        rel_kws = item.get("relevant_keywords", [])
        q_type = item.get("type", "SIMPLE")
        is_not_present = (q_type == "NOT_PRESENT")

        # ---------------------------------------------------------------------
        # 1. Run Baseline RAG
        # ---------------------------------------------------------------------
        b_res = run_baseline_rag(
            query=q,
            vector_store=vector_store,
            top_k=TOP_K,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
        )
        b_p = calculate_precision_at_k(b_res.retrieved_chunks, rel_pages, k=TOP_K)
        b_r = calculate_recall_at_k(b_res.retrieved_chunks, rel_pages, k=TOP_K)
        b_hr = calculate_hit_rate_at_k(b_res.retrieved_chunks, rel_pages, k=TOP_K)
        b_f = calculate_faithfulness(b_res.answer, b_res.retrieved_chunks, is_not_present)
        b_rel = calculate_answer_relevance(q, b_res.answer)
        b_corr = calculate_answer_correctness(b_res.answer, gt, rel_kws, is_not_present)

        baseline_metrics["precision"].append(b_p)
        baseline_metrics["recall"].append(b_r)
        baseline_metrics["hit_rate"].append(b_hr)
        baseline_metrics["faithfulness"].append(b_f)
        baseline_metrics["relevance"].append(b_rel)
        baseline_metrics["correctness"].append(b_corr)

        # ---------------------------------------------------------------------
        # 2. Run Agentic RAG
        # ---------------------------------------------------------------------
        a_res = run_agentic_rag(
            query=q,
            vector_store=vector_store,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
        )
        a_chunks = a_res.get("retrieved_chunks", [])
        a_ans = a_res.get("answer", "")

        a_p = calculate_precision_at_k(a_chunks, rel_pages, k=TOP_K)
        a_r = calculate_recall_at_k(a_chunks, rel_pages, k=TOP_K)
        a_hr = calculate_hit_rate_at_k(a_chunks, rel_pages, k=TOP_K)
        a_f = calculate_faithfulness(a_ans, a_chunks, is_not_present)
        a_rel = calculate_answer_relevance(q, a_ans)
        a_corr = calculate_answer_correctness(a_ans, gt, rel_kws, is_not_present)

        agentic_metrics["precision"].append(a_p)
        agentic_metrics["recall"].append(a_r)
        agentic_metrics["hit_rate"].append(a_hr)
        agentic_metrics["faithfulness"].append(a_f)
        agentic_metrics["relevance"].append(a_rel)
        agentic_metrics["correctness"].append(a_corr)

        detailed_results.append({
            "id": item.get("id"),
            "question": q,
            "type": q_type,
            "baseline": {
                "answer": b_res.answer,
                "precision": round(b_p, 2),
                "recall": round(b_r, 2),
                "correctness": round(b_corr, 2),
            },
            "agentic": {
                "classification": a_res.get("classification"),
                "answer": a_ans,
                "precision": round(a_p, 2),
                "recall": round(a_r, 2),
                "correctness": round(a_corr, 2),
                "evidence_sufficient": a_res.get("evidence_sufficient"),
            },
        })

    def avg(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    b_avg_p = avg(baseline_metrics["precision"]) * 100
    b_avg_r = avg(baseline_metrics["recall"]) * 100
    b_avg_hr = avg(baseline_metrics["hit_rate"]) * 100
    b_avg_f = avg(baseline_metrics["faithfulness"]) * 100
    b_avg_rel = avg(baseline_metrics["relevance"]) * 100
    b_avg_corr = avg(baseline_metrics["correctness"]) * 100

    a_avg_p = avg(agentic_metrics["precision"]) * 100
    a_avg_r = avg(agentic_metrics["recall"]) * 100
    a_avg_hr = avg(agentic_metrics["hit_rate"]) * 100
    a_avg_f = avg(agentic_metrics["faithfulness"]) * 100
    a_avg_rel = avg(agentic_metrics["relevance"]) * 100
    a_avg_corr = avg(agentic_metrics["correctness"]) * 100

    # Composite scores (weighted average)
    # Retrieval score: 50% Recall + 30% Precision + 20% Hit Rate
    b_retrieval_score = (0.5 * b_avg_r) + (0.3 * b_avg_p) + (0.2 * b_avg_hr)
    a_retrieval_score = (0.5 * a_avg_r) + (0.3 * a_avg_p) + (0.2 * a_avg_hr)

    # Generation score: 40% Correctness + 35% Faithfulness + 25% Relevance
    b_generation_score = (0.4 * b_avg_corr) + (0.35 * b_avg_f) + (0.25 * b_avg_rel)
    a_generation_score = (0.4 * a_avg_corr) + (0.35 * a_avg_f) + (0.25 * a_avg_rel)

    # Overall score: 45% Retrieval + 55% Generation
    b_overall = (0.45 * b_retrieval_score) + (0.55 * b_generation_score)
    a_overall = (0.45 * a_retrieval_score) + (0.55 * a_generation_score)

    return {
        "summary": {
            "evaluated_questions": len(test_items),
            "baseline": {
                "precision": round(b_avg_p, 1),
                "recall": round(b_avg_r, 1),
                "hit_rate": round(b_avg_hr, 1),
                "faithfulness": round(b_avg_f, 1),
                "relevance": round(b_avg_rel, 1),
                "correctness": round(b_avg_corr, 1),
                "retrieval_score": round(b_retrieval_score, 1),
                "generation_score": round(b_generation_score, 1),
                "overall_score": round(b_overall, 1),
                "grade": compute_overall_grade(b_overall),
            },
            "agentic": {
                "precision": round(a_avg_p, 1),
                "recall": round(a_avg_r, 1),
                "hit_rate": round(a_avg_hr, 1),
                "faithfulness": round(a_avg_f, 1),
                "relevance": round(a_avg_rel, 1),
                "correctness": round(a_avg_corr, 1),
                "retrieval_score": round(a_retrieval_score, 1),
                "generation_score": round(a_generation_score, 1),
                "overall_score": round(a_overall, 1),
                "grade": compute_overall_grade(a_overall),
            },
        },
        "details": detailed_results,
    }
