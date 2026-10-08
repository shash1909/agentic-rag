"""Agentic RAG workflow module using LangGraph.

This module implements:
1. Baseline RAG: Direct Query -> Semantic Retrieval -> Grounded Generation.
2. Agentic RAG: Lightweight LangGraph state machine:
   - Query Analysis (SIMPLE, COMPLEX, AMBIGUOUS)
   - Conditional routing (Direct, Decompose, or Rewrite)
   - Deduplicated multi-query retrieval
   - Evidence sufficiency evaluation
   - Maximum 1 refinement/retry loop if evidence is lacking
   - Strictly grounded generation with citations
"""

from typing import TypedDict, List, Dict, Any, Optional
import logging

from langgraph.graph import StateGraph, END

from config import TOP_K
from vector_store import VectorStore, RetrievedChunk
from retrieval import retrieve_chunks, retrieve_multi_query, format_context_for_prompt
from query_analyzer import analyze_query, QueryAnalysisResult
from generation import (
    generate_grounded_answer,
    call_llm_json,
    evaluate_evidence_sufficiency,
    extract_content_keywords,
    GroundedAnswer,
    INSUFFICIENT_EVIDENCE_PHRASE,
)
from prompts import EVIDENCE_CHECK_PROMPT

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Agent State Definition
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    """Internal state passed across nodes in the LangGraph workflow."""
    original_query: str
    query_type: str                   # 'SIMPLE', 'COMPLEX', 'AMBIGUOUS'
    classification_reason: str
    rewritten_query: Optional[str]
    sub_questions: List[str]
    search_queries: List[str]
    retrieved_chunks: List[RetrievedChunk]
    evidence_sufficient: bool
    evidence_reason: str
    retry_count: int                  # Capped at 1
    refined_query: Optional[str]
    final_answer: str
    citations: List[str]
    trace: List[Dict[str, Any]]
    api_key: Optional[str]
    provider: Optional[str]
    model_name: Optional[str]
    vector_store: Any


# -----------------------------------------------------------------------------
# LangGraph Nodes
# -----------------------------------------------------------------------------

def analyze_query_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Classify query and determine query transformation strategy."""
    query = state["original_query"]
    api_key = state.get("api_key")

    analysis: QueryAnalysisResult = analyze_query(query, api_key=api_key)

    search_queries = [query]
    if analysis.classification == "AMBIGUOUS" and analysis.rewritten_query:
        search_queries = [analysis.rewritten_query]
    elif analysis.classification == "COMPLEX" and analysis.sub_questions:
        search_queries = analysis.sub_questions

    trace_entry = {
        "step": "Query Analysis",
        "query_type": analysis.classification,
        "reason": analysis.reason,
        "rewritten_query": analysis.rewritten_query,
        "sub_questions": analysis.sub_questions,
        "active_search_queries": search_queries,
    }

    return {
        "query_type": analysis.classification,
        "classification_reason": analysis.reason,
        "rewritten_query": analysis.rewritten_query,
        "sub_questions": analysis.sub_questions,
        "search_queries": search_queries,
        "trace": state.get("trace", []) + [trace_entry],
    }


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Retrieve relevant chunks for all active search queries."""
    vector_store: VectorStore = state["vector_store"]
    queries = state["search_queries"]

    if len(queries) > 1:
        # Multi-query retrieval for decomposed sub-questions
        chunks = retrieve_multi_query(
            queries=queries,
            vector_store=vector_store,
            top_k_per_query=3,
        )
    else:
        # Single query retrieval
        q = queries[0] if queries else state["original_query"]
        chunks = retrieve_chunks(
            query=q,
            vector_store=vector_store,
            top_k=TOP_K,
        )

    trace_entry = {
        "step": "Retrieval",
        "queries_searched": queries,
        "chunks_retrieved_count": len(chunks),
        "top_score": round(chunks[0].score, 4) if chunks else 0.0,
        "chunks": [c.to_dict() for c in chunks[:4]],
    }

    return {
        "retrieved_chunks": chunks,
        "trace": state.get("trace", []) + [trace_entry],
    }


def check_evidence_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Check whether retrieved chunks provide sufficient factual evidence."""
    chunks = state.get("retrieved_chunks", [])
    query = state["original_query"]
    api_key = state.get("api_key")
    provider = state.get("provider")
    model_name = state.get("model_name")

    # 1. Base sufficiency evaluation
    is_sufficient, reason = evaluate_evidence_sufficiency(query, chunks, min_score_threshold=0.35)
    refined_query = None

    if not is_sufficient:
        content_words = extract_content_keywords(query)
        words_focus = " ".join(content_words[:4]) if content_words else query
        refined_query = f"Key factual evidence, data, and details regarding {words_focus}"

    # 2. If LLM backend is configured and initial heuristic found matching tokens, verify with LLM
    if is_sufficient and (api_key or provider in ["ollama", "qwen"]):
        try:
            context_str = format_context_for_prompt(chunks[:4])
            prompt = EVIDENCE_CHECK_PROMPT.format(context=context_str, query=query)
            data = call_llm_json(
                prompt=prompt,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
            )

            if data and "is_sufficient" in data:
                llm_sufficient = bool(data["is_sufficient"])
                llm_reason = data.get("reason", "Evaluated by LLM evidence checker.")
                llm_refined = data.get("refined_query")

                if not llm_sufficient:
                    is_sufficient = False
                    reason = llm_reason
                    refined_query = llm_refined or refined_query
                else:
                    reason = f"{reason} | Confirmed by LLM: {llm_reason}"
        except Exception as e:
            logger.warning(f"LLM evidence check bypassed ({e}). Using grounding evaluator.")

    trace_entry = {
        "step": "Evidence Check",
        "is_sufficient": is_sufficient,
        "reason": reason,
        "refined_query": refined_query,
    }

    return {
        "evidence_sufficient": is_sufficient,
        "evidence_reason": reason,
        "refined_query": refined_query,
        "trace": state.get("trace", []) + [trace_entry],
    }


def refine_and_retry_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Perform ONE query refinement and second retrieval attempt."""
    vector_store: VectorStore = state["vector_store"]
    original_query = state["original_query"]
    refined_q = state.get("refined_query") or f"Detailed facts and figures for {original_query}"

    # Retrieve with refined query
    additional_chunks = retrieve_chunks(
        query=refined_q,
        vector_store=vector_store,
        top_k=3,
    )

    # Merge and deduplicate with existing chunks
    merged = {c.chunk_id: c for c in state.get("retrieved_chunks", [])}
    for chunk in additional_chunks:
        if chunk.chunk_id not in merged or chunk.score > merged[chunk.chunk_id].score:
            merged[chunk.chunk_id] = chunk

    sorted_chunks = sorted(merged.values(), key=lambda c: c.score, reverse=True)

    trace_entry = {
        "step": "Query Refinement & Retry",
        "refined_query": refined_q,
        "new_chunks_added": len(additional_chunks),
        "total_merged_chunks": len(sorted_chunks),
    }

    return {
        "retrieved_chunks": sorted_chunks,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": state.get("trace", []) + [trace_entry],
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Generate grounded answer from retrieved chunks."""
    query = state["original_query"]
    chunks = state.get("retrieved_chunks", [])
    api_key = state.get("api_key")
    provider = state.get("provider")
    model_name = state.get("model_name")
    evidence_sufficient = state.get("evidence_sufficient", True)

    # If evidence is insufficient (either initially or after retry)
    if not evidence_sufficient or not chunks:
        final_answer = INSUFFICIENT_EVIDENCE_PHRASE
        citations = []
        trace_status = "Refused due to insufficient factual evidence in documents."
    else:
        grounded_result: GroundedAnswer = generate_grounded_answer(
            query=query,
            retrieved_chunks=chunks,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
        )
        final_answer = grounded_result.answer
        citations = grounded_result.citations
        trace_status = "Generated grounded answer with citations."

    trace_entry = {
        "step": "Answer Generation",
        "answer_preview": final_answer[:160] + "..." if len(final_answer) > 160 else final_answer,
        "citations": citations,
        "chunks_used": len(chunks),
        "status": trace_status,
    }

    return {
        "final_answer": final_answer,
        "citations": citations,
        "trace": state.get("trace", []) + [trace_entry],
    }


# -----------------------------------------------------------------------------
# Routing logic (Conditional Edges)
# -----------------------------------------------------------------------------

def route_after_evidence(state: AgentState) -> str:
    """Determine whether to proceed to generation or execute ONE retry."""
    is_sufficient = state.get("evidence_sufficient", False)
    retry_count = state.get("retry_count", 0)

    if is_sufficient:
        return "generate"

    # Only allow 1 retry
    if retry_count < 1:
        return "refine_and_retry"

    return "generate"


# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------

def build_agent_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("check_evidence", check_evidence_node)
    workflow.add_node("refine_and_retry", refine_and_retry_node)
    workflow.add_node("generate", generate_node)

    # Add Edges
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "check_evidence")

    # Conditional edge after evidence check
    workflow.add_conditional_edges(
        "check_evidence",
        route_after_evidence,
        {
            "generate": "generate",
            "refine_and_retry": "refine_and_retry",
        },
    )

    workflow.add_edge("refine_and_retry", "check_evidence")
    workflow.add_edge("generate", END)

    return workflow.compile()


# Singleton compiled graph
_compiled_agent = None


def get_agent_graph():
    """Lazy initialize and return compiled LangGraph agent."""
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = build_agent_graph()
    return _compiled_agent


# -----------------------------------------------------------------------------
# Public Execution APIs: Baseline RAG & Agentic RAG
# -----------------------------------------------------------------------------

def run_baseline_rag(
    query: str,
    vector_store: VectorStore,
    top_k: int = TOP_K,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> GroundedAnswer:
    """Run standard Baseline RAG: Direct Query -> Retrieve Top-K -> LLM Generation.

    Args:
        query: User input.
        vector_store: VectorStore instance.
        top_k: Number of chunks to retrieve.
        provider: 'ollama', 'builtin', or 'gemini'.
        api_key: Optional Gemini API key.
        model_name: Optional model name (e.g. 'qwen2.5').

    Returns:
        GroundedAnswer instance.
    """
    retrieved_chunks = retrieve_chunks(
        query=query,
        vector_store=vector_store,
        top_k=top_k,
    )

    answer = generate_grounded_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
    )

    return answer


def run_agentic_rag(
    query: str,
    vector_store: VectorStore,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Adaptive Agentic RAG via the LangGraph state machine.

    Performs query classification, rewriting or decomposition when necessary,
    multi-query retrieval, evidence verification, and single-retry refinement.

    Args:
        query: User input.
        vector_store: VectorStore instance.
        provider: 'ollama', 'builtin', or 'gemini'.
        api_key: Optional Gemini API key.
        model_name: Optional model name (e.g. 'qwen2.5').

    Returns:
        Dictionary containing final answer, citations, retrieved chunks,
        classification details, and full trace log.
    """
    agent = get_agent_graph()

    initial_state: AgentState = {
        "original_query": query,
        "query_type": "SIMPLE",
        "classification_reason": "",
        "rewritten_query": None,
        "sub_questions": [],
        "search_queries": [query],
        "retrieved_chunks": [],
        "evidence_sufficient": True,
        "evidence_reason": "",
        "retry_count": 0,
        "refined_query": None,
        "final_answer": "",
        "citations": [],
        "trace": [],
        "api_key": api_key,
        "provider": provider,
        "model_name": model_name,
        "vector_store": vector_store,
    }

    final_state = agent.invoke(initial_state)

    return {
        "query": query,
        "classification": final_state.get("query_type", "SIMPLE"),
        "classification_reason": final_state.get("classification_reason", ""),
        "rewritten_query": final_state.get("rewritten_query"),
        "sub_questions": final_state.get("sub_questions", []),
        "agentic_activated": final_state.get("query_type") in ["COMPLEX", "AMBIGUOUS"] or final_state.get("retry_count", 0) > 0,
        "answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "evidence_sufficient": final_state.get("evidence_sufficient", True),
        "evidence_reason": final_state.get("evidence_reason", ""),
        "retry_count": final_state.get("retry_count", 0),
        "trace": final_state.get("trace", []),
    }
