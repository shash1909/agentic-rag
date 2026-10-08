"""Query Analyzer module for Adaptive Agentic RAG.

Classifies incoming user queries into:
1. SIMPLE: Direct single-fact lookup -> normal RAG pipeline.
2. COMPLEX: Multi-part, causal, or comparative -> Agentic RAG with query decomposition.
3. AMBIGUOUS: Underspecified, vague query -> Agentic RAG with query rewriting.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import logging

from prompts import (
    QUERY_ANALYSIS_PROMPT,
    QUERY_REWRITE_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
)
from generation import call_gemini_json, call_gemini_text, get_gemini_client

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysisResult:
    """Structured output of the Query Analyzer.

    Attributes:
        original_query: The raw query from the user.
        classification: One of 'SIMPLE', 'COMPLEX', 'AMBIGUOUS'.
        reason: Explanation of the classification.
        rewritten_query: Rewritten query if AMBIGUOUS (None otherwise).
        sub_questions: List of sub-questions if COMPLEX (empty list otherwise).
    """
    original_query: str
    classification: str
    reason: str
    rewritten_query: Optional[str] = None
    sub_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "original_query": self.original_query,
            "classification": self.classification,
            "reason": self.reason,
            "rewritten_query": self.rewritten_query,
            "sub_questions": self.sub_questions,
        }


def _rewrite_ambiguous_query(query: str) -> str:
    """Rewrite a vague, ambiguous, or keyword-only query into a clean, natural search query."""
    q_clean = query.strip()
    strip_chars = " ?.!/\\"
    q_lower = q_clean.lower().strip(strip_chars)

    # 1. Check for specific risk questions
    if "risk" in q_lower:
        return "What are the primary operational, financial, and strategic risks described in the documents?"

    # 2. Check for vague negative/impact questions
    if "why is this bad" in q_lower or "why is it bad" in q_lower or "bad for the company" in q_lower:
        return "What negative financial, operational, or strategic impacts of this initiative are described in the uploaded documents?"

    # 3. Check for vague summary / overview prompts
    if q_lower in ["tell me more", "tell me about it", "tell me about this", "what about it", "summarize", "summary"]:
        return "What is an overview and summary of the key findings described in the documents?"

    if q_lower in ["explain this", "explain it", "explain", "is it good", "what does this mean"]:
        return "What are the core concepts and explanations described in the documents?"

    # 4. Canonical tech term & acronym lookups (singular/plural)
    canonical_terms = {
        "llm": "What is an LLM?",
        "llms": "What is an LLM?",
        "rag": "What is RAG?",
        "transformer": "What is a transformer?",
        "transformers": "What is a transformer?",
        "embedding": "What is an embedding?",
        "embeddings": "What is an embedding?",
        "token": "What is a token?",
        "tokens": "What is a token?",
        "agent": "What is an AI agent?",
        "agents": "What is an AI agent?",
        "ai agent": "What is an AI agent?",
        "ai agents": "What is an AI agent?",
        "self attention": "What is self attention?",
        "attention": "What is the attention mechanism?",
        "vector db": "What is a vector database?",
        "vector database": "What is a vector database?",
        "vector databases": "What is a vector database?",
    }
    if q_lower in canonical_terms:
        return canonical_terms[q_lower]

    words = q_clean.split()
    if len(words) == 1:
        single_word = words[0].strip(strip_chars)
        sw_lower = single_word.lower()

        # Plural acronyms (e.g. cnns -> CNNs, rnns -> RNNs, apis -> APIs)
        if sw_lower.endswith("s") and len(sw_lower) <= 6:
            stem = sw_lower[:-1]
            if stem in ["cnn", "rnn", "api", "gpu", "tpu", "svm", "gan"]:
                return f"What is an {stem.upper()}?"

        # General acronyms
        if len(single_word) <= 5 and single_word.isalpha():
            article = "an" if sw_lower in ["llm", "rnn", "api", "svm", "mlp", "nlp"] else "a"
            if sw_lower in ["rag", "bert", "gpt", "rlhf", "lora"]:
                return f"What is {single_word.upper()}?"
            return f"What is {article} {single_word.upper()}?"

        if sw_lower in ["autoencoder", "classifier", "decoder", "encoder", "model"]:
            article = "an" if sw_lower[0] in "aeiou" else "a"
            return f"What is {article} {sw_lower}?"
        return f"What is {single_word}?"

    # If it's a short 2-3 word concept without question words (e.g. "self attention", "vector database"):
    question_words = {"what", "why", "how", "who", "when", "where", "which", "is", "are", "can", "could", "does", "did"}
    has_question_word = any(w.lower() in question_words for w in words)
    if not has_question_word:
        clean_concept = q_clean.strip(strip_chars)
        return f"What is {clean_concept}?"

    # If already phrased as "what is ..." or "what are ...", preserve cleanly
    if q_lower.startswith("what is ") or q_lower.startswith("what are "):
        return f"{q_clean.rstrip('?')}?"

    clean_fallback = q_clean.strip(strip_chars)
    return f"What is {clean_fallback}?"


def _heuristic_classify_query(query: str) -> QueryAnalysisResult:
    """Rule-based heuristic classifier used as a fast offline fallback or when LLM is unavailable.

    Provides predictable classification for test suites and offline demos.
    """
    q_clean = query.strip()
    q_lower = q_clean.lower()
    words = q_clean.split()

    # 1. Ambiguous detection: very brief, underspecified, or vague pronouns
    ambiguous_patterns = [
        r"^what are (the )?risks\??$",
        r"^what are risks\??$",
        r"^tell me more\??$",
        r"^tell me about (it|results|this)\??$",
        r"^why is this bad( for the company)?\??$",
        r"^is it good\??$",
        r"^summarize\??$",
        r"^what about it\??$",
        r"^explain( this| it)?\??$",
    ]

    is_pattern_ambiguous = any(re.match(p, q_lower) for p in ambiguous_patterns)
    
    # Bare keywords or concepts (e.g. "rag", "transformer", "self attention", "risks")
    question_words = {"what", "why", "how", "who", "when", "where", "which", "is", "are", "can", "could", "does", "did"}
    has_question_word = any(w.lower().strip("?.!") in question_words for w in words)
    is_bare_keyword = (len(words) <= 3 and not has_question_word)

    if is_pattern_ambiguous or is_bare_keyword:
        rewritten = _rewrite_ambiguous_query(q_clean)
        return QueryAnalysisResult(
            original_query=q_clean,
            classification="AMBIGUOUS",
            reason="Query is a bare keyword, underspecified topic, or lacks concrete question context.",
            rewritten_query=rewritten,
            sub_questions=[],
        )

    # 2. Complex detection: comparative, multi-clause, or multi-question
    complex_indicators = [
        " and what impact ",
        " and explain why",
        "compare",
        "versus",
        " vs ",
        "differences between",
        "how did",
        "why did",
        "affect profitability",
    ]

    has_multiple_questions = q_clean.count("?") > 1
    has_complex_indicator = any(ind in q_lower for ind in complex_indicators)
    is_multi_part = (" and " in q_lower and ("why" in q_lower or "how" in q_lower or "what impact" in q_lower))

    if has_multiple_questions or has_complex_indicator or is_multi_part:
        # Generate standard decomposed sub-questions
        sub_qs = []
        if "compare" in q_lower or " vs " in q_lower or "versus" in q_lower:
            sub_qs = [
                f"What are the metrics and details for the first entity in: '{q_clean}'?",
                f"What are the metrics and details for the second entity in: '{q_clean}'?",
                f"What are the key drivers explaining differences in: '{q_clean}'?",
            ]
        elif "and what impact" in q_lower or "impact did this have" in q_lower:
            parts = re.split(r" and (?:what )?impact", q_clean, flags=re.IGNORECASE)
            q1 = parts[0].strip() + ("?" if not parts[0].strip().endswith("?") else "")
            q2 = f"What was the impact on profitability or performance related to '{parts[0].strip()}'?"
            sub_qs = [q1, q2]
        else:
            sub_qs = [
                f"What are the primary background factors for: '{q_clean}'?",
                f"What are the consequences and explanations for: '{q_clean}'?",
            ]

        return QueryAnalysisResult(
            original_query=q_clean,
            classification="COMPLEX",
            reason="Query involves multiple topics, comparative analysis, or cause-and-effect relationships.",
            rewritten_query=None,
            sub_questions=sub_qs,
        )

    # 3. Default to SIMPLE for single-fact lookups
    return QueryAnalysisResult(
        original_query=q_clean,
        classification="SIMPLE",
        reason="Query is a focused, single-fact question answerable with standard retrieval.",
        rewritten_query=None,
        sub_questions=[],
    )


def analyze_query(
    query: str,
    api_key: Optional[str] = None,
    force_heuristic: bool = False,
) -> QueryAnalysisResult:
    """Analyze and classify a user query into SIMPLE, COMPLEX, or AMBIGUOUS.

    Uses Gemini LLM when available, with automatic graceful fallback to the rule-based heuristic.

    Args:
        query: User input text.
        api_key: Optional Gemini API key.
        force_heuristic: If True, uses the fast heuristic rule engine without calling LLM.

    Returns:
        QueryAnalysisResult instance.
    """
    if not query or not query.strip():
        return QueryAnalysisResult(
            original_query="",
            classification="AMBIGUOUS",
            reason="Empty query.",
            rewritten_query=None,
            sub_questions=[],
        )

    # If force_heuristic or Gemini client is not configured, use fast rule-based engine
    client = get_gemini_client(api_key=api_key)
    if force_heuristic or client is None:
        return _heuristic_classify_query(query)

    # Call Gemini for LLM-driven classification
    try:
        prompt = QUERY_ANALYSIS_PROMPT.format(query=query.strip())
        data = call_gemini_json(prompt, api_key=api_key)

        if data and "classification" in data:
            cls_name = str(data.get("classification", "SIMPLE")).upper()
            if cls_name not in ["SIMPLE", "COMPLEX", "AMBIGUOUS"]:
                cls_name = "SIMPLE"

            reason = data.get("reason", "Classified by Gemini Query Analyzer.")
            rewritten = data.get("rewritten_query")
            sub_qs = data.get("sub_questions", [])

            # Validation
            if cls_name == "AMBIGUOUS" and not rewritten:
                rewritten = _heuristic_classify_query(query).rewritten_query

            if cls_name == "COMPLEX" and (not sub_qs or not isinstance(sub_qs, list)):
                sub_qs = _heuristic_classify_query(query).sub_questions

            return QueryAnalysisResult(
                original_query=query.strip(),
                classification=cls_name,
                reason=reason,
                rewritten_query=rewritten if cls_name == "AMBIGUOUS" else None,
                sub_questions=sub_qs if cls_name == "COMPLEX" else [],
            )

    except Exception as e:
        logger.warning(f"LLM query analysis failed ({e}). Falling back to heuristic classifier.")

    # Fallback if LLM call or parsing failed
    return _heuristic_classify_query(query)
