"""Generation module for Adaptive Agentic RAG.

Supports multiple LLM backends:
1. Ollama (Free local Qwen2.5 / Llama - no API key, runs offline on localhost).
2. Built-in Grounded Synthesizer (100% Free, zero-setup, instant offline fallback).
3. Google Gemini API (Free tier cloud model via google-genai SDK).

Also handles citation parsing, grounding validation, and insufficient evidence detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os
import re
import json
import logging
import urllib.request
import urllib.error

from config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
)
from vector_store import RetrievedChunk
from retrieval import format_context_for_prompt
from prompts import RAG_GENERATION_PROMPT

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_PHRASE = "I don't have sufficient evidence in the uploaded documents to answer this question."


@dataclass
class GroundedAnswer:
    """Represents a generated answer with citation and grounding metadata."""
    query: str
    answer: str
    citations: List[str] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    is_sufficient: bool = True
    provider: str = "ollama"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert answer object to standard dictionary representation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "is_sufficient": self.is_sufficient,
            "provider": self.provider,
            "chunks_count": len(self.retrieved_chunks),
            "error": self.error,
        }


def extract_citations(text: str) -> List[str]:
    """Extract citations formatted like [Document.pdf, Page 12] from text."""
    pattern = r"\[([^\]\n]+,\s*(?:Page|page|p\.)\s*\d+)\]"
    matches = re.findall(pattern, text)

    seen = set()
    unique_citations = []
    for m in matches:
        formatted = f"[{m.strip()}]"
        if formatted not in seen:
            seen.add(formatted)
            unique_citations.append(formatted)

    return unique_citations


def strip_citations(text: str) -> str:
    """Remove inline citation brackets like [Document.pdf, Page 12] and source labels from answer text.

    Ensures the final grounded answer is clean, readable prose without embedded source tags.
    """
    if not text:
        return text

    # 1. Remove parenthetical or bracketed sources: (Source: ...), [Source: ...], (Sources: ...)
    cleaned = re.sub(r"(?:\(|\[)\s*Sources?:?[^)\n\]]*(?:\)|\])", "", text, flags=re.IGNORECASE)
    # 2. Remove standard citation brackets: [Document.pdf, Page 12] or [doc.pdf, p. 5]
    cleaned = re.sub(r"\s*\[[^\]\n]+,\s*(?:Page|page|p\.)\s*\d+\]", "", cleaned)
    # 3. Remove generic document filename brackets: [Document.pdf] or [report_2024.pdf]
    cleaned = re.sub(r"\s*\[[a-zA-Z0-9_\-\.\s]+\.pdf[^\]]*\]", "", cleaned)
    # 4. Remove page brackets: [Page 12] or [p. 12]
    cleaned = re.sub(r"\s*\[(?:Page|page|p\.)\s*\d+\]", "", cleaned)
    # 5. Remove standalone or dangling Source / Sources labels and any following content on that line
    cleaned = re.sub(r"(?:\n|\r|\s)*\bSources?:\s*.*$", "", cleaned, flags=re.IGNORECASE)
    # 6. Clean any leftover trailing brackets
    cleaned = re.sub(r"\s*\[[^\]]+\]\s*$", "", cleaned)
    # 7. Fix spacing before punctuation (e.g. "word ." -> "word.")
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    # 8. Collapse multiple whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


# -----------------------------------------------------------------------------
# Provider 1: Ollama (Free Local Qwen / Llama)
# -----------------------------------------------------------------------------

def is_ollama_available(base_url: str = OLLAMA_BASE_URL) -> Tuple[bool, List[str]]:
    """Check if Ollama service is reachable and return list of installed models."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "agentic-rag"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []


def call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL_NAME,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.1,
) -> str:
    """Call local Ollama instance running Qwen or any local model.

    Args:
        prompt: Full prompt string.
        model: Model name in Ollama (e.g. 'qwen2.5', 'qwen2.5:7b', 'llama3.2').
        base_url: Ollama API URL (defaults to http://localhost:11434).
        temperature: Sampling temperature.

    Returns:
        Generated text string from the local model.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to local Ollama at {base_url}. "
            f"Please run 'ollama run {model}' or switch to Built-in Synthesizer mode. Error: {e}"
        )


# -----------------------------------------------------------------------------
# Provider 2: Google Gemini API (Free Tier)
# -----------------------------------------------------------------------------

def get_gemini_client(api_key: Optional[str] = None):
    """Retrieve an authenticated google-genai Client or None if missing."""
    key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if not key or key == "your_gemini_api_key_here":
        return None
    try:
        from google import genai
        return genai.Client(api_key=key.strip())
    except Exception:
        return None


def call_gemini_api(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    """Call Google Gemini API via official google-genai SDK."""
    from google import genai
    from google.genai import types

    key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if not key or key == "your_gemini_api_key_here":
        raise ValueError("Google Gemini API key is not configured.")

    client = genai.Client(api_key=key.strip())
    model = model_name or os.getenv("GEMINI_MODEL_NAME") or GEMINI_MODEL_NAME

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )

    return (response.text or "").strip()


# -----------------------------------------------------------------------------
# Grounding Guard & Content Keyword Extraction
# -----------------------------------------------------------------------------

STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "and", "for", "with", "this", "that", "from", "are", "is", "was",
    "were", "been", "has", "have", "had", "does", "did", "about", "into",
    "over", "after", "tell", "explain", "give", "can", "could", "would",
    "should", "will", "shall", "their", "there", "they", "them", "some",
    "more", "most", "than", "then", "also", "any", "all", "each", "both",
    "please", "describe", "many", "much", "find", "show", "detail", "details",
}


def extract_content_keywords(query: str) -> List[str]:
    """Extract non-stop semantic content keywords from a user query."""
    tokens = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def evaluate_evidence_sufficiency(
    query: str,
    chunks: List[RetrievedChunk],
    min_score_threshold: float = 0.35,
) -> Tuple[bool, str]:
    """Check if retrieved chunks contain sufficient factual evidence for the query.

    Evaluates semantic similarity score, query keyword overlap, and specific entity targets
    (such as target years or out-of-domain entities) to prevent hallucinations.

    Returns:
        Tuple of (is_sufficient, reason_string)
    """
    if not chunks:
        return False, "No chunks found in vector store."

    top_chunk = chunks[0]
    if top_chunk.score < min_score_threshold:
        return False, f"Top retrieval similarity score ({top_chunk.score:.3f}) is below confidence threshold ({min_score_threshold})."

    content_words = extract_content_keywords(query)
    if not content_words:
        is_ok = top_chunk.score >= 0.40
        return is_ok, f"Retrieval score: {top_chunk.score:.3f}"

    all_text = " ".join(c.text.lower() for c in chunks[:3])

    # 1. Target 4-digit years (e.g. 1995, 2025): if specified year is absent, context lacks evidence
    target_years = [w for w in content_words if re.match(r"^(?:19|20)\d{2}$", w)]
    if target_years and not any(y in all_text for y in target_years):
        years_str = ", ".join(target_years)
        return False, f"Target year ({years_str}) is not present in retrieved document context."

    # 2. Check keyword overlap with plural/singular lemma awareness
    def _matches_word(w: str, text: str) -> bool:
        if w in text:
            return True
        if w.endswith("s") and len(w) > 3 and w[:-1] in text:
            return True
        if not w.endswith("s") and (w + "s") in text:
            return True
        return False

    matched_words = [w for w in content_words if _matches_word(w, all_text)]
    specific_missing = [w for w in content_words if not _matches_word(w, all_text)]

    if len(matched_words) == 0:
        words_str = ", ".join(content_words[:4])
        return False, f"Retrieved documents do not contain any mentions of: {words_str}."

    # 3. Check for essential query target nouns (e.g. president, minister, ceo, founder, revenue, profit, margin, etc.)
    core_target_nouns = {
        "president", "minister", "prime", "ceo", "founder", "revenue", "profit",
        "profitability", "margin", "cost", "price", "complexity", "salary", "author",
        "headquarters", "owner", "location", "deadline", "date",
    }
    missing_targets = [w for w in specific_missing if w in core_target_nouns]
    if missing_targets:
        missing_name = missing_targets[0]
        return False, f"Retrieved document context lacks required information about '{missing_name}'."

    # 4. For 2-word queries (e.g. subject + attribute), both terms must be grounded
    if len(content_words) == 2 and len(matched_words) < 2:
        missing_str = ", ".join(specific_missing)
        return False, f"Retrieved documents lack factual evidence regarding '{missing_str}'."

    # 5. For 3+ word queries, require at least 50% keyword presence
    if len(content_words) >= 3 and len(matched_words) / len(content_words) < 0.50:
        missing_str = ", ".join(specific_missing[:3])
        return False, f"Retrieved documents lack evidence for: {missing_str}."

    return True, f"Found {len(matched_words)} matching keywords with retrieval score {top_chunk.score:.3f}."


# -----------------------------------------------------------------------------
# Provider 3: Built-in Grounded Synthesizer (Instant Zero-Setup Mode)
# -----------------------------------------------------------------------------

DANGLING_ENDINGS = {
    "without", "with", "and", "or", "because", "that", "which", "where",
    "the", "a", "an", "of", "to", "in", "for", "on", "by", "as", "at", "from",
    "due", "into", "onto", "then", "than", "so", "such",
}


def normalize_text_spacing(text: str) -> str:
    """Normalize PDF soft line-wraps into continuous paragraph text while preserving true sentence boundaries."""
    # 1. Clean citation markers like 【14†L264-L272】
    t = re.sub(r"【\d+†L\d+[-–]L\d+】", "", text)
    # 2. Join lines wrapped mid-sentence (newline not preceded by sentence end or colon)
    t = re.sub(r"(?<![.!?:\n])\n+(?!\n|[•\-\d+\.]|[A-Z0-9][a-z0-9]*:)", " ", t)
    # 3. Collapse multiple spaces
    t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()


def is_question_like(text: str) -> bool:
    """Check if a line or sentence is a question, quiz prompt, or section header rather than an informative answer."""
    clean = text.strip()
    if clean.endswith("?"):
        return True
    if re.match(r"^(?:Q\d*[:.]|Question\s*\d*[:.]|Quiz\s*\d*[:.]|Exercise\s*\d*[:.]|Self-Check\b)", clean, re.I):
        return True
    if re.match(r"^(?:What|Why|How|When|Where|Which|Who|Explain|Describe|Discuss)\b", clean, re.I) and ("?" in clean or len(clean.split()) < 15):
        return True
    return False


def is_clean_declarative_sentence(s: str) -> bool:
    """Verify that a sentence candidate is a complete, clean declarative statement."""
def sanitize_declarative_sentence(s: str) -> Optional[str]:
    """Sanitize and validate a candidate declarative sentence.

    Ensures the text is a complete, grammatically sound statement:
    - Rejects metadata headers, question headers, and study guide boilerplate.
    - Strips trailing dangling conjunctions/prepositions (e.g. 'and', 'with', 'to') rather than throwing away good content.
    - Enforces minimum word and character counts.
    """
    clean = s.strip()
    if not clean or len(clean) < 25:
        return None
    if re.match(r"^(?:Difficulty:|Probability:|Topic Distribution|Key Concepts:|Follow-ups:|Common Follow-ups:|<!--|-->)", clean, re.I):
        return None
    if re.match(r"^(?:q\d*[:.]|\d+[:.]|question\b|interview\b|table\b|figure\b|exercise\b)", clean, re.I):
        return None
    if is_question_like(clean):
        return None
    if re.search(r"[↓→←↑]", clean):
        return None
    if re.match(r"^[\d\s【】†L\-_.,:;[\]()|/]+$", clean):
        return None

    words = clean.split()
    while words and words[-1].lower().rstrip(".,;:?!'\"") in DANGLING_ENDINGS:
        words = words[:-1]

    if len(words) < 5:
        return None

    trimmed = " ".join(words).strip()
    if len(trimmed) < 25:
        return None

    return trimmed


def is_clean_declarative_sentence(s: str) -> bool:
    """Check if the string forms a clean, complete declarative statement."""
    return sanitize_declarative_sentence(s) is not None


def synthesize_grounded_answer_from_chunks(
    query: str,
    chunks: List[RetrievedChunk],
) -> GroundedAnswer:
    """Synthesize a strictly grounded factual answer directly from retrieved chunks.

    Extracts high-salience factual sentences matching the query from the top chunks,
    normalizes soft line breaks to prevent truncated sentences, filters out study questions/quizzes,
    attaches verifiable citations, and strictly refuses if factual evidence is insufficient.
    """
    if not chunks:
        return GroundedAnswer(
            query=query,
            answer=INSUFFICIENT_EVIDENCE_PHRASE,
            citations=[],
            retrieved_chunks=[],
            is_sufficient=False,
            provider="builtin",
        )

    # 1. Pre-flight sufficiency check
    is_sufficient, reason = evaluate_evidence_sufficiency(query, chunks, min_score_threshold=0.35)
    if not is_sufficient:
        return GroundedAnswer(
            query=query,
            answer=INSUFFICIENT_EVIDENCE_PHRASE,
            citations=[],
            retrieved_chunks=chunks,
            is_sufficient=False,
            provider="builtin",
        )

    content_words = extract_content_keywords(query)
    content_stems = set(content_words)
    for w in content_words:
        if w.endswith("s") and len(w) > 3 and w not in STOP_WORDS:
            content_stems.add(w[:-1])
        elif not w.endswith("s"):
            content_stems.add(w + "s")

    candidates = []
    seen_texts = set()

    for chunk in chunks[:4]:
        clean_chunk = normalize_text_spacing(chunk.text)

        # 1. Direct Q&A detection: If chunk contains a question matching query followed by Answer:
        qa_pattern = r"(?:(?:Q\d*|\d+)[:.]\s*([^?]+?\?))\s*(?:Answer:\s*)?([^\n]+(?:\n[^\n]+)?)"
        for match in re.finditer(qa_pattern, clean_chunk, re.IGNORECASE):
            q_text, a_text = match.group(1).strip(), match.group(2).strip()
            
            # Extract core subject of question
            subj_match = re.search(r'(?:what is|what are|explain|describe|why|how)\s+(?:an?|the)?\s*([a-zA-Z0-9_\- ]+?)(?:\s+(?:in|of|for|with|using|and)\b|\?)', q_text, re.IGNORECASE)
            core_subj = subj_match.group(1).lower() if subj_match else q_text.lower()
            core_subj_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', core_subj))
            subj_overlap = len(content_stems.intersection(core_subj_words))

            if subj_overlap >= 1:
                clean_a = re.sub(r"^(?:Answer:\s*|\([^\)]+\)\s*)", "", a_text).strip()
                for sub_s in re.split(r"(?<=[.!?])\s+", clean_a):
                    sub_clean = sanitize_declarative_sentence(sub_s)
                    if sub_clean and sub_clean not in seen_texts:
                        seen_texts.add(sub_clean)
                        candidates.append((subj_overlap * 3 + 10, sub_clean, chunk.citation))

        # 2. General sentence splitting on clean sentence boundaries
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean_chunk)
        for s in raw_sentences:
            s_clean = re.sub(r"^(?:Answer:\s*|\([^\)]+\)\s*)", "", s).strip()
            s_clean = sanitize_declarative_sentence(s_clean)
            if not s_clean or s_clean in seen_texts:
                continue

            s_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", s_clean.lower()))
            s_stems = set(s_words)
            for sw in s_words:
                if sw.endswith("s") and len(sw) > 3:
                    s_stems.add(sw[:-1])
                elif not sw.endswith("s"):
                    s_stems.add(sw + "s")

            overlap = content_stems.intersection(s_stems)
            if len(overlap) >= 1:
                score = len(overlap) * 2

                # Check if sentence is defining a DIFFERENT subject
                def_match = re.match(r"^(?:an?|the)?\s*([a-zA-Z0-9_\- ]+?)\s+(?:is|are|refers to|means)\b", s_clean, re.IGNORECASE)
                if def_match:
                    def_subj = def_match.group(1).lower()
                    def_subj_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', def_subj))
                    # If defined subject does not overlap with query target
                    if not content_stems.intersection(def_subj_words) and def_subj_words:
                        score -= 8  # Penalize defining wrong entity
                    elif content_stems.intersection(def_subj_words):
                        score += 6  # Strong bonus for defining exact queried entity

                if any(term in s_clean.lower() for term in ["is a", "trained", "capabilities", "processes", "architecture", "attention", "complexity", "quadratic", "revenue", "margin", "%", "$"]):
                    score += 3
                seen_texts.add(s_clean)
                candidates.append((score, s_clean, chunk.citation))

    # If no sentences met the grounding match criteria, strictly refuse to hallucinate
    if not candidates:
        return GroundedAnswer(
            query=query,
            answer=INSUFFICIENT_EVIDENCE_PHRASE,
            citations=[],
            retrieved_chunks=chunks,
            is_sufficient=False,
            provider="builtin",
        )

    # Sort all candidates across chunks by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    best_candidate = candidates[0]
    best_txt = best_candidate[1].strip()
    if not best_txt.endswith((".", "!", ")")):
        best_txt += "."

    answer_parts = [best_txt]
    final_citations = [best_candidate[2]]

    # If a second strong candidate provides complementary facts without redundancy, include it
    for cand in candidates[1:3]:
        cand_txt = cand[1].strip()
        if cand_txt in best_txt or best_txt in cand_txt:
            continue
        if len(cand_txt) > 25 and cand[0] >= 4:
            if not cand_txt.endswith((".", "!", ")")):
                cand_txt += "."
            answer_parts.append(cand_txt)
            if cand[2] not in final_citations:
                final_citations.append(cand[2])
            break

    combined_answer = strip_citations(" ".join(answer_parts))

    return GroundedAnswer(
        query=query,
        answer=combined_answer,
        citations=final_citations,
        retrieved_chunks=chunks,
        is_sufficient=True,
        provider="builtin",
    )


# -----------------------------------------------------------------------------
# Unified LLM Dispatcher
# -----------------------------------------------------------------------------

def call_llm_text(
    prompt: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
) -> Tuple[str, str]:
    """Call the active LLM provider (Ollama, Gemini, or fallback).

    Returns:
        Tuple of (generated_text, active_provider_name)
    """
    chosen_provider = provider or os.getenv("LLM_PROVIDER", "ollama").lower()

    # 1. Try Ollama (Free Local Qwen)
    if chosen_provider in ["ollama", "qwen"]:
        ollama_model = model_name or os.getenv("OLLAMA_MODEL_NAME", "qwen2.5")
        try:
            text = call_ollama(prompt, model=ollama_model, temperature=temperature)
            return text, f"ollama ({ollama_model})"
        except Exception as e:
            logger.info(f"Ollama not active ({e}). Checking other providers...")

    # 2. Try Gemini API
    if (chosen_provider == "gemini") or (api_key and api_key != "your_gemini_api_key_here"):
        try:
            text = call_gemini_api(prompt, api_key=api_key, model_name=model_name, temperature=temperature)
            return text, "gemini"
        except Exception as e:
            logger.info(f"Gemini call failed ({e}). Falling back...")

    # 3. If explicit call failed, raise descriptive error
    raise RuntimeError(
        "No LLM backend available. To use a 100% free model:\n"
        "1. Start Ollama with: 'ollama run qwen2.5'\n"
        "2. Or enter a Google Gemini API key in the sidebar."
    )


def call_llm_json(
    prompt: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call LLM and parse JSON response with automatic cleanup."""
    try:
        raw_text, _ = call_llm_text(
            prompt,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=0.0,
        )
    except Exception:
        return None

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except Exception:
                pass
        return None


# Backward compatibility aliases
def call_gemini_text(prompt: str, api_key: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.2) -> str:
    text, _ = call_llm_text(prompt, provider="gemini", api_key=api_key, model_name=model_name, temperature=temperature)
    return text


def call_gemini_json(prompt: str, api_key: Optional[str] = None, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return call_llm_json(prompt, provider="gemini", api_key=api_key, model_name=model_name)


def generate_grounded_answer(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> GroundedAnswer:
    """Generate a strictly grounded answer from retrieved chunks using the configured LLM.

    Guards against hallucinations on out-of-domain or unanswerable queries by enforcing
    evidence sufficiency verification, and falls back gracefully to the Built-in Grounded Synthesizer.
    """
    if not retrieved_chunks:
        return GroundedAnswer(
            query=query,
            answer=INSUFFICIENT_EVIDENCE_PHRASE,
            citations=[],
            retrieved_chunks=[],
            is_sufficient=False,
            provider="system",
        )

    # Grounding Guard: Pre-flight check to verify if retrieved context contains sufficient evidence
    is_sufficient, _ = evaluate_evidence_sufficiency(query, retrieved_chunks, min_score_threshold=0.35)
    if not is_sufficient:
        return GroundedAnswer(
            query=query,
            answer=INSUFFICIENT_EVIDENCE_PHRASE,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            is_sufficient=False,
            provider="grounding_guard",
        )

    context_str = format_context_for_prompt(retrieved_chunks)
    prompt = RAG_GENERATION_PROMPT.format(context=context_str, query=query)

    # Try LLM generation (Ollama Qwen or Gemini)
    try:
        raw_answer, active_provider = call_llm_text(
            prompt=prompt,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=0.1,
        )

        insufficient_markers = [
            INSUFFICIENT_EVIDENCE_PHRASE.lower(),
            "sufficient evidence",
            "insufficient evidence",
            "not enough information",
            "not enough evidence",
            "does not provide enough information",
            "does not contain enough information",
            "does not mention",
            "is not mentioned",
            "no mention of",
            "not stated in the",
            "no information provided",
            "cannot answer based on the provided",
            "context does not contain",
            "provided context does not",
            "uploaded documents do not",
        ]
        raw_lower = raw_answer.lower()
        is_insufficient = any(marker in raw_lower for marker in insufficient_markers)

        if is_insufficient:
            return GroundedAnswer(
                query=query,
                answer=INSUFFICIENT_EVIDENCE_PHRASE,
                citations=[],
                retrieved_chunks=retrieved_chunks,
                is_sufficient=False,
                provider=active_provider,
            )

        citations = extract_citations(raw_answer)
        if not citations and retrieved_chunks:
            seen_cits = set()
            citations = []
            for c in retrieved_chunks[:2]:
                if c.citation not in seen_cits:
                    seen_cits.add(c.citation)
                    citations.append(c.citation)

        clean_answer = strip_citations(raw_answer)

        return GroundedAnswer(
            query=query,
            answer=clean_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            is_sufficient=True,
            provider=active_provider,
        )

    except Exception as e:
        logger.info(f"External LLM generation bypassed ({e}). Using Built-in Grounded Synthesizer.")
        # Seamless zero-setup fallback: produces grounded, citation-backed answers directly from context!
        return synthesize_grounded_answer_from_chunks(query, retrieved_chunks)
