"""Prompt templates for Adaptive Agentic RAG.

All prompts used for query classification, rewriting, decomposition,
evidence checking, and grounded generation are defined here.
"""

# -----------------------------------------------------------------------------
# 1. Query Analysis Prompt
# -----------------------------------------------------------------------------
QUERY_ANALYSIS_PROMPT = """You are a Query Analyzer in an Adaptive RAG system.
Your job is to analyze the user's question and classify it into EXACTLY ONE of three categories:

1. SIMPLE: A straightforward, single-fact or single-topic lookup that can be directly answered by retrieving 2-3 document chunks.
   Examples:
   - "What was the company's revenue in 2024?"
   - "Who is the CEO of Acme Corp?"
   - "What is the return policy period?"

2. COMPLEX: A multi-part, comparative, causal, or multi-step question requiring synthesis across multiple sections or topics.
   Examples:
   - "Why did revenue decline and what impact did this have on profitability?"
   - "Compare the profitability of Company A and Company B and explain why."
   - "What were the primary revenue drivers in 2023 versus 2024?"

3. AMBIGUOUS: An underspecified, vague, or bare-keyword question missing essential context or entities.
   Examples:
   - "rag" -> Rewritten: "What is RAG?"
   - "transformer" -> Rewritten: "What is a transformer?"
   - "What are the risks?" -> Rewritten: "What are the primary operational, financial, and strategic risks described in the documents?"
   - "Why is this bad for the company?" -> Rewritten: "What negative financial, operational, or strategic impacts of this initiative are described in the uploaded documents?"
   - "Tell me about results." -> Rewritten: "What were the key financial and operating results described in the documents?"

Respond strictly in valid JSON format with the following structure:
{
  "classification": "SIMPLE" | "COMPLEX" | "AMBIGUOUS",
  "reason": "Brief 1-sentence explanation of why this classification was chosen.",
  "rewritten_query": "For AMBIGUOUS queries, a clear, natural search query (e.g. 'What is RAG?'). Otherwise null.",
  "sub_questions": ["For COMPLEX queries, 2 to 4 focused sub-questions. Otherwise empty list []"]
}

User Question: {query}
"""

# -----------------------------------------------------------------------------
# 2. Query Rewriting Prompt (for Ambiguous Queries)
# -----------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = """You are an expert search query optimizer for a RAG system.
The user provided a vague, ambiguous, or single-keyword question.
Rewrite it into a clear, natural, retrieval-optimized search query for semantic vector search over uploaded documents.

Guidelines:
- If the user provides a bare keyword, concept, or acronym (e.g. "rag", "transformer", "attention"), rewrite it as a direct question asking for its definition and function (e.g. "What is RAG?", "What is a transformer?").
- If the user asks a vague question (e.g. "What are the risks?"), make implicit context explicit (e.g. "What are the primary operational and strategic risks described in the documents?").
- Keep it concise, natural, and free of unnecessary filler words.
- DO NOT add artificial boilerplate like "key details, risks, operational factors" unless the user specifically asked about risks or operations.
- Return ONLY the rewritten query text, with no preamble or explanation.

Original Query: {query}
Rewritten Query:"""

# -----------------------------------------------------------------------------
# 3. Query Decomposition Prompt (for Complex Queries)
# -----------------------------------------------------------------------------
QUERY_DECOMPOSITION_PROMPT = """You are an expert query planner for an Agentic RAG system.
The user asked a complex, multi-part, or comparative question.
Break it down into 2 to 4 distinct, atomic sub-questions that can each be searched independently in the document store.

Respond strictly in valid JSON format:
{
  "sub_questions": [
    "Sub-question 1",
    "Sub-question 2"
  ]
}

Complex Question: {query}
"""

# -----------------------------------------------------------------------------
# 4. Evidence Sufficiency Check Prompt
# -----------------------------------------------------------------------------
EVIDENCE_CHECK_PROMPT = """You are an Evidence Evaluator in an Agentic RAG system.
Your job is to determine whether the retrieved document passages contain sufficient factual evidence to answer the user's question.

Retrieved Passages:
{context}

User Question: {query}

Evaluate:
1. Does the retrieved context contain direct factual evidence to answer the core question?
2. If NOT, what specific information is missing?

Respond strictly in valid JSON format:
{
  "is_sufficient": true | false,
  "reason": "Brief explanation of whether context is sufficient or what is missing.",
  "refined_query": "If NOT sufficient, a refined query to search for the missing piece. Otherwise null."
}
"""

# -----------------------------------------------------------------------------
# 5. Grounded RAG Generation Prompt
# -----------------------------------------------------------------------------
RAG_GENERATION_PROMPT = """You are an accurate, strictly grounded AI assistant answering questions based ONLY on uploaded documents.

CRITICAL GROUNDING RULES:
1. Answer the question using ONLY the factual information provided in the Context below.
2. DO NOT make assumptions, invent facts, or extrapolate beyond what is explicitly stated in the Context.
3. If the Context does NOT contain enough information to answer the question, state EXACTLY:
   "I don't have sufficient evidence in the uploaded documents to answer this question."
4. Do NOT include inline document names or bracketed citations (e.g. [Document.pdf, Page X]) in your answer text. Provide only the clear, direct, and factual answer.
5. Keep your answer clear, well-structured, and concise.
6. Provide a direct, declarative answer to the user's question. If the Context contains study questions, quizzes, or Q&A headers (such as "Q1:", "Q2:", "Question:"), DO NOT echo or repeat the questions back to the user—synthesize the factual explanation from the answer text.

Context:
{context}

Question: {query}

Grounded Answer:"""
