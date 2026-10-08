"""Streamlit Application for Adaptive Agentic RAG.

Provides 4 main sections:
1. Chat: Document uploading, query classification badge, agentic activation, grounded answer, citations.
2. Retrieval: Inspection of retrieved chunks, similarity scores, source document, and page numbers.
3. Agent Trace: Visual step-by-step trace (Analysis -> Rewrite/Decompose -> Retrieval -> Evidence Check -> Retry -> Answer).
4. Evaluation: Ground-truth benchmark runner, side-by-side comparison (Baseline RAG vs Agentic RAG), and grading.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json
import streamlit as st
import streamlit.components.v1 as components


def render_mermaid(chart_code: str, height: int = 420):
    """Render Mermaid chart in Streamlit via components.html with fallback."""
    html = f"""
    <div style="display:flex; justify-content:center; width:100%;">
        <pre class="mermaid" style="background:transparent; font-family:sans-serif;">
{chart_code.strip()}
        </pre>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    try:
        components.html(html, height=height, scrolling=False)
    except Exception:
        st.code(chart_code, language="mermaid")

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
    TOP_K,
    EVAL_DATASET_PATH,
)
from ingestion import ingest_pdf, DocumentChunk
from vector_store import VectorStore, RetrievedChunk
from retrieval import retrieve_chunks
from agent import run_baseline_rag, run_agentic_rag
from generation import is_ollama_available
from evaluation import (
    load_evaluation_dataset,
    generate_sample_benchmark_pdf,
    run_evaluation_benchmark,
)

# Streamlit multi-threading safety for local Qdrant SQLite persistence
try:
    from qdrant_client.local.persistence import CollectionPersistence
    CollectionPersistence.CHECK_SAME_THREAD = False
except Exception:
    pass

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Session State & Vector Store Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_vector_store() -> VectorStore:
    """Initialize persistent local VectorStore safely across Streamlit sessions."""
    return VectorStore()


vector_store = get_vector_store()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None


# -----------------------------------------------------------------------------
# Sidebar: LLM Provider, Configuration & Document Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Adaptive Agentic RAG")
    st.caption("Grounded, Citation-Backed RAG with Dynamic Agentic Routing")

    st.markdown("---")
    st.subheader("🧠 Free LLM Provider")

    provider_choice = st.selectbox(
        "Select LLM Backend",
        [
            "🦙 Ollama (Local Qwen2.5 / Llama)",
            "⚡ Built-in Grounded Synthesizer (Zero-Setup & Free)",
            "💎 Google Gemini (Free Tier Cloud)",
        ],
        index=0,
        help="Choose between completely free local Ollama, built-in in-memory synthesizer, or Google Gemini.",
    )

    selected_provider = "ollama"
    selected_model = OLLAMA_MODEL_NAME
    active_api_key = None

    if "Ollama" in provider_choice:
        selected_provider = "ollama"
        selected_model = st.text_input(
            "Ollama Model Name",
            value=OLLAMA_MODEL_NAME,
            help="Common models: qwen2.5, qwen2.5:7b, qwen2.5:3b, qwen2.5:1.5b, llama3.2",
        )
        ollama_url = st.text_input(
            "Ollama URL",
            value=OLLAMA_BASE_URL,
            help="Default: http://localhost:11434",
        )

        # Quick connection probe
        is_up, available_models = is_ollama_available(ollama_url)
        if is_up:
            st.success(f"Ollama connected! Found {len(available_models)} model(s) ✅", icon="🟢")
        else:
            st.info(
                "Ollama not detected on localhost.\n\n"
                "To start: `ollama run qwen2.5`\n\n"
                "*Note: If Ollama is offline, the app automatically uses the Built-in Free Synthesizer!*",
                icon="💡",
            )

    elif "Built-in" in provider_choice:
        selected_provider = "builtin"
        selected_model = "semantic-synthesizer"
        st.success("Built-in Synthesizer active! 100% Free, runs locally with zero setup.", icon="⚡")

    else:
        selected_provider = "gemini"
        selected_model = st.text_input("Gemini Model", value=GEMINI_MODEL_NAME)
        env_key = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Enter Gemini API Key",
            value=env_key if env_key and env_key != "your_gemini_api_key_here" else "",
            type="password",
            help="Get a free key from https://aistudio.google.com/ and paste here.",
        )
        active_api_key = api_key_input.strip() if api_key_input else None
        if active_api_key:
            st.success("Gemini API Key detected ✅", icon="🔒")
        else:
            st.warning("No API Key entered. Will fall back to Built-in Synthesizer.", icon="⚠️")

    st.markdown("---")
    st.subheader("📚 Knowledge Base Status")

    stats = vector_store.get_collection_stats()
    indexed_docs = vector_store.get_indexed_documents()

    col1, col2 = st.columns(2)
    col1.metric("Indexed Chunks", stats["points_count"])
    col2.metric("Documents", len(indexed_docs))

    if indexed_docs:
        with st.expander("Active Documents", expanded=False):
            for doc in indexed_docs:
                st.caption(f"📄 {doc}")

    st.markdown("---")
    st.subheader("🚀 Quick Setup Actions")

    if st.button("📥 Load Sample Corporate PDF", use_container_width=True):
        with st.spinner("Generating and indexing 4-page sample corporate report..."):
            pdf_path = generate_sample_benchmark_pdf()
            chunks = ingest_pdf(pdf_path)
            vector_store.index_chunks(chunks)
            st.success(f"Indexed {len(chunks)} chunks from sample_corporate_report.pdf!")
            st.rerun()

    if st.button("🗑️ Reset Database", use_container_width=True, type="secondary"):
        vector_store.clear()
        st.session_state.last_result = None
        st.session_state.chat_history = []
        st.info("Vector database cleared.")
        st.rerun()

    st.markdown("---")
    st.caption(f"**Embedding:** `{EMBEDDING_MODEL_NAME}` (Local)")
    st.caption(f"**Vector Store:** Qdrant Local (Cosine)")


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_retrieval, tab_trace, tab_eval = st.tabs([
    "💬 Chat & Q&A",
    "🔍 Retrieval Inspection",
    "🧭 Agent Trace",
    "📊 Benchmark Evaluation",
])


# =============================================================================
# TAB 1: Chat & Q&A
# =============================================================================
with tab_chat:
    st.header("Document Q&A with Adaptive Agentic RAG")
    st.markdown(
        f"Active LLM Backend: **`{selected_provider.upper()}`** "
        f"({selected_model}) | Powered by **Local Sentence Transformers & Qdrant**."
    )

    # Document Uploader
    with st.expander("📤 Upload Custom PDF Documents", expanded=len(indexed_docs) == 0):
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Index Uploaded Documents"):
            for file in uploaded_files:
                try:
                    with st.spinner(f"Processing and chunking '{file.name}'..."):
                        file_bytes = file.read()
                        chunks = ingest_pdf(file_bytes, doc_name=file.name)
                        count = vector_store.index_chunks(chunks)
                        st.success(f"Successfully indexed **{file.name}** ({count} chunks)!")
                except Exception as e:
                    st.error(f"Error indexing {file.name}: {str(e)}")
            st.rerun()

    # Pre-built Example Questions
    st.markdown("##### 💡 Try an Example Question:")
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

    example_query = None
    if col_ex1.button("🟢 Simple Query\n(Direct Revenue)", use_container_width=True):
        example_query = "What was the company's total revenue in 2024?"
    if col_ex2.button("🟡 Complex Query\n(Comparative & Causal)", use_container_width=True):
        example_query = "Why did North American revenue increase and how did European supply chain delays affect regional profitability?"
    if col_ex3.button("🔵 Ambiguous Query\n(Needs Rewriting)", use_container_width=True):
        example_query = "What are the risks?"
    if col_ex4.button("🔴 Unanswerable Query\n(Tests Hallucination)", use_container_width=True):
        example_query = "What was the company's annual revenue in fiscal year 1995?"

    # Query Input Box
    query_input = st.text_input(
        "Ask a question about the uploaded documents:",
        value=example_query if example_query else "",
        placeholder="e.g., What was the operating margin in 2024?",
    )

    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 4])
    submit_adaptive = col_btn1.button("🚀 Ask (Adaptive Agentic)", type="primary", use_container_width=True)
    submit_baseline = col_btn2.button("⚡ Force Baseline RAG", type="secondary", use_container_width=True)

    if (submit_adaptive or submit_baseline) and query_input:
        if stats["points_count"] == 0:
            st.error("⚠️ No documents have been indexed yet. Please upload a PDF or click 'Load Sample Corporate PDF' in the sidebar!")
        else:
            with st.spinner("Processing question..."):
                if submit_baseline:
                    # Run standard Baseline RAG
                    answer_obj = run_baseline_rag(
                        query=query_input,
                        vector_store=vector_store,
                        top_k=TOP_K,
                        provider=selected_provider,
                        api_key=active_api_key,
                        model_name=selected_model,
                    )
                    result_data = {
                        "query": query_input,
                        "mode": "Baseline RAG",
                        "classification": "SIMPLE",
                        "classification_reason": "Baseline mode forced manually.",
                        "rewritten_query": None,
                        "sub_questions": [],
                        "agentic_activated": False,
                        "answer": answer_obj.answer,
                        "citations": answer_obj.citations,
                        "retrieved_chunks": answer_obj.retrieved_chunks,
                        "evidence_sufficient": answer_obj.is_sufficient,
                        "provider": answer_obj.provider,
                        "trace": [
                            {"step": "Mode", "details": "Baseline RAG (Single-shot retrieval & generation)"},
                            {"step": "Retrieval", "chunks_retrieved_count": len(answer_obj.retrieved_chunks)},
                            {"step": "Generation", "answer_preview": answer_obj.answer[:120]},
                        ],
                    }
                else:
                    # Run Adaptive Agentic RAG
                    result_data = run_agentic_rag(
                        query=query_input,
                        vector_store=vector_store,
                        provider=selected_provider,
                        api_key=active_api_key,
                        model_name=selected_model,
                    )
                    result_data["mode"] = "Adaptive Agentic RAG"

                st.session_state.last_result = result_data
                st.session_state.chat_history.append(result_data)

    # Display Current Query Result
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")

        # Classification Badge Bar
        cls_color = {
            "SIMPLE": "🟢",
            "COMPLEX": "🟡",
            "AMBIGUOUS": "🔵",
        }.get(res["classification"], "⚪")

        badge_col1, badge_col2, badge_col3 = st.columns([2, 3, 2])
        badge_col1.markdown(f"**Classification:** {cls_color} `{res['classification']}`")
        badge_col2.markdown(f"**Agentic RAG Active:** {'✅ Yes (Multi-step)' if res.get('agentic_activated') else '⚡ No (Direct RAG)'}")
        badge_col3.markdown(f"**Evidence Sufficient:** {'✅ Yes' if res.get('evidence_sufficient') else '⚠️ No'}")

        # Rewritten Query or Sub-questions Callouts
        if res.get("rewritten_query"):
            st.info(f"🔄 **Query Rewriter Activated:** Transformed ambiguous query into:\n\n> *\"{res['rewritten_query']}\"*")

        if res.get("sub_questions"):
            with st.expander("🧩 Query Decomposition (Sub-questions generated):", expanded=True):
                for idx, sq in enumerate(res["sub_questions"], 1):
                    st.markdown(f"**Q{idx}:** {sq}")

        # Final Answer Box
        st.markdown("### 📝 Grounded Answer")
        is_suff = res.get("evidence_sufficient", True)
        border_color = "#2e7d32" if is_suff else "#d32f2f"
        bg_color = "#f8f9fa" if is_suff else "#fff8f8"
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; color: #1e1e1e; padding: 20px; border-radius: 8px; border-left: 5px solid {border_color}; font-size: 16px; line-height: 1.6;">
            {res['answer']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Citations
        if res.get("citations"):
            st.markdown("##### 📌 Verified Citations:")
            c_cols = st.columns(len(res["citations"]))
            for idx, cit in enumerate(res["citations"]):
                c_cols[idx].markdown(f"🏷️ `{cit}`")
        elif not is_suff:
            st.warning("⚠️ No citations available: The uploaded documents do not contain evidence for this question.")


# =============================================================================
# TAB 2: Retrieval Inspection
# =============================================================================
with tab_retrieval:
    st.header("🔍 Retrieval Inspection")
    st.markdown("Inspect the exact text chunks retrieved from Qdrant and their semantic similarity scores.")

    if not st.session_state.last_result:
        st.info("Ask a question in the Chat tab first to inspect retrieved chunks.")
    else:
        last_res = st.session_state.last_result
        chunks: List[RetrievedChunk] = last_res.get("retrieved_chunks", [])

        st.caption(f"Showing **{len(chunks)} chunks** retrieved for query: *\"{last_res['query']}\"*")

        for idx, chunk in enumerate(chunks, 1):
            with st.container():
                st.markdown(f"#### Chunk #{idx}: {chunk.citation}")
                col_score, col_page, col_id = st.columns([2, 2, 3])

                col_score.metric("Cosine Similarity", f"{chunk.score:.4f}")
                col_page.metric("Page", chunk.page_number)
                col_id.caption(f"**Chunk ID:** `{chunk.chunk_id}`")

                st.progress(min(1.0, max(0.0, chunk.score)))
                st.text_area(
                    label="Chunk Text Content",
                    value=chunk.text,
                    height=120,
                    key=f"chunk_txt_{idx}",
                )
                st.markdown("---")


# =============================================================================
# TAB 3: Agent Trace
# =============================================================================
with tab_trace:
    st.header("🧭 Agent Execution Trace")
    st.markdown("Step-by-step trace of the LangGraph workflow and decision path.")

    if not st.session_state.last_result:
        st.info("Execute a question in the Chat tab to view the live agent trace.")
    else:
        last_res = st.session_state.last_result
        trace_steps: List[Dict[str, Any]] = last_res.get("trace", [])

        # Visual Flowchart
        st.markdown("### Agent Workflow Diagram")
        flowchart = """
        flowchart TD
            A[User Query] --> B[1. Query Analyzer]
            B --> C{Query Type}
            C -->|SIMPLE| D[2. Direct Retrieval]
            C -->|COMPLEX| E[2. Decompose Sub-questions]
            C -->|AMBIGUOUS| F[2. Query Rewrite]
            E --> D
            F --> D
            D --> G[3. Evidence Sufficiency Check]
            G --> H{Sufficient?}
            H -->|YES| I[4. Grounded Generation]
            H -->|NO & Retry=0| J[Refine Query & Retrieve Once More]
            J --> I
            H -->|NO & Retry>=1| K[Say Insufficient Evidence]
            I --> L[Final Grounded Answer + Citations]
            K --> L
        """
        render_mermaid(flowchart)

        st.markdown("### Step-by-Step Decision Log")
        for i, step in enumerate(trace_steps, 1):
            step_name = step.get("step", f"Step {i}")
            with st.expander(f"Step {i}: {step_name}", expanded=True):
                st.json(step)


# =============================================================================
# TAB 4: Benchmark Evaluation (Baseline vs Agentic)
# =============================================================================
with tab_eval:
    st.header("📊 Benchmark Evaluation: Baseline RAG vs Agentic RAG")
    st.markdown(
        "Run an empirical side-by-side evaluation against the ground-truth benchmark (`data/evaluation.json`). "
        "Calculates actual **Precision@K**, **Recall@K**, **Faithfulness**, and **Answer Correctness**."
    )

    try:
        eval_data = load_evaluation_dataset()
        st.caption(f"Loaded ground-truth dataset with **{len(eval_data)} benchmark questions**.")
    except Exception as e:
        st.error(f"Failed to load evaluation dataset: {e}")
        eval_data = []

    col_bench1, col_bench2 = st.columns([2, 4])
    sample_limit = col_bench1.slider("Questions to evaluate", min_value=3, max_value=len(eval_data), value=10)
    run_eval_btn = col_bench2.button("▶️ Run Ground-Truth Benchmark", type="primary", use_container_width=True)

    if run_eval_btn:
        if stats["points_count"] == 0:
            st.error("Please load the sample benchmark PDF first so the questions have indexed documents to query!")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            status_text.text("Running Baseline and Agentic RAG across evaluation items...")
            results = run_evaluation_benchmark(
                vector_store=vector_store,
                dataset=eval_data,
                provider=selected_provider,
                api_key=active_api_key,
                model_name=selected_model,
                limit=sample_limit,
            )
            progress_bar.progress(1.0)
            status_text.success("Benchmark completed successfully!")
            st.session_state.eval_results = results

    if st.session_state.eval_results:
        eval_res = st.session_state.eval_results
        summary = eval_res["summary"]
        b_sum = summary["baseline"]
        a_sum = summary["agentic"]

        st.markdown("---")
        st.subheader("🏆 Overall Evaluation Grades")

        grade_col1, grade_col2 = st.columns(2)
        with grade_col1:
            st.markdown(f"### Baseline RAG Grade: `{b_sum['grade']}`")
            st.metric("Overall Score", f"{b_sum['overall_score']:.1f} / 100")
            st.caption(f"Retrieval: {b_sum['retrieval_score']} | Generation: {b_sum['generation_score']}")

        with grade_col2:
            st.markdown(f"### Agentic RAG Grade: `{a_sum['grade']}`")
            st.metric("Overall Score", f"{a_sum['overall_score']:.1f} / 100", delta=f"{a_sum['overall_score'] - b_sum['overall_score']:+.1f}")
            st.caption(f"Retrieval: {a_sum['retrieval_score']} | Generation: {a_sum['generation_score']}")

        st.caption("Note: This grade is the project's internal benchmark evaluation grade, not an absolute measure of general AI quality.")

        st.markdown("### 📈 Side-by-Side Metric Comparison")

        # Metric Comparison Table
        metric_rows = [
            {"Metric": "Precision@K", "Baseline RAG": f"{b_sum['precision']:.1f}%", "Agentic RAG": f"{a_sum['precision']:.1f}%", "Difference": f"{a_sum['precision'] - b_sum['precision']:+.1f}%"},
            {"Metric": "Recall@K", "Baseline RAG": f"{b_sum['recall']:.1f}%", "Agentic RAG": f"{a_sum['recall']:.1f}%", "Difference": f"{a_sum['recall'] - b_sum['recall']:+.1f}%"},
            {"Metric": "Hit Rate@K", "Baseline RAG": f"{b_sum['hit_rate']:.1f}%", "Agentic RAG": f"{a_sum['hit_rate']:.1f}%", "Difference": f"{a_sum['hit_rate'] - b_sum['hit_rate']:+.1f}%"},
            {"Metric": "Faithfulness", "Baseline RAG": f"{b_sum['faithfulness']:.1f}%", "Agentic RAG": f"{a_sum['faithfulness']:.1f}%", "Difference": f"{a_sum['faithfulness'] - b_sum['faithfulness']:+.1f}%"},
            {"Metric": "Answer Relevance", "Baseline RAG": f"{b_sum['relevance']:.1f}%", "Agentic RAG": f"{a_sum['relevance']:.1f}%", "Difference": f"{a_sum['relevance'] - b_sum['relevance']:+.1f}%"},
            {"Metric": "Ground Truth Correctness", "Baseline RAG": f"{b_sum['correctness']:.1f}%", "Agentic RAG": f"{a_sum['correctness']:.1f}%", "Difference": f"{a_sum['correctness'] - b_sum['correctness']:+.1f}%"},
            {"Metric": "Retrieval Score (Aggregate)", "Baseline RAG": f"{b_sum['retrieval_score']:.1f}", "Agentic RAG": f"{a_sum['retrieval_score']:.1f}", "Difference": f"{a_sum['retrieval_score'] - b_sum['retrieval_score']:+.1f}"},
            {"Metric": "Generation Score (Aggregate)", "Baseline RAG": f"{b_sum['generation_score']:.1f}", "Agentic RAG": f"{a_sum['generation_score']:.1f}", "Difference": f"{a_sum['generation_score'] - b_sum['generation_score']:+.1f}"},
            {"Metric": "Overall Project Grade", "Baseline RAG": b_sum['grade'], "Agentic RAG": a_sum['grade'], "Difference": "Improved" if a_sum['overall_score'] >= b_sum['overall_score'] else "Equal"},
        ]

        st.table(metric_rows)

        # Per-question Detailed Inspection
        with st.expander("🔎 Per-Question Detailed Comparison", expanded=False):
            for d in eval_res.get("details", []):
                st.markdown(f"**Q{d['id']} [{d['type']}]:** {d['question']}")
                c_b, c_a = st.columns(2)
                with c_b:
                    st.caption("⚡ Baseline RAG")
                    st.markdown(f"> *{d['baseline']['answer'][:160]}...*")
                    st.caption(f"P@K: {d['baseline']['precision']} | R@K: {d['baseline']['recall']} | Correctness: {d['baseline']['correctness']}")
                with c_a:
                    st.caption(f"🤖 Agentic RAG ({d['agentic']['classification']})")
                    st.markdown(f"> *{d['agentic']['answer'][:160]}...*")
                    st.caption(f"P@K: {d['agentic']['precision']} | R@K: {d['agentic']['recall']} | Correctness: {d['agentic']['correctness']}")
                st.markdown("---")
