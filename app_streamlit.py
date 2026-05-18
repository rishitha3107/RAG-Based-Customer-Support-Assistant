"""
Streamlit Web UI for the RAG Customer Support Assistant.
Modern chat interface with HITL panel, source viewer, and system dashboard.

Usage:
    streamlit run app_streamlit.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import logging

import config
from src.embedding_manager import EmbeddingManager
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.hitl_manager import HITLManager
from src.graph_workflow import RAGGraphWorkflow
from ingest import ingest_pdf

logging.basicConfig(level=logging.WARNING)

# ─── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="TechCorp Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-active { background-color: #d4edda; color: #155724; }
    .badge-error { background-color: #f8d7da; color: #721c24; }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.9rem;
    }
    .escalation-banner {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False
    if "workflow" not in st.session_state:
        st.session_state.workflow = None
    if "embedding_mgr" not in st.session_state:
        st.session_state.embedding_mgr = None
    if "pending_escalation" not in st.session_state:
        st.session_state.pending_escalation = None
    if "escalation_count" not in st.session_state:
        st.session_state.escalation_count = 0


def initialize_system():
    """Initialize all RAG system components."""
    try:
        embedding_mgr = EmbeddingManager()
        vectorstore = embedding_mgr.load_vectorstore()

        if vectorstore is None:
            return False, "No knowledge base found. Please ingest PDFs first.", embedding_mgr

        stats = embedding_mgr.get_collection_stats()
        if stats.get("document_count", 0) == 0:
            return False, "Knowledge base is empty. Please ingest PDFs.", embedding_mgr

        retriever = DocumentRetriever(vectorstore)
        llm_handler = LLMHandler()
        hitl_manager = HITLManager(mode="streamlit")
        workflow = RAGGraphWorkflow(retriever, llm_handler, hitl_manager)

        st.session_state.workflow = workflow
        st.session_state.embedding_mgr = embedding_mgr
        st.session_state.system_ready = True

        return True, f"System ready ({stats['document_count']} documents)", embedding_mgr

    except Exception as e:
        return False, f"Initialization error: {str(e)}", None


# ─── Main App ────────────────────────────────────────────────
def main():
    init_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 TechCorp Customer Support Assistant</h1>
        <p>Powered by RAG • LangGraph • Gemini • ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ System Controls")

        # Initialize button
        if st.button("🔄 Initialize / Reload System", type="primary", use_container_width=True):
            with st.spinner("Initializing system..."):
                success, message, emgr = initialize_system()
                if success:
                    st.success(message)
                else:
                    st.error(message)

        # Auto-initialize on first load
        if not st.session_state.system_ready:
            with st.spinner("Loading system..."):
                success, message, emgr = initialize_system()
                if success:
                    st.sidebar.success(message)
                else:
                    st.sidebar.warning(message)

        st.divider()

        # PDF Upload
        st.subheader("📄 Knowledge Base")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and st.button("📥 Ingest PDF"):
            # Save uploaded file temporarily
            temp_path = os.path.join(config.PDF_DIRECTORY, uploaded_file.name)
            os.makedirs(config.PDF_DIRECTORY, exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    ingest_pdf(temp_path, reset=False)
                    st.success(f"✅ Ingested: {uploaded_file.name}")
                    # Reinitialize to pick up new docs
                    initialize_system()
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

        st.divider()

        # System Status
        st.subheader("📊 System Status")
        if st.session_state.embedding_mgr:
            stats = st.session_state.embedding_mgr.get_collection_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("document_count", 0))
            with col2:
                st.metric("Escalations", st.session_state.escalation_count)

            status = "🟢 Active" if st.session_state.system_ready else "🔴 Offline"
            st.markdown(f"**Status:** {status}")
            st.markdown(f"**LLM:** {config.LLM_MODEL_NAME}")
            st.markdown(f"**Embeddings:** {config.EMBEDDING_MODEL_NAME}")
        else:
            st.warning("System not initialized")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_escalation = None
            st.rerun()

    # ─── Chat Interface ──────────────────────────────────────
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                cols = st.columns(4)
                with cols[0]:
                    st.caption(f"🏷️ {meta.get('intent', 'N/A')}")
                with cols[1]:
                    conf = meta.get('confidence', 0)
                    emoji = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.5 else "🔴"
                    st.caption(f"{emoji} {conf:.0%}")
                with cols[2]:
                    st.caption(f"📄 {meta.get('doc_count', 0)} sources")
                with cols[3]:
                    handler = meta.get('handled_by', 'ai')
                    st.caption(f"{'🤖' if handler == 'ai' else '👤'} {handler}")

                # Expandable sources
                sources = meta.get("sources", [])
                if sources:
                    with st.expander("📑 View Sources"):
                        for src in sources:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Page {src.get('page', '?')}</strong> — {src.get('source', 'N/A')}<br>
                                <em>{src.get('excerpt', 'N/A')}</em>
                            </div>
                            """, unsafe_allow_html=True)

    # ─── HITL Escalation Panel ───────────────────────────────
    if st.session_state.pending_escalation:
        esc = st.session_state.pending_escalation
        st.markdown("""
        <div class="escalation-banner">
            <h3>🚨 Escalation Required</h3>
            <p>The AI needs human assistance with this query.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Escalation Details", expanded=True):
            st.write(f"**Query:** {esc.get('query', 'N/A')}")
            st.write(f"**Reason:** {esc.get('escalation_reason', 'N/A')}")
            st.write(f"**Intent:** {esc.get('intent', 'N/A')}")
            if esc.get('response'):
                st.write(f"**AI's Attempt:** {esc.get('response', 'N/A')}")

        human_response = st.text_area(
            "👤 Human Agent Response:",
            placeholder="Type your response to the customer...",
            key="hitl_response"
        )

        if st.button("✅ Submit Human Response", type="primary"):
            if human_response.strip():
                # Add human response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "avatar": "👤",
                    "content": f"**[Human Agent]** {human_response}",
                    "metadata": {
                        "intent": esc.get("intent", "N/A"),
                        "confidence": 1.0,
                        "doc_count": 0,
                        "handled_by": "human",
                        "sources": []
                    }
                })
                st.session_state.pending_escalation = None
                st.session_state.escalation_count += 1
                st.rerun()
            else:
                st.warning("Please enter a response before submitting.")

    # ─── Chat Input ──────────────────────────────────────────
    if query := st.chat_input("Ask a question about TechCorp products & services..."):
        if not st.session_state.system_ready:
            st.error("⚠️ System not initialized. Click 'Initialize / Reload System' in the sidebar.")
            return

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "avatar": "👤",
            "content": query
        })
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Process query
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Processing your query..."):
                # Override HITL to not block in Streamlit
                workflow = st.session_state.workflow
                workflow.hitl_manager.mode = "streamlit"

                # Run workflow but intercept escalation
                result = workflow.run(query)

                if result.get("handled_by") == "human" or result.get("needs_escalation"):
                    # Store escalation for HITL panel
                    st.session_state.pending_escalation = result
                    st.markdown("🚨 **This query requires human assistance.** Please see the escalation panel below.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "avatar": "🤖",
                        "content": "🚨 **This query requires human assistance.** Please see the escalation panel below.",
                        "metadata": {
                            "intent": result.get("intent", "N/A"),
                            "confidence": result.get("confidence", 0),
                            "doc_count": len(result.get("retrieved_docs", [])),
                            "handled_by": "escalated",
                            "sources": result.get("source_documents", [])
                        }
                    })
                else:
                    response = result.get("response", "No response generated.")
                    st.markdown(response)

                    metadata = {
                        "intent": result.get("intent", "N/A"),
                        "confidence": result.get("confidence", 0),
                        "doc_count": len(result.get("retrieved_docs", [])),
                        "handled_by": result.get("handled_by", "ai"),
                        "sources": result.get("source_documents", [])
                    }

                    # Show inline metadata
                    cols = st.columns(4)
                    with cols[0]:
                        st.caption(f"🏷️ {metadata['intent']}")
                    with cols[1]:
                        conf = metadata['confidence']
                        emoji = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.5 else "🔴"
                        st.caption(f"{emoji} {conf:.0%}")
                    with cols[2]:
                        st.caption(f"📄 {metadata['doc_count']} sources")
                    with cols[3]:
                        st.caption(f"🤖 {metadata['handled_by']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "avatar": "🤖",
                        "content": response,
                        "metadata": metadata
                    })

                st.rerun()


if __name__ == "__main__":
    main()
