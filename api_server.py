"""
FastAPI Backend Server for the RAG Customer Support Assistant.
Exposes REST endpoints for the React frontend.

Usage:
    uvicorn api_server:app --reload --port 8000
"""
import sys
import os
import uuid
import time
import json
import logging
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.hitl_manager import HITLManager
from src.graph_workflow import RAGGraphWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Customer Support API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ────────────────────────────────────────────
workflow = None
embedding_mgr = None
chat_sessions = {}


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class HITLResponse(BaseModel):
    session_id: str
    message_id: str
    human_response: str


class SettingsUpdate(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    enable_hitl: Optional[bool] = None


def initialize_system():
    """Initialize all RAG components."""
    global workflow, embedding_mgr
    try:
        embedding_mgr = EmbeddingManager()
        vectorstore = embedding_mgr.load_vectorstore()
        if vectorstore is None:
            logger.warning("No vectorstore found")
            return False
        stats = embedding_mgr.get_collection_stats()
        if stats.get("document_count", 0) == 0:
            logger.warning("Empty vectorstore")
            return False
        retriever = DocumentRetriever(vectorstore)
        llm_handler = LLMHandler()
        hitl_manager = HITLManager(mode="streamlit")
        workflow = RAGGraphWorkflow(retriever, llm_handler, hitl_manager)
        logger.info(f"System initialized with {stats['document_count']} documents")
        return True
    except Exception as e:
        logger.error(f"Init failed: {e}")
        return False


@app.on_event("startup")
async def startup():
    initialize_system()


@app.get("/api/status")
async def get_status():
    """Get system status."""
    stats = {}
    if embedding_mgr:
        stats = embedding_mgr.get_collection_stats()
    return {
        "ready": workflow is not None,
        "documents": stats.get("document_count", 0),
        "model": config.LLM_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "collection": stats.get("collection_name", "N/A"),
    }


@app.post("/api/query")
async def handle_query(req: QueryRequest):
    """Process a user query through the RAG pipeline."""
    if workflow is None:
        raise HTTPException(status_code=503, detail="System not initialized. Ingest documents first.")

    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"messages": [], "created_at": datetime.now().isoformat()}

    start_time = time.time()
    result = workflow.run(req.query)
    elapsed = time.time() - start_time

    msg_id = str(uuid.uuid4())
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "page": doc.get("page", "N/A"),
            "source": doc.get("source", "N/A"),
            "excerpt": doc.get("excerpt", ""),
        })

    response_data = {
        "message_id": msg_id,
        "session_id": session_id,
        "query": req.query,
        "response": result.get("response", "No response generated."),
        "confidence": result.get("confidence", 0),
        "intent": result.get("intent", "unknown"),
        "intent_confidence": result.get("intent_confidence", 0),
        "handled_by": result.get("handled_by", "ai"),
        "escalated": result.get("handled_by") == "human" or result.get("needs_escalation", False),
        "escalation_reason": result.get("escalation_reason", ""),
        "sources": sources,
        "doc_count": len(result.get("retrieved_docs", [])),
        "latency_ms": round(elapsed * 1000),
        "timestamp": datetime.now().isoformat(),
    }

    chat_sessions[session_id]["messages"].append({
        "id": msg_id,
        "role": "user",
        "content": req.query,
        "timestamp": datetime.now().isoformat(),
    })
    chat_sessions[session_id]["messages"].append({
        "id": str(uuid.uuid4()),
        "role": "assistant",
        **response_data,
    })

    return response_data


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF file."""
    global workflow, embedding_mgr
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(config.PDF_DIRECTORY, exist_ok=True)
    file_path = os.path.join(config.PDF_DIRECTORY, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        processor = DocumentProcessor()
        chunks = processor.process(file_path)
        if not embedding_mgr:
            embedding_mgr = EmbeddingManager()
        vectorstore = embedding_mgr.create_vectorstore(chunks)
        initialize_system()

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "total_documents": embedding_mgr.get_collection_stats().get("document_count", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hitl/respond")
async def hitl_respond(req: HITLResponse):
    """Submit a human agent response for an escalated query."""
    if req.session_id in chat_sessions:
        chat_sessions[req.session_id]["messages"].append({
            "id": str(uuid.uuid4()),
            "role": "human_agent",
            "content": req.human_response,
            "in_response_to": req.message_id,
            "timestamp": datetime.now().isoformat(),
        })
    return {
        "status": "resolved",
        "message": "Human response recorded successfully.",
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all chat sessions."""
    sessions = []
    for sid, data in chat_sessions.items():
        msgs = data["messages"]
        first_query = next((m["content"] for m in msgs if m["role"] == "user"), "New Chat")
        sessions.append({
            "id": sid,
            "title": first_query[:60] + ("..." if len(first_query) > 60 else ""),
            "message_count": len(msgs),
            "created_at": data.get("created_at", ""),
        })
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get messages for a session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_sessions[session_id]


@app.get("/api/documents")
async def list_documents():
    """List uploaded documents."""
    docs = []
    if os.path.exists(config.PDF_DIRECTORY):
        for f in os.listdir(config.PDF_DIRECTORY):
            if f.lower().endswith(".pdf"):
                path = os.path.join(config.PDF_DIRECTORY, f)
                docs.append({
                    "name": f,
                    "size_kb": round(os.path.getsize(path) / 1024, 1),
                })
    stats = {}
    if embedding_mgr:
        stats = embedding_mgr.get_collection_stats()
    return {
        "documents": docs,
        "total_chunks": stats.get("document_count", 0),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
