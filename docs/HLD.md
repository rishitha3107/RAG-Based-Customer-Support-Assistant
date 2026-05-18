# High-Level Design (HLD)
# RAG-Based Customer Support Assistant with LangGraph & Human-in-the-Loop

**Version:** 1.0  
**Date:** April 22, 2026  
**Author:** System Architect  
**Status:** Production Design

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Component Descriptions](#3-component-descriptions)
4. [Data Flow](#4-data-flow)
5. [Technology Choices](#5-technology-choices)
6. [Scalability Considerations](#6-scalability-considerations)

---

## 1. System Overview

### 1.1 Problem Definition

Enterprise customer support operations face three critical challenges:

1. **Knowledge Fragmentation** — Support documentation lives in scattered PDFs, wikis, and manuals. Agents spend 40-60% of their time searching for answers rather than solving problems.
2. **Inconsistent Response Quality** — Different agents provide varying quality answers for the same question, leading to poor customer satisfaction scores.
3. **Scalability Bottleneck** — Human agents cannot scale linearly with query volume. Peak times create long wait queues, and hiring/training new agents is expensive and slow.

Traditional chatbots fail because they rely on rigid keyword matching or pre-defined decision trees that cannot handle the nuance and variability of real customer queries. Pure LLM-based solutions hallucinate answers when they lack specific domain knowledge.

**RAG solves this** by grounding LLM responses in actual company documentation — the model retrieves relevant context before generating, dramatically reducing hallucination while maintaining natural language fluency.

### 1.2 Scope of System

**In Scope:**
- PDF knowledge base ingestion with intelligent chunking
- Semantic embedding and vector storage in ChromaDB
- Context-aware retrieval with relevance scoring
- LLM-powered response generation with custom prompt engineering
- LangGraph-based stateful workflow orchestration
- Intent classification and conditional routing
- Human-in-the-Loop (HITL) escalation for low-confidence or complex queries
- CLI and Streamlit web interfaces
- Confidence scoring and quality assessment

**Out of Scope (Future Phases):**
- Multi-language support
- Real-time document sync from external CMS
- User authentication and role-based access
- Production deployment (Kubernetes, cloud infrastructure)
- Conversation memory across sessions (stateless per query in v1)

### 1.3 Key Design Principles

| Principle | Application |
|-----------|------------|
| **Modularity** | Each pipeline stage is an independent, replaceable module |
| **Fail-Safe** | Unknown or low-confidence queries escalate to humans rather than generating wrong answers |
| **Transparency** | Every response includes source documents and confidence scores |
| **Extensibility** | New document types, LLMs, or vector stores can be swapped without redesigning the system |

---

## 2. Architecture Diagram

### 2.1 System Architecture (Text Representation)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE LAYER                               │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────────┐   │
│  │   CLI Interface      │    │   Streamlit Web UI                          │   │
│  │   (app_cli.py)       │    │   (app_streamlit.py)                        │   │
│  │                      │    │   • Chat Interface    • PDF Upload          │   │
│  │   • Terminal I/O     │    │   • HITL Panel        • Source Viewer       │   │
│  │   • Colored Output   │    │   • Confidence Gauge  • Status Dashboard   │   │
│  └──────────┬───────────┘    └─────────────────┬────────────────────────────┘   │
│             │                                  │                                │
│             └──────────────┬───────────────────┘                                │
└────────────────────────────┼────────────────────────────────────────────────────┘
                             │ User Query
                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH ORCHESTRATION LAYER                            │
│                                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐   │
│  │  CLASSIFY    │───▶│  RETRIEVE     │───▶│  GENERATE   │───▶│  ASSESS       │   │
│  │  INTENT      │    │  DOCUMENTS    │    │  RESPONSE   │    │  QUALITY      │   │
│  └──────┬──────┘    └──────────────┘    └─────────────┘    └───────┬───────┘   │
│         │                                                          │            │
│         │ (explicit escalation)                    (low confidence) │            │
│         │         ┌───────────────────┐                             │            │
│         └────────▶│  ESCALATE TO      │◀────────────────────────────┘            │
│                   │  HUMAN (HITL)     │                                          │
│                   └────────┬──────────┘                                          │
│                            │                                                    │
│                   ┌────────▼──────────┐                                          │
│                   │  FORMAT OUTPUT    │──────▶ Final Response to User            │
│                   └───────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                             │
                    Retrieval Requests
                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DATA & INTELLIGENCE LAYER                                │
│                                                                                 │
│  ┌─────────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │  EMBEDDING MODEL    │  │  VECTOR STORE     │  │  LLM (Google Gemini)    │   │
│  │  (MiniLM-L6-v2)    │  │  (ChromaDB)       │  │  (gemini-1.5-flash)     │   │
│  │                     │  │                    │  │                          │   │
│  │  • 384-dim vectors  │  │  • Persistent     │  │  • Response Generation  │   │
│  │  • Local inference  │  │  • HNSW index     │  │  • Intent Classification│   │
│  │  • Batch encoding   │  │  • Metadata filter│  │  • Confidence Scoring   │   │
│  └─────────────────────┘  └──────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                             ▲
                    Document Ingestion
                             │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT INGESTION PIPELINE                              │
│                                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  PDF     │───▶│  TEXT         │───▶│  CHUNK       │───▶│  EMBED &        │   │
│  │  SOURCE  │    │  EXTRACTION   │    │  (Recursive) │    │  STORE          │   │
│  └──────────┘    └──────────────┘    └──────────────┘    └─────────────────┘   │
│                                                                                 │
│  PyPDFLoader      Raw text +          1000 chars/chunk    HuggingFace →        │
│                   page metadata       200 char overlap    ChromaDB             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Map

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       ▼
┌──────────────┐     ┌──────────────┐
│ Intent       │────▶│ Escalation?  │──── YES ──▶ HITL Module
│ Classifier   │     │ (explicit)   │
└──────────────┘     └──────┬───────┘
                        NO  │
                            ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ Query        │────▶│ ChromaDB     │
                     │ Embedder     │     │ Similarity   │
                     │              │     │ Search       │
                     └──────────────┘     └──────┬───────┘
                                                 │ Top-K Docs
                                                 ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ Prompt       │────▶│ LLM          │
                     │ Constructor  │     │ (Gemini)     │
                     └──────────────┘     └──────┬───────┘
                                                 │ Response
                                                 ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ Confidence   │────▶│ Route        │
                     │ Assessor     │     │ Decision     │
                     └──────────────┘     └──────┬───────┘
                                           ┌─────┴─────┐
                                      ≥0.7 │           │ <0.7
                                           ▼           ▼
                                    ┌──────────┐ ┌──────────┐
                                    │ Format & │ │ HITL     │
                                    │ Return   │ │ Escalate │
                                    └──────────┘ └──────────┘
```

---

## 3. Component Descriptions

### 3.1 Document Loader

| Aspect | Detail |
|--------|--------|
| **Purpose** | Extract raw text content from PDF knowledge base documents |
| **Implementation** | `PyPDFLoader` from LangChain |
| **Input** | PDF file path(s) |
| **Output** | List of `Document` objects with text content and metadata (page number, source file) |
| **Why PyPDF** | Lightweight, no external dependencies (unlike Tika/Unstructured), handles standard PDFs well. For production with complex layouts, upgrade to `Unstructured` |

### 3.2 Chunking Strategy

| Aspect | Detail |
|--------|--------|
| **Purpose** | Break documents into semantically meaningful, retrieval-optimized segments |
| **Implementation** | `RecursiveCharacterTextSplitter` |
| **Chunk Size** | 1000 characters |
| **Chunk Overlap** | 200 characters |
| **Separators** | `["\n\n", "\n", ". ", " ", ""]` (paragraph → sentence → word hierarchy) |
| **Why Recursive** | Preserves semantic boundaries better than naive fixed-size splitting. The hierarchy of separators ensures paragraphs stay intact when possible, falling back to sentences, then words only when necessary |
| **Why 1000/200** | 1000 chars ≈ 150-200 tokens — large enough for context, small enough for precise retrieval. 200-char overlap prevents information loss at chunk boundaries |

### 3.3 Embedding Model

| Aspect | Detail |
|--------|--------|
| **Purpose** | Convert text chunks and queries into dense vector representations for semantic search |
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Dimensions** | 384 |
| **Execution** | Local inference (no API calls for embeddings) |
| **Why This Model** | Best balance of quality and speed for English text. 5x faster than larger models with only ~2% quality loss on retrieval benchmarks. Zero API cost. Production alternative: OpenAI `text-embedding-3-small` for higher quality |

### 3.4 Vector Store (ChromaDB)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Store, index, and query document embeddings for fast similarity search |
| **Implementation** | ChromaDB with persistent storage |
| **Index Type** | HNSW (Hierarchical Navigable Small World) — default |
| **Distance Metric** | Cosine similarity |
| **Storage** | Local disk (`./chroma_db/` directory) |
| **Why ChromaDB** | Purpose-built for AI applications, zero-config setup, native LangChain integration, supports metadata filtering, persistent storage out of the box. For production scale: migrate to Pinecone or Weaviate |

### 3.5 Retriever

| Aspect | Detail |
|--------|--------|
| **Purpose** | Find the most relevant document chunks for a given user query |
| **Method** | Semantic similarity search with score threshold |
| **Top-K** | 4 documents (configurable) |
| **Relevance Threshold** | 0.3 minimum similarity score (below this, chunks are discarded) |
| **Output** | Ranked list of `(document, score)` tuples |
| **Why K=4** | Provides sufficient context without overwhelming the LLM's context window. Balances recall (getting all relevant info) with precision (avoiding noise) |

### 3.6 LLM (Large Language Model)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Generate natural language responses, classify intents, and assess confidence |
| **Model** | Google Gemini `gemini-1.5-flash` |
| **Temperature** | 0.3 for response generation (low creativity, high accuracy); 0.0 for classification (deterministic) |
| **Max Tokens** | 1024 for responses, 256 for classification |
| **Why Gemini Flash** | Free tier with generous rate limits (15 RPM, 1M TPM), fast inference (~1-2s), strong instruction following. Cost-effective for an internship project. Production alternative: GPT-4o or Claude for higher quality |

### 3.7 Graph Workflow Engine (LangGraph)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Orchestrate the multi-step RAG pipeline as a stateful, conditional workflow |
| **Implementation** | `StateGraph` from LangGraph |
| **Nodes** | 6 processing nodes (classify → retrieve → generate → assess → escalate → format) |
| **Edges** | Conditional edges for intent-based and confidence-based routing |
| **State** | `TypedDict` carrying query, intent, documents, response, confidence, escalation flags |
| **Why LangGraph** | Unlike sequential chains, LangGraph supports conditional branching, cycles, and state persistence. Critical for HITL (pause/resume execution) and intent-based routing. LangChain's LCEL cannot handle these patterns |

### 3.8 Routing Layer

| Aspect | Detail |
|--------|--------|
| **Purpose** | Make intelligent decisions about query processing path |
| **Decision Points** | 2 conditional edges in the graph |
| **Routing Rules** | Intent-based (post-classification) and confidence-based (post-assessment) |
| **Intents Supported** | `general_inquiry`, `technical_support`, `billing`, `complaint`, `escalation_request` |
| **Escalation Triggers** | Explicit request, confidence < 0.7, no relevant documents, complaint intent |

### 3.9 HITL Module

| Aspect | Detail |
|--------|--------|
| **Purpose** | Seamlessly transfer complex or uncertain queries to human agents |
| **Trigger Conditions** | (1) Confidence < 0.7, (2) Empty retrieval results, (3) User says "talk to human", (4) Complaint with negative sentiment |
| **Human Interface** | CLI: stdin prompt with context display; Streamlit: dedicated HITL panel with query context, attempted AI response, and input form |
| **Reintegration** | Human response replaces AI response in graph state, flows to format_output node |
| **Context Provided to Human** | Original query, retrieved documents, AI's attempted response, escalation reason, confidence score |

---

## 4. Data Flow

### 4.1 Document Ingestion Flow (Offline / Batch)

```
PDF File(s)
    │
    ▼
┌─────────────────────────────────┐
│ 1. LOAD                        │
│    PyPDFLoader reads PDF        │
│    Output: Raw text per page    │
│    + metadata (page_num, source)│
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 2. CHUNK                        │
│    RecursiveCharacterTextSplitter│
│    1000 chars, 200 overlap      │
│    Output: ~50-100 chunks       │
│    (for a 10-page PDF)          │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 3. EMBED                        │
│    MiniLM-L6-v2 encodes chunks  │
│    Output: 384-dim vectors      │
│    Batch processing for speed   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 4. STORE                        │
│    ChromaDB persists vectors    │
│    + original text + metadata   │
│    HNSW index built             │
└─────────────────────────────────┘
```

### 4.2 Query Processing Flow (Online / Real-Time)

```
User Query: "How do I reset my SmartHome Hub?"
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. CLASSIFY INTENT                                              │
│    LLM analyzes query → intent: "technical_support"             │
│    Not an escalation request → proceed to retrieval              │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. RETRIEVE DOCUMENTS                                           │
│    Query embedded → 384-dim vector                              │
│    ChromaDB similarity search → top 4 chunks returned           │
│    Chunks about "SmartHome Hub reset procedure" ranked highest   │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. GENERATE RESPONSE                                            │
│    Prompt = System Instructions + Retrieved Context + Query     │
│    LLM generates: "To reset your SmartHome Hub: 1) Press and   │
│    hold the reset button for 10 seconds. 2) Wait for the LED   │
│    to blink blue. 3) Reconnect via the mobile app..."          │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. ASSESS QUALITY                                               │
│    LLM evaluates: Does the response answer the query?           │
│    Are sources relevant? Confidence: 0.92 → HIGH                │
│    Route: → format_output (no escalation needed)                │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. FORMAT OUTPUT                                                │
│    Response + Sources + Confidence Score → User                 │
│    "Based on our knowledge base (Page 4): To reset your        │
│    SmartHome Hub..."  [Confidence: 92%] [Sources: 3 docs]      │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Escalation Flow (HITL Path)

```
User Query: "I've been charged twice and your product broke my router"
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ CLASSIFY: intent = "complaint" → proceed to retrieve │
└──────┬───────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────┐
│ RETRIEVE: finds partial matches (billing + tech)     │
└──────┬───────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────┐
│ GENERATE: Attempts response but combines two topics  │
└──────┬───────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────┐
│ ASSESS: Confidence = 0.45 (LOW)                      │
│ Reason: Multi-topic complaint, partial context       │
│ Route: → ESCALATE TO HUMAN                           │
└──────┬───────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────┐
│ HITL MODULE:                                         │
│ Display to agent:                                    │
│   - Query: "charged twice... broke my router"        │
│   - AI Attempt: [partial response]                   │
│   - Retrieved Docs: [billing FAQ, troubleshooting]   │
│   - Reason: Low confidence (0.45)                    │
│                                                      │
│ Human agent types response → injected into state     │
└──────┬───────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────┐
│ FORMAT OUTPUT: Human response → User                 │
│ "[Handled by Support Agent] We apologize for..."     │
└──────────────────────────────────────────────────────┘
```

---

## 5. Technology Choices

### 5.1 Technology Stack Summary

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.10+ | ML/AI ecosystem standard, LangChain native support |
| LLM Framework | LangChain | 0.2+ | Modular abstractions for LLM apps, massive integration library |
| Workflow Engine | LangGraph | 0.2+ | Stateful graphs with conditional edges, built for agent workflows |
| Vector Database | ChromaDB | 0.4+ | Zero-config, embedded, persistent, AI-native |
| Embedding Model | all-MiniLM-L6-v2 | - | Free, local, fast, 384-dim, strong retrieval quality |
| LLM | Google Gemini Flash | 1.5 | Free tier, fast, good instruction following |
| PDF Processing | PyPDF | 3.0+ | Lightweight, pure Python, no system dependencies |
| Web UI | Streamlit | 1.30+ | Rapid prototyping, built-in chat components, no frontend code needed |
| Config Management | python-dotenv | - | Secure API key management |

### 5.2 Why ChromaDB Over Alternatives

| Factor | ChromaDB | Pinecone | FAISS | Weaviate |
|--------|----------|----------|-------|----------|
| Setup Complexity | Zero-config | Cloud setup required | Manual index management | Server setup required |
| Persistence | Built-in disk storage | Cloud-managed | Manual save/load | Server-based |
| Metadata Filtering | Native support | Native support | Not supported | Native support |
| Cost | Free (open source) | Free tier limited | Free (open source) | Free (open source) |
| LangChain Integration | First-class | First-class | Good | Good |
| Best For | Prototyping, small-medium scale | Production, large scale | High-performance local | Production, hybrid search |

**Decision:** ChromaDB is chosen for its zero-configuration setup, native persistence, metadata filtering capabilities, and seamless LangChain integration. It allows the project to focus on RAG logic rather than infrastructure. For production deployment with millions of documents, migrate to Pinecone or Weaviate.

### 5.3 Why LangGraph Over Alternatives

| Approach | Capability | Limitation |
|----------|-----------|------------|
| **LangChain LCEL** | Sequential chains, simple branching | No cycles, no state persistence, no HITL pause/resume |
| **Custom Python** | Full control | No built-in state management, visualization, or checkpointing |
| **LangGraph** | Conditional edges, cycles, state persistence, HITL interrupts | Slightly higher complexity |
| **CrewAI/AutoGen** | Multi-agent orchestration | Overkill for single-agent RAG, less control over routing |

**Decision:** LangGraph is the only framework that natively supports all three requirements: conditional routing (intent-based decisions), stateful execution (carrying context through nodes), and HITL interrupts (pausing execution for human input). These are non-negotiable for the project's design.

### 5.4 Why Gemini Flash

- **Free tier**: 15 requests/minute, 1 million tokens/minute — sufficient for development and demo
- **Speed**: ~1-2 second latency per request (vs 3-5s for GPT-4)
- **Quality**: Strong instruction following for structured outputs (intent classification, confidence scoring)
- **Cost**: $0 for internship-scale usage
- **Trade-off**: Slightly lower reasoning quality than GPT-4o or Claude, acceptable for customer support domain

---

## 6. Scalability Considerations

### 6.1 Large Document Handling

| Challenge | Current Design | Scale-Up Strategy |
|-----------|---------------|-------------------|
| **Large PDFs (100+ pages)** | Process entire PDF at once | Batch processing with progress tracking; process 10 pages at a time |
| **Multiple PDFs** | Single PDF ingestion | Directory scanner with incremental ingestion; hash-based deduplication to avoid re-processing unchanged docs |
| **Document Updates** | Full re-ingestion | Implement document versioning; only re-embed changed chunks using content hashing |
| **Storage Growth** | Local ChromaDB | Migrate to managed vector DB (Pinecone) with automatic scaling; implement TTL for stale documents |

### 6.2 High Query Load

| Challenge | Current Design | Scale-Up Strategy |
|-----------|---------------|-------------------|
| **Concurrent Users** | Single-threaded CLI/Streamlit | Deploy behind FastAPI with async handlers; horizontal scaling with load balancer |
| **LLM Rate Limits** | 15 RPM (Gemini free tier) | Upgrade to paid tier or implement request queuing with exponential backoff; cache frequent query-response pairs |
| **Embedding Computation** | On-demand per query | Pre-compute embeddings for common queries; use GPU-accelerated inference for batch processing |
| **Vector Search Speed** | ChromaDB default HNSW | Tune HNSW parameters (ef_construction, M); for >1M vectors, switch to Pinecone or use FAISS with IVF index |

### 6.3 Latency Optimization

| Stage | Current Latency | Optimization |
|-------|----------------|-------------|
| **Query Embedding** | ~50ms (MiniLM local) | Already optimal; batch queries if applicable |
| **Vector Search** | ~10-20ms (ChromaDB) | Acceptable up to 100K vectors; tune HNSW params beyond that |
| **LLM Generation** | ~1-2s (Gemini Flash) | Implement streaming responses for perceived speed; cache common Q&A pairs with TTL |
| **Intent Classification** | ~1s (separate LLM call) | Combine with response generation in single prompt; or use lightweight local classifier (distilbert) |
| **Total Pipeline** | ~3-4s end-to-end | Target: <2s with caching + combined prompts; streaming reduces perceived latency to <500ms |

### 6.4 Production Deployment Architecture (Future State)

```
┌─────────────────────────────────────────────────────┐
│                   Load Balancer                      │
│                   (nginx/ALB)                        │
└────────────┬─────────────┬──────────────────────────┘
             │             │
     ┌───────▼───┐   ┌────▼────┐
     │ FastAPI   │   │ FastAPI  │    (Horizontal scaling)
     │ Worker 1  │   │ Worker 2 │
     └─────┬─────┘   └────┬────┘
           │               │
     ┌─────▼───────────────▼─────┐
     │      Redis Cache          │    (Query-response caching)
     └─────────────┬─────────────┘
                   │
     ┌─────────────▼─────────────┐
     │    Pinecone / Weaviate    │    (Managed vector DB)
     └───────────────────────────┘
```

---

*End of High-Level Design Document*
