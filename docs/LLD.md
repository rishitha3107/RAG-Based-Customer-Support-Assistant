# Low-Level Design (LLD)
# RAG-Based Customer Support Assistant with LangGraph & HITL

**Version:** 1.0 | **Date:** April 22, 2026

---

## 1. Module-Level Design

### 1.1 Document Processing Module

```python
# src/document_processor.py
class DocumentProcessor:
    """Handles PDF loading and text chunking."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF and return raw Document objects with page metadata."""
        
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into retrieval-optimized chunks."""
        
    def process(self, pdf_path: str) -> List[Document]:
        """Full pipeline: load → chunk → return."""
```

**Design Decisions:**
- `RecursiveCharacterTextSplitter` chosen over `CharacterTextSplitter` because it respects semantic boundaries (paragraphs → sentences → words)
- 1000-char chunks ≈ 150-200 tokens — fits well within retriever context windows
- 200-char overlap prevents information loss at boundaries (e.g., a sentence split across two chunks)

---

### 1.2 Embedding Module

```python
# src/embedding_manager.py
class EmbeddingManager:
    """Manages embedding model and ChromaDB vector store."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_dir="./chroma_db", collection_name="support_kb"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create new ChromaDB collection from documents."""
        
    def load_vectorstore(self) -> Chroma:
        """Load existing ChromaDB collection from disk."""
        
    def get_collection_stats(self) -> dict:
        """Return collection metadata: doc count, etc."""
```

---

### 1.3 Retrieval Module

```python
# src/retriever.py
class DocumentRetriever:
    """Retrieves relevant documents from vector store."""
    
    def __init__(self, vectorstore: Chroma, top_k=4, score_threshold=0.3):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Return top-k docs with relevance scores above threshold."""
        
    def has_relevant_results(self, results: List[Tuple[Document, float]]) -> bool:
        """Check if retrieval returned meaningful results."""
```

---

### 1.4 LLM Handler Module

```python
# src/llm_handler.py
class LLMHandler:
    """Manages all LLM interactions: generation, classification, scoring."""
    
    def __init__(self, model_name="gemini-1.5-flash", api_key=None):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate customer support response from query + retrieved context."""
        
    def assess_confidence(self, query: str, context: str, response: str) -> float:
        """Score response confidence from 0.0 to 1.0."""
        
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify query intent. Returns (intent_label, confidence)."""
```

---

### 1.5 Intent Classifier Module

```python
# src/intent_classifier.py
class IntentClassifier:
    """Classifies user queries into actionable intents."""
    
    INTENTS = {
        "general_inquiry": "General product questions, feature inquiries",
        "technical_support": "Setup, troubleshooting, configuration issues", 
        "billing": "Payments, subscriptions, pricing, invoices",
        "complaint": "Negative feedback, service issues, dissatisfaction",
        "escalation_request": "Explicit request to speak with human agent"
    }
    
    def classify(self, query: str) -> dict:
        """Returns: {intent: str, confidence: float, reasoning: str}"""
```

---

### 1.6 Graph Execution Module

```python
# src/graph_workflow.py
class RAGGraphWorkflow:
    """LangGraph-based stateful workflow for RAG pipeline."""
    
    def __init__(self, retriever, llm_handler, hitl_manager):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledGraph:
        """Construct the state graph with nodes and conditional edges."""
        
    # Node functions:
    def _classify_intent(self, state: GraphState) -> GraphState: ...
    def _retrieve_documents(self, state: GraphState) -> GraphState: ...
    def _generate_response(self, state: GraphState) -> GraphState: ...
    def _assess_quality(self, state: GraphState) -> GraphState: ...
    def _escalate_to_human(self, state: GraphState) -> GraphState: ...
    def _format_output(self, state: GraphState) -> GraphState: ...
    
    # Routing functions:
    def _route_after_intent(self, state: GraphState) -> str: ...
    def _route_after_assessment(self, state: GraphState) -> str: ...
    
    def run(self, query: str) -> GraphState:
        """Execute full workflow for a query."""
```

---

### 1.7 HITL Module

```python
# src/hitl_manager.py
class HITLManager:
    """Manages Human-in-the-Loop escalation and response collection."""
    
    def __init__(self, mode="cli"):  # "cli" or "streamlit"
        self.mode = mode
        self.pending_escalations = []
    
    def should_escalate(self, state: GraphState) -> Tuple[bool, str]:
        """Evaluate state and return (should_escalate, reason)."""
        
    def request_human_input(self, context: dict) -> str:
        """Display context to human and collect response."""
        
    def format_escalation_context(self, state: GraphState) -> dict:
        """Package state into human-readable escalation context."""
```

---

## 2. Data Structures

### 2.1 Document Schema

```python
# LangChain Document object
Document = {
    "page_content": str,      # Actual text content of the chunk
    "metadata": {
        "source": str,         # PDF file path
        "page": int,           # Original page number (0-indexed)
        "chunk_id": int,       # Sequential chunk identifier
        "chunk_size": int      # Character count of this chunk
    }
}
```

### 2.2 Chunk Format

```python
# Example chunk after processing
{
    "page_content": "To reset your SmartHome Hub, press and hold the reset button on the back of the device for 10 seconds. The LED will flash blue three times, indicating the device is resetting to factory defaults. After reset, open the TechCorp mobile app and follow the setup wizard to reconnect your device.",
    "metadata": {
        "source": "knowledge_base/techcorp_support_kb.pdf",
        "page": 3,
        "chunk_id": 12,
        "chunk_size": 347
    }
}
```

### 2.3 Embedding Representation

```python
# ChromaDB stores internally as:
{
    "ids": ["chunk_0", "chunk_1", ...],           # Unique chunk IDs
    "embeddings": [[0.023, -0.041, ...], ...],    # 384-dim float vectors
    "documents": ["chunk text...", ...],           # Original text
    "metadatas": [{"source": "...", "page": 3}, ...] # Metadata dicts
}
```

### 2.4 Graph State Object

```python
class GraphState(TypedDict):
    """Central state object passed through all graph nodes."""
    query: str                          # Original user query
    intent: str                         # Classified intent label
    intent_confidence: float            # Intent classification confidence
    retrieved_docs: List[Document]      # Retrieved document chunks
    retrieval_scores: List[float]       # Similarity scores for each doc
    context: str                        # Formatted context string for LLM
    response: str                       # Generated response text
    confidence: float                   # Response quality confidence (0-1)
    needs_escalation: bool              # Whether HITL is needed
    escalation_reason: str              # Why escalation was triggered
    human_response: Optional[str]       # Response from human agent (if escalated)
    source_documents: List[dict]        # Source references for transparency
    error: Optional[str]               # Error message if any step failed
```

### 2.5 Query-Response Schema

```python
# Final output returned to user
{
    "query": "How do I reset my SmartHome Hub?",
    "response": "To reset your SmartHome Hub, press and hold...",
    "confidence": 0.92,
    "intent": "technical_support",
    "sources": [
        {"page": 3, "excerpt": "To reset your SmartHome Hub..."},
        {"page": 4, "excerpt": "After reset, reconnect via..."}
    ],
    "escalated": False,
    "escalation_reason": None,
    "handled_by": "ai"  # or "human"
}
```

---

## 3. Workflow Design (LangGraph)

### 3.1 Node Definitions

| Node | Input State Fields | Processing | Output State Fields |
|------|-------------------|------------|-------------------|
| `classify_intent` | `query` | LLM classifies query intent | `intent`, `intent_confidence` |
| `retrieve_documents` | `query` | Embed query → ChromaDB search | `retrieved_docs`, `retrieval_scores`, `context` |
| `generate_response` | `query`, `context`, `intent` | LLM generates response with prompt template | `response` |
| `assess_quality` | `query`, `context`, `response` | LLM evaluates response quality | `confidence`, `needs_escalation`, `escalation_reason` |
| `escalate_to_human` | Full state | Display context → collect human input | `human_response`, `response` (overwritten) |
| `format_output` | Full state | Package final response with sources | `source_documents`, final response dict |

### 3.2 Edge Definitions

```python
# Graph construction
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("generate_response", generate_response)
graph.add_node("assess_quality", assess_quality)
graph.add_node("escalate_to_human", escalate_to_human)
graph.add_node("format_output", format_output)

# Entry point
graph.set_entry_point("classify_intent")

# Conditional edge after intent classification
graph.add_conditional_edges(
    "classify_intent",
    route_after_intent,
    {
        "retrieve": "retrieve_documents",
        "escalate": "escalate_to_human"
    }
)

# Sequential edges
graph.add_edge("retrieve_documents", "generate_response")
graph.add_edge("generate_response", "assess_quality")

# Conditional edge after quality assessment
graph.add_conditional_edges(
    "assess_quality",
    route_after_assessment,
    {
        "output": "format_output",
        "escalate": "escalate_to_human"
    }
)

# Terminal edges
graph.add_edge("escalate_to_human", "format_output")
graph.add_edge("format_output", END)
```

### 3.3 State Transition Diagram

```
             ┌──────────────────────────────────────────────────────────┐
             │                    STATE MACHINE                         │
             │                                                          │
             │  State 1: INIT                                          │
             │  {query: "...", all others: null/default}               │
             │           │                                              │
             │           ▼                                              │
             │  State 2: INTENT_CLASSIFIED                             │
             │  {+ intent: "technical_support", intent_confidence: 0.9}│
             │           │                                              │
             │     ┌─────┴─────┐                                       │
             │     │ Routing   │                                       │
             │     └─────┬─────┘                                       │
             │    normal │        │ "escalation_request"               │
             │           ▼        ▼                                     │
             │  State 3: DOCS_RETRIEVED          State 6: ESCALATED   │
             │  {+ retrieved_docs, context}      {+ human_response}   │
             │           │                                │             │
             │           ▼                                │             │
             │  State 4: RESPONSE_GENERATED               │            │
             │  {+ response: "To reset..."}               │            │
             │           │                                │             │
             │           ▼                                │             │
             │  State 5: QUALITY_ASSESSED                 │            │
             │  {+ confidence: 0.92}                      │            │
             │           │                                │             │
             │     ┌─────┴─────┐                          │            │
             │     │ Routing   │                          │            │
             │     └─────┬─────┘                          │            │
             │   ≥0.7    │    <0.7                        │            │
             │           ▼        ▼                       │            │
             │           │   State 6: ESCALATED ──────────┘            │
             │           │                                │             │
             │           └────────────┬───────────────────┘            │
             │                        ▼                                │
             │  State 7: OUTPUT_FORMATTED                              │
             │  {+ source_documents, final response}                   │
             │                        │                                │
             │                        ▼                                │
             │                       END                               │
             └──────────────────────────────────────────────────────────┘
```

---

## 4. Conditional Routing Logic

### 4.1 Intent-Based Routing (Post-Classification)

```python
def route_after_intent(state: GraphState) -> str:
    """Route based on classified intent."""
    if state["intent"] == "escalation_request":
        return "escalate"       # User explicitly wants human
    return "retrieve"           # All other intents proceed to RAG
```

### 4.2 Confidence-Based Routing (Post-Assessment)

```python
def route_after_assessment(state: GraphState) -> str:
    """Route based on response quality assessment."""
    # Rule 1: Low confidence
    if state["confidence"] < 0.7:
        state["escalation_reason"] = f"Low confidence ({state['confidence']:.2f})"
        return "escalate"
    
    # Rule 2: No relevant documents retrieved
    if not state["retrieved_docs"] or len(state["retrieved_docs"]) == 0:
        state["escalation_reason"] = "No relevant documents found"
        return "escalate"
    
    # Rule 3: Complaint intent with low confidence
    if state["intent"] == "complaint" and state["confidence"] < 0.8:
        state["escalation_reason"] = "Complaint requires human attention"
        return "escalate"
    
    return "output"  # High confidence → return to user
```

### 4.3 Escalation Decision Matrix

| Condition | Threshold | Action | Reason |
|-----------|-----------|--------|--------|
| Explicit escalation request | Intent = `escalation_request` | Immediate HITL | User preference |
| Low response confidence | confidence < 0.7 | HITL after generation | Uncertain answer quality |
| No relevant documents | retrieved_docs is empty | HITL after retrieval | Knowledge base gap |
| Complaint with medium confidence | intent = `complaint` AND confidence < 0.8 | HITL after generation | Sensitive situation needs human empathy |
| LLM generation failure | Error in generate step | HITL fallback | System error recovery |

---

## 5. HITL Design

### 5.1 Escalation Triggers

```python
ESCALATION_RULES = [
    {
        "name": "explicit_request",
        "check": lambda s: s["intent"] == "escalation_request",
        "stage": "post_classification",
        "priority": "HIGH"
    },
    {
        "name": "low_confidence",
        "check": lambda s: s["confidence"] < 0.7,
        "stage": "post_assessment",
        "priority": "MEDIUM"
    },
    {
        "name": "empty_retrieval",
        "check": lambda s: len(s.get("retrieved_docs", [])) == 0,
        "stage": "post_retrieval",
        "priority": "MEDIUM"
    },
    {
        "name": "complaint_handling",
        "check": lambda s: s["intent"] == "complaint" and s["confidence"] < 0.8,
        "stage": "post_assessment",
        "priority": "HIGH"
    }
]
```

### 5.2 Human Response Flow

```
1. TRIGGER: Escalation condition met
       │
       ▼
2. PACKAGE CONTEXT:
   ┌─────────────────────────────────────────┐
   │ Escalation Report                       │
   │ ──────────────────                      │
   │ Query: "I was charged twice for..."     │
   │ Intent: complaint (confidence: 0.85)    │
   │ AI Response: "We apologize for..."      │
   │ Response Confidence: 0.45               │
   │ Reason: Low confidence                  │
   │ Retrieved Sources:                      │
   │   - Billing FAQ (page 6, score: 0.72)   │
   │   - Refund Policy (page 7, score: 0.68) │
   └─────────────────────────────────────────┘
       │
       ▼
3. COLLECT HUMAN INPUT:
   CLI:       stdin prompt with above context
   Streamlit: Dedicated HITL panel with text area
       │
       ▼
4. REINTEGRATE:
   state["human_response"] = human_input
   state["response"] = human_input  # Override AI response
   state["handled_by"] = "human"
       │
       ▼
5. CONTINUE GRAPH: → format_output → END
```

### 5.3 Reintegration Logic

```python
def escalate_to_human(state: GraphState) -> GraphState:
    context = {
        "query": state["query"],
        "intent": state["intent"],
        "ai_response": state.get("response", "No response generated"),
        "confidence": state.get("confidence", 0.0),
        "reason": state.get("escalation_reason", "Unknown"),
        "sources": [doc.page_content[:200] for doc in state.get("retrieved_docs", [])]
    }
    
    human_response = hitl_manager.request_human_input(context)
    
    return {
        **state,
        "human_response": human_response,
        "response": human_response,
        "needs_escalation": False,  # Resolved
        "handled_by": "human"
    }
```

---

## 6. API / Interface Design

### 6.1 CLI Interface

```
$ python app_cli.py

╔══════════════════════════════════════════════╗
║  TechCorp Customer Support Assistant (RAG)  ║
║  Type 'help' for commands | 'quit' to exit  ║
╚══════════════════════════════════════════════╝

You: How do I set up my SmartHome Hub?

🔍 Intent: technical_support (confidence: 0.93)
📄 Retrieved 4 relevant documents
🤖 Generating response...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer: To set up your SmartHome Hub:
1. Unbox and plug in the device...
2. Download the TechCorp app...
3. Follow the pairing wizard...

📊 Confidence: 94%  |  🏷️ Handled by: AI
📑 Sources: Page 3, Page 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.2 Input/Output Formats

**Input:**
```json
{
    "query": "string (user's natural language question)",
    "session_id": "optional string (for future session tracking)"
}
```

**Output:**
```json
{
    "status": "success | escalated | error",
    "response": "string (final answer)",
    "confidence": 0.92,
    "intent": "technical_support",
    "sources": [
        {"page": 3, "excerpt": "First 200 chars...", "score": 0.87}
    ],
    "handled_by": "ai | human",
    "escalation_reason": "null | string"
}
```

---

## 7. Error Handling

### 7.1 Error Matrix

| Error Scenario | Detection | Handling | User Impact |
|---------------|-----------|----------|-------------|
| **PDF load failure** | `PyPDFLoader` exception | Log error, return descriptive message | Ingestion fails with clear error |
| **Empty document** | Zero chunks after splitting | Skip file, warn user | Other documents still processed |
| **Embedding failure** | Model loading/inference error | Retry 2x, then fail with message | Ingestion paused |
| **ChromaDB unavailable** | Connection/read error | Create new collection or report error | Query fails gracefully |
| **No relevant chunks** | Empty retrieval results | Set `needs_escalation=True` | Auto-escalate to HITL |
| **LLM API failure** | API timeout/rate limit/error | Retry with exponential backoff (3 attempts) | Delayed response or HITL fallback |
| **LLM hallucination** | Confidence < 0.5 | Force escalation | Human provides answer |
| **Invalid query** | Empty/too short query | Return prompt for valid input | Immediate user feedback |
| **Graph execution error** | Any node throws exception | Catch at graph level, return error state | Graceful error message |

### 7.2 Error Handling Strategy

```python
# Centralized error wrapper for graph nodes
def safe_node_execution(node_func):
    def wrapper(state: GraphState) -> GraphState:
        try:
            return node_func(state)
        except RateLimitError:
            time.sleep(2)
            return node_func(state)  # Retry once
        except Exception as e:
            logger.error(f"Node {node_func.__name__} failed: {e}")
            return {
                **state,
                "error": str(e),
                "needs_escalation": True,
                "escalation_reason": f"System error: {str(e)}"
            }
    return wrapper
```

---

*End of Low-Level Design Document*
