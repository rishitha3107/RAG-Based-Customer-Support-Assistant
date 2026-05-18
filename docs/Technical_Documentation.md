# Technical Documentation
# RAG-Based Customer Support Assistant with LangGraph & HITL

**Version:** 1.0 | **Date:** April 22, 2026

---

## 1. Introduction

### 1.1 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an architecture pattern that enhances Large Language Model (LLM) outputs by grounding them in external, domain-specific knowledge. Instead of relying solely on the LLM's training data (which may be outdated, generic, or hallucinated), RAG retrieves relevant documents from a knowledge base and injects them as context into the LLM's prompt.

**The RAG Pipeline:**
```
User Query → Embed Query → Search Vector DB → Retrieve Top-K Documents
    → Construct Prompt (Query + Context) → LLM Generates Response
```

This approach combines the **factual accuracy** of information retrieval with the **natural language fluency** of generative models.

### 1.2 Why RAG is Needed

| Problem | Without RAG | With RAG |
|---------|------------|----------|
| **Hallucination** | LLM invents plausible but incorrect answers | Responses grounded in actual documents |
| **Stale Knowledge** | LLM training data has a cutoff date | Knowledge base can be updated anytime |
| **Domain Specificity** | Generic responses for specialized domains | Precise answers from company-specific docs |
| **Traceability** | No way to verify answer source | Every response includes source references |
| **Cost** | Fine-tuning is expensive ($1000s+) | RAG uses existing models with context injection |

### 1.3 Use Case: Customer Support

Customer support is an ideal RAG application because:

1. **Structured Knowledge Base** — Support documentation (FAQs, troubleshooting guides, policies) is well-organized and factual
2. **High Query Volume** — Companies handle thousands of repetitive queries daily
3. **Accuracy is Critical** — Wrong answers damage customer trust and can have legal implications (refund policies, warranties)
4. **Human Escalation is Natural** — Complex cases already require human intervention; HITL fits the existing workflow
5. **Measurable Impact** — Response time, accuracy, and escalation rate are quantifiable metrics

---

## 2. System Architecture Explanation

### 2.1 Four-Layer Architecture

The system is organized into four distinct layers, each with clear responsibilities:

**Layer 1: User Interface Layer**
- Accepts user queries via CLI or Streamlit web UI
- Displays formatted responses with confidence scores and source references
- Provides HITL interface for human agents during escalation
- Handles PDF upload for knowledge base ingestion

**Layer 2: Orchestration Layer (LangGraph)**
- Central brain of the system — manages the entire query lifecycle
- Implements a stateful directed graph with 6 processing nodes
- Makes routing decisions based on intent classification and confidence scoring
- Supports pause/resume for HITL escalation

**Layer 3: Intelligence Layer**
- Embedding model (MiniLM-L6-v2) converts text to semantic vectors
- ChromaDB stores and indexes document embeddings for fast similarity search
- Google Gemini LLM generates responses, classifies intents, and scores confidence
- All ML operations are abstracted behind clean interfaces for swappability

**Layer 4: Data Layer**
- PDF ingestion pipeline (load → chunk → embed → store)
- ChromaDB persistent storage on local disk
- Document metadata tracking (source file, page numbers, chunk IDs)

### 2.2 Component Interactions

```
User → [UI Layer] → Query enters system
                  → [Orchestration] classify_intent node
                  → [Intelligence] LLM classifies intent
                  → [Orchestration] routing decision
                  → [Intelligence] Embed query → ChromaDB search
                  → [Data Layer] ChromaDB returns top-K chunks
                  → [Intelligence] LLM generates response from context
                  → [Orchestration] assess_quality node
                  → [Intelligence] LLM scores confidence
                  → [Orchestration] routing decision (output or escalate)
                  → [UI Layer] → Response displayed to user
```

---

## 3. Design Decisions

### 3.1 Chunk Size: Why 1000 Characters with 200 Overlap

**Analysis of chunk size trade-offs:**

| Chunk Size | Pros | Cons |
|-----------|------|------|
| **256 chars** | High retrieval precision, finds exact answers | Loses paragraph context, LLM gets fragmented info |
| **512 chars** | Good balance for short-form content | May split mid-thought for complex topics |
| **1000 chars** ✅ | Preserves full paragraphs, rich context for LLM | Slightly lower retrieval precision |
| **2000 chars** | Maximum context per chunk | Too broad for similarity search, includes irrelevant content |

**Why 1000:** Customer support documents typically organize information in paragraph-length sections (150-250 words). A 1000-char chunk captures one complete concept (e.g., "How to reset SmartHome Hub") without mixing unrelated topics.

**Why 200 overlap:** Ensures sentences at chunk boundaries appear in both adjacent chunks. If a key instruction spans two chunks, the overlap guarantees it's fully captured in at least one.

### 3.2 Embedding Strategy

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

| Criteria | all-MiniLM-L6-v2 | OpenAI text-embedding-3-small | BGE-large-en |
|----------|-------------------|-------------------------------|-------------|
| Dimensions | 384 | 1536 | 1024 |
| Speed | ~14,000 sentences/sec | API-bound (~100/sec) | ~1,000 sentences/sec |
| Quality (MTEB) | 68.1 | 72.3 | 73.5 |
| Cost | Free (local) | $0.02/1M tokens | Free (local) |
| Setup | `pip install` | API key required | `pip install` (large download) |

**Decision:** MiniLM provides the best speed/quality/cost ratio for this project. 384 dimensions keep ChromaDB storage efficient. The ~4-point quality gap vs OpenAI embeddings is negligible for a focused customer support knowledge base.

### 3.3 Retrieval Approach

**Method:** Semantic similarity search with cosine distance

**Top-K = 4** rationale:
- K=1: Risk of missing relevant context if the best match isn't comprehensive
- K=4: Provides diverse context covering different aspects of the query
- K=10: Too much noise; irrelevant chunks dilute the context and confuse the LLM

**Score threshold = 0.3** rationale:
- Below 0.3: Retrieved chunks are essentially random — not semantically related to the query
- This threshold ensures only meaningfully related content reaches the LLM
- If all chunks score below 0.3, the system recognizes a knowledge gap and triggers escalation

### 3.4 Prompt Design

**Response Generation Prompt:**
```
You are a helpful, professional customer support assistant for TechCorp.
Answer the customer's question using ONLY the provided context.

RULES:
1. Be concise, friendly, and professional
2. If the context doesn't contain enough information to fully answer,
   say so honestly — do NOT make up information
3. Reference specific steps or details from the context
4. For technical issues, provide step-by-step instructions
5. For billing/policy questions, quote the relevant policy

CONTEXT:
{retrieved_context}

CUSTOMER QUESTION:
{query}

YOUR RESPONSE:
```

**Design rationale:**
- **"ONLY the provided context"** — prevents hallucination
- **"say so honestly"** — triggers low confidence on the assessment, leading to HITL
- **Role framing** ("customer support assistant") — keeps tone appropriate
- **Rules are numbered** — LLMs follow structured instructions more reliably

---

## 4. Workflow Explanation

### 4.1 LangGraph Usage

LangGraph extends LangChain with **stateful, graph-based** workflow execution. Unlike LangChain's LCEL (which is linear), LangGraph supports:

- **Conditional branching:** Different paths based on runtime decisions
- **State persistence:** A shared state object flows through all nodes
- **Interrupts:** Pause execution mid-graph for external input (HITL)
- **Cycles:** Nodes can loop back (not used in v1, but enables retry logic)

### 4.2 Node Responsibilities

| Node | Responsibility | Failure Mode |
|------|---------------|-------------|
| `classify_intent` | Determine what the user wants (intent + confidence) | Default to `general_inquiry` if classification fails |
| `retrieve_documents` | Find relevant knowledge base chunks | Empty results trigger escalation flag |
| `generate_response` | Produce a human-readable answer from context | Fallback to "I need to escalate this" message |
| `assess_quality` | Score response confidence and decide routing | Default to escalation if assessment fails (fail-safe) |
| `escalate_to_human` | Collect human agent input | Wait indefinitely (CLI) or timeout with message (Streamlit) |
| `format_output` | Package final response with metadata | Minimal formatting — always succeeds |

### 4.3 State Transitions

The `GraphState` TypedDict accumulates information as it flows through nodes:

1. **Initial state:** Only `query` is populated
2. **After classify_intent:** `intent` and `intent_confidence` added
3. **After retrieve_documents:** `retrieved_docs`, `retrieval_scores`, `context` added
4. **After generate_response:** `response` added
5. **After assess_quality:** `confidence`, `needs_escalation`, `escalation_reason` added
6. **After format_output:** `source_documents` added, final response packaged

Each node reads what it needs and writes its outputs — nodes are independent and testable.

---

## 5. Conditional Logic

### 5.1 Intent Detection

The system classifies queries into 5 intent categories using LLM-based classification:

| Intent | Example Queries | Routing |
|--------|----------------|---------|
| `general_inquiry` | "What products do you offer?", "What are your hours?" | Standard RAG pipeline |
| `technical_support` | "My device won't connect", "How to update firmware?" | Standard RAG pipeline |
| `billing` | "Why was I charged twice?", "How to cancel subscription?" | Standard RAG pipeline |
| `complaint` | "Your service is terrible", "I want a refund immediately" | RAG pipeline with lower escalation threshold (0.8 vs 0.7) |
| `escalation_request` | "Let me talk to a human", "I need a manager" | Immediate HITL bypass |

### 5.2 Routing Decision Trees

**Decision Point 1: Post-Classification**
```
Intent == "escalation_request"?
    YES → Skip RAG, go directly to HITL
    NO  → Proceed to document retrieval
```

**Decision Point 2: Post-Quality-Assessment**
```
Confidence >= 0.7?
    AND Intent != "complaint"?
        YES → Return response to user
    OR Intent == "complaint" AND Confidence >= 0.8?
        YES → Return response to user
    NO → Escalate to human agent
```

---

## 6. HITL Implementation

### 6.1 Role of Humans

Humans serve as the **safety net** in the system. They handle cases where the AI lacks sufficient knowledge or the query requires empathy, judgment, or authority that AI cannot provide.

**Human agent responsibilities:**
- Answer queries outside the knowledge base scope
- Handle sensitive complaints with empathy
- Make decisions requiring authority (refunds, exceptions, account changes)
- Provide feedback that could improve the knowledge base

### 6.2 Benefits

| Benefit | Explanation |
|---------|-------------|
| **Accuracy safety net** | Wrong answers never reach customers — uncertain responses are caught |
| **Customer satisfaction** | Complex issues get human attention, not generic AI responses |
| **Continuous improvement** | Human responses identify knowledge base gaps for future updates |
| **Regulatory compliance** | Certain actions (refunds, account changes) may legally require human authorization |
| **Trust building** | Customers know a human is available when needed |

### 6.3 Limitations

| Limitation | Mitigation |
|-----------|-----------|
| **Human availability** | Queue system with estimated wait times; async response option |
| **Scalability** | HITL should handle <10% of queries; if higher, expand knowledge base |
| **Response time** | AI handles 90%+ instantly; escalated queries have SLA targets |
| **Subjectivity** | Provide templates and guidelines to human agents for consistency |

---

## 7. Challenges & Trade-offs

### 7.1 Accuracy vs Speed

| Optimization | Speed Impact | Accuracy Impact |
|-------------|-------------|----------------|
| Reduce top-K from 4 to 2 | ~20% faster retrieval | Risk missing relevant context |
| Use smaller LLM (flash vs pro) | 2x faster generation | Slightly lower reasoning quality |
| Skip confidence assessment | Removes 1 LLM call (~1s) | No quality gate — hallucinations reach users |
| Cache frequent queries | Near-instant for cached | Stale answers if KB updated |

**Our choice:** Keep all 4 steps (classify → retrieve → generate → assess) and optimize with caching for frequently asked questions. Accuracy is paramount in customer support.

### 7.2 Chunk Size vs Context Quality

```
Small chunks (256 chars):
  ✅ Precise retrieval — finds exact relevant sentences
  ❌ LLM lacks context to form coherent answers
  ❌ More chunks needed → more embedding storage

Large chunks (2000 chars):
  ✅ Rich context for LLM response generation
  ❌ Retrieval noise — irrelevant content mixed in
  ❌ Fewer chunks fit in LLM context window

Our choice: 1000 chars (balanced):
  ✅ One complete concept per chunk
  ✅ Good retrieval precision and generation context
  ⚠️ Slightly lower precision than 256-char for exact matches
```

### 7.3 Cost vs Performance

| Configuration | Monthly Cost | Quality | Latency |
|--------------|-------------|---------|---------|
| Gemini Flash (free tier) | $0 | Good | ~2s |
| Gemini Pro | ~$50 | Better | ~3s |
| GPT-4o | ~$200 | Best | ~4s |
| Local Ollama (Llama 3) | $0 (compute cost) | Good | ~5-10s |

**Our choice:** Gemini Flash free tier — zero cost with acceptable quality for customer support. Upgrade path to Pro/GPT-4o is a configuration change, not a code change.

---

## 8. Testing Strategy

### 8.1 Testing Approach

| Test Type | Scope | Method |
|-----------|-------|--------|
| **Unit Tests** | Individual modules (chunker, retriever, classifier) | pytest with mocked LLM calls |
| **Integration Tests** | Full pipeline (ingest → query → response) | End-to-end with sample PDF |
| **Routing Tests** | Conditional edges in LangGraph | Predefined queries with expected routes |
| **HITL Tests** | Escalation triggering and reintegration | Simulated human responses |
| **Quality Tests** | Response accuracy against known answers | Sample Q&A pairs from knowledge base |

### 8.2 Sample Test Queries

| Query | Expected Intent | Expected Route | Expected Behavior |
|-------|---------------|---------------|-------------------|
| "How do I set up my SmartHome Hub?" | technical_support | RAG → Output | High confidence answer with setup steps |
| "What is your refund policy?" | billing | RAG → Output | Policy details from KB |
| "I want to talk to a human" | escalation_request | Direct HITL | Immediate escalation, no RAG |
| "Your product destroyed my entire home network and I want compensation" | complaint | RAG → HITL | Low confidence on complex complaint → escalate |
| "What is the meaning of life?" | general_inquiry | RAG → HITL | No relevant docs → escalate |
| "How do I update firmware on SecureVPN?" | technical_support | RAG → Output | Step-by-step from KB |
| "Cancel my subscription immediately" | billing | RAG → Output or HITL | Depends on KB coverage of cancellation process |
| "" (empty) | N/A | Error handling | Prompt for valid input |

### 8.3 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Response Accuracy** | >85% correct answers | Manual review of 50 test queries |
| **Retrieval Relevance** | >90% of retrieved docs are relevant | Compare retrieved docs to manually tagged relevant docs |
| **Appropriate Escalation** | <5% false escalations (AI could have answered) | Review escalated queries for necessity |
| **Missed Escalation** | <2% (should have escalated but didn't) | Review low-rated AI responses |
| **End-to-End Latency** | <4 seconds (non-escalated) | Automated timing |

---

## 9. Future Enhancements

### 9.1 Multi-Document Support

**Current:** Single PDF ingestion
**Future:** Directory scanning with support for PDF, DOCX, TXT, HTML, Markdown. Each document tracked with metadata for source attribution and selective re-ingestion.

### 9.2 Feedback Loop

**Current:** One-shot responses with no learning
**Future:** User feedback buttons (👍/👎) after each response. Negative feedback triggers:
- Response flagged for review
- Knowledge base gap identification
- Prompt refinement based on failure patterns
- Automatic HITL threshold adjustment

### 9.3 Conversation Memory

**Current:** Stateless — each query is independent
**Future:** Session-based memory using LangGraph's checkpointing:
- Track conversation history per session
- Enable follow-up questions ("What about the warranty?" after a product query)
- Context-aware intent classification

### 9.4 Advanced Retrieval

**Current:** Basic cosine similarity search
**Future:**
- **Hybrid search:** Combine semantic (vector) with keyword (BM25) search
- **Re-ranking:** Use a cross-encoder model to re-rank retrieved results
- **Query expansion:** Automatically generate related queries for broader retrieval
- **Parent document retrieval:** Retrieve small chunks for precision, but pass the full parent document to the LLM for context

### 9.5 Deployment

**Current:** Local execution (CLI + Streamlit)
**Future:**
- **Containerization:** Docker + Docker Compose for reproducible deployment
- **API Server:** FastAPI backend with REST endpoints
- **Cloud Deployment:** AWS/GCP with managed vector DB (Pinecone)
- **Monitoring:** LangSmith for LLM call tracing, Prometheus for system metrics
- **CI/CD:** GitHub Actions for automated testing and deployment

---

*End of Technical Documentation*
