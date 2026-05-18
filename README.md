# 🤖 RAG-Based Customer Support Assistant

> **Design & Build a RAG-Based Customer Support Assistant with LangGraph & Human-in-the-Loop**

A production-grade Retrieval-Augmented Generation (RAG) system that processes PDF knowledge bases, retrieves relevant context using semantic search, and generates accurate customer support responses — with intelligent intent routing and human escalation capabilities.

![Tech Stack](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Gemini](https://img.shields.io/badge/Gemini_2.0-Flash-blue?logo=google)

---

## 🏗️ Architecture Overview

```
User Query → Intent Classification → Document Retrieval (ChromaDB)
    → LLM Response Generation (Gemini) → Quality Assessment
    → [High Confidence] → Return Answer
    → [Low Confidence]  → Human-in-the-Loop Escalation
```

**Key Components:**
- **Document Pipeline**: PDF → Chunks → Embeddings → ChromaDB
- **RAG Engine**: Semantic retrieval + LLM generation with source attribution
- **LangGraph Workflow**: 6-node stateful graph with conditional routing
- **HITL System**: Automatic escalation for low-confidence or complex queries
- **React Frontend**: Premium ChatGPT-style UI with glassmorphism design
- **FastAPI Backend**: REST API powering the frontend

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for React frontend)
- **Google Gemini API Key** — free at [aistudio.google.com](https://aistudio.google.com)

---

### Step 1: Clone & Navigate

```bash
cd "RAG INTERNSHIP PROJECT"
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### Step 3: Set Up Environment

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Edit `.env` and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

### Step 4: Generate Sample Knowledge Base

```bash
python knowledge_base/generate_sample_kb.py
```

This creates a 12-page TechCorp customer support PDF covering products, troubleshooting, billing, refunds, and escalation procedures.

### Step 5: Ingest Documents into ChromaDB

```bash
python -X utf8 ingest.py --reset
```

This processes the PDF → chunks it → generates embeddings → stores in ChromaDB.

> **Note:** The `-X utf8` flag is required on Windows for Unicode support.

### Step 6: Start the FastAPI Backend

```bash
python -X utf8 -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API server starts at `http://localhost:8000`.

### Step 7: Start the React Frontend

Open a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

The React app starts at `http://localhost:3000`.

### Step 8: Open the App

Navigate to **http://localhost:3000** in your browser.

---

## 🖥️ Alternative Interfaces

### CLI Chatbot (No frontend needed)

```bash
python -X utf8 app_cli.py
```

### Streamlit Web UI

```bash
streamlit run app_streamlit.py
```

---

## 📁 Project Structure

```
RAG INTERNSHIP PROJECT/
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── .env.example                         # Environment template
├── .env                                 # Your API keys (not in git)
├── config.py                            # Central configuration
│
├── api_server.py                        # FastAPI backend (REST API)
├── app_cli.py                           # CLI chatbot interface
├── app_streamlit.py                     # Streamlit web UI
├── ingest.py                            # PDF ingestion script
│
├── src/                                 # Core Python modules
│   ├── __init__.py
│   ├── document_processor.py            # PDF loading & chunking
│   ├── embedding_manager.py             # Embeddings & ChromaDB
│   ├── retriever.py                     # Semantic document retrieval
│   ├── llm_handler.py                   # LLM interactions (Gemini)
│   ├── intent_classifier.py             # Intent detection & routing
│   ├── graph_workflow.py                # LangGraph state machine
│   └── hitl_manager.py                  # Human-in-the-Loop escalation
│
├── frontend/                            # React frontend (Vite)
│   ├── package.json
│   ├── vite.config.js                   # Vite + Tailwind + API proxy
│   ├── index.html
│   └── src/
│       ├── main.jsx                     # React entry point
│       ├── App.jsx                      # Main UI component
│       ├── api.js                       # API client
│       └── index.css                    # Tailwind + custom styles
│
├── knowledge_base/                      # PDF knowledge base
│   ├── generate_sample_kb.py            # Sample KB generator
│   └── techcorp_support_kb.pdf          # Generated sample PDF
│
├── docs/                                # Design documentation
│   ├── HLD.md                           # High-Level Design
│   ├── LLD.md                           # Low-Level Design
│   └── Technical_Documentation.md       # Technical Documentation
│
└── chroma_db/                           # Vector store (auto-created)
```

---

## 🧠 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **LLM** | Google Gemini 2.0 Flash | Free tier, fast inference, strong instruction following |
| **Embeddings** | all-MiniLM-L6-v2 | Local, free, 384-dim, excellent speed/quality ratio |
| **Vector DB** | ChromaDB | Zero-config, persistent, native LangChain integration |
| **Orchestration** | LangGraph | Stateful graphs, conditional routing, HITL support |
| **Framework** | LangChain | Modular abstractions, extensive integrations |
| **Backend API** | FastAPI | Async, fast, automatic OpenAPI docs |
| **Frontend** | React + Vite | Modern, fast HMR, component-based |
| **Styling** | Tailwind CSS v4 | Utility-first, dark mode, responsive |
| **Animations** | Framer Motion | Smooth transitions, gesture support |
| **PDF Processing** | PyPDF | Lightweight, pure Python |

---

## 📋 LangGraph Workflow

```
START → classify_intent → [conditional]
  ├─ escalation_request → escalate_to_human → format_output → END
  └─ other intents → retrieve_documents → generate_response
     → assess_quality → [conditional]
        ├─ confidence ≥ 0.7 → format_output → END
        └─ confidence < 0.7 → escalate_to_human → format_output → END
```

### Escalation Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Explicit request | Intent = `escalation_request` | Immediate HITL |
| Low confidence | confidence < 0.7 | HITL after generation |
| No relevant docs | empty retrieval | HITL after retrieval |
| Complaint | intent = `complaint` AND confidence < 0.8 | HITL after generation |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | System health check |
| `POST` | `/api/query` | Submit a user query |
| `POST` | `/api/upload` | Upload a PDF file |
| `POST` | `/api/hitl/respond` | Submit human agent response |
| `GET` | `/api/sessions` | List chat sessions |
| `GET` | `/api/sessions/{id}` | Get session messages |
| `GET` | `/api/documents` | List uploaded documents |

---

## 🎨 Frontend Features

- **Premium dark theme** with glassmorphism effects
- **Chat bubbles** with user/assistant differentiation
- **Typing animation** (three-dot loading indicator)
- **Markdown rendering** in responses
- **Confidence score bar** with color coding (green/amber/red)
- **Intent & handler badges** per response
- **Expandable source documents** with page references
- **Copy & regenerate buttons** on each response
- **Timestamps** on all messages
- **Collapsible sidebar** with chat history
- **PDF upload** via drag & drop
- **System status indicator** (online/offline, doc count)
- **Suggested questions** on empty state
- **Responsive design** (mobile + desktop)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [HLD.md](docs/HLD.md) | System architecture, component design, scalability |
| [LLD.md](docs/LLD.md) | Module design, data structures, API design, error handling |
| [Technical_Documentation.md](docs/Technical_Documentation.md) | Design decisions, trade-offs, testing strategy |

---

## 🧪 Testing

### Sample Queries to Try

| Query | Expected Behavior |
|-------|-------------------|
| "How do I set up my SmartHome Hub?" | High-confidence answer with setup steps |
| "What is your refund policy?" | Policy details from KB |
| "I want to talk to a human" | Immediate HITL escalation |
| "My VPN is slow" | Troubleshooting steps |
| "Why was I charged twice?" | Billing FAQ response |
| "Your product is terrible" | Complaint → possible escalation |

---

## 🔧 Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVAL_TOP_K` | 4 | Documents to retrieve |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.3 | Minimum relevance score |
| `CONFIDENCE_THRESHOLD` | 0.7 | Below this → escalate |
| `LLM_MODEL_NAME` | gemini-2.0-flash | Gemini model to use |
| `LLM_TEMPERATURE_GENERATION` | 0.3 | Response creativity |

---

## 📝 License

This project is for educational and internship purposes.
