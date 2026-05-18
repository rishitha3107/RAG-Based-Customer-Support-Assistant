"""
Central configuration for the RAG Customer Support Assistant.
All tunable parameters are defined here for easy modification.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ─── Document Processing ────────────────────────────────────
CHUNK_SIZE = 800            # Characters per chunk (smaller = more precise retrieval)
CHUNK_OVERLAP = 150         # Overlap between adjacent chunks
PDF_DIRECTORY = "knowledge_base"

# ─── Embedding Model ────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ─── Vector Store (ChromaDB) ────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "support_kb"

# ─── Retrieval ──────────────────────────────────────────────
RETRIEVAL_TOP_K = 5                 # Number of documents to retrieve
RETRIEVAL_SCORE_THRESHOLD = 0.25    # Minimum relevance score (lowered to avoid over-filtering)

# ─── LLM Configuration ─────────────────────────────────────
LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_TEMPERATURE_GENERATION = 0.3   # Low creativity for accurate answers
LLM_TEMPERATURE_CLASSIFICATION = 0.0  # Deterministic for classification
LLM_MAX_TOKENS_RESPONSE = 1024
LLM_MAX_TOKENS_CLASSIFICATION = 256

# ─── Confidence & Escalation ────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35         # Only truly bad answers escalate (was 0.7 — too aggressive)
COMPLAINT_CONFIDENCE_THRESHOLD = 0.5  # Complaints need slightly higher quality

# ─── Intent Categories ──────────────────────────────────────
INTENT_CATEGORIES = {
    "general_inquiry": "General product questions, feature inquiries, company info",
    "technical_support": "Setup, troubleshooting, configuration, connectivity issues",
    "billing": "Payments, subscriptions, pricing, invoices, refunds",
    "complaint": "Negative feedback, service issues, dissatisfaction, compensation",
    "escalation_request": "Explicit request to speak with a human agent or manager"
}

# ─── Prompt Templates ───────────────────────────────────────
RESPONSE_PROMPT_TEMPLATE = """You are a helpful, professional customer support assistant for TechCorp.
Answer the customer's question using the provided context.

CRITICAL RULES:
1. ALWAYS provide the best possible answer from the context — even if partial.
2. If the context covers the topic partially, answer what you CAN and note what's missing.
3. NEVER say "I don't have information" if the context contains ANY relevant details.
4. Be concise, friendly, and professional.
5. Reference specific steps, prices, or details from the context.
6. For technical issues, provide step-by-step instructions.
7. For billing/policy questions, quote the relevant policy.
8. Do NOT fabricate information not present in the context.

CONTEXT:
{context}

CUSTOMER QUESTION:
{query}

Provide a helpful, complete response based on the context above:"""

INTENT_CLASSIFICATION_PROMPT = """Classify the following customer support query into exactly ONE of these categories:
- general_inquiry: General product questions, feature inquiries
- technical_support: Setup, troubleshooting, configuration issues
- billing: Payments, subscriptions, pricing, invoices
- complaint: Negative feedback, service issues, dissatisfaction
- escalation_request: Explicit request to speak with human agent

Query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{"intent": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

CONFIDENCE_ASSESSMENT_PROMPT = """Evaluate how well the following response answers the customer's question.

CUSTOMER QUESTION: {query}

AVAILABLE CONTEXT: {context}

GENERATED RESPONSE: {response}

Scoring guidelines:
- 0.9-1.0: Response directly and completely answers the question using context
- 0.7-0.9: Response answers the question well with minor gaps
- 0.5-0.7: Response provides a partial but useful answer
- 0.3-0.5: Response is vaguely related but missing key information
- 0.0-0.3: Response is wrong, irrelevant, or the context has no useful information

IMPORTANT: If the context contains relevant information and the response uses it correctly, score AT LEAST 0.6. Only give low scores if the response is actually wrong or the context truly has no relevant information.

Respond with ONLY a JSON object:
{{"confidence": <score>, "reasoning": "<brief explanation>"}}"""
