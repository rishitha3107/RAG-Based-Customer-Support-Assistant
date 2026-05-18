"""
LangGraph Workflow Module
Implements the stateful RAG pipeline as a directed graph with conditional routing.
This is the core orchestration engine of the system.
"""
import logging
from typing import TypedDict, List, Optional, Annotated

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

import config
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.intent_classifier import IntentClassifier
from src.hitl_manager import HITLManager

logger = logging.getLogger(__name__)


# ─── Graph State Schema ─────────────────────────────────────
class GraphState(TypedDict):
    """Central state object passed through all graph nodes."""
    query: str
    intent: str
    intent_confidence: float
    retrieved_docs: List[Document]
    retrieval_scores: List[float]
    context: str
    response: str
    confidence: float
    needs_escalation: bool
    escalation_reason: str
    human_response: Optional[str]
    source_documents: List[dict]
    handled_by: str
    error: Optional[str]


class RAGGraphWorkflow:
    """
    LangGraph-based stateful workflow for the RAG pipeline.
    
    Graph Structure:
        START → classify_intent → [conditional]
          ├─ escalation_request → escalate_to_human → format_output → END
          └─ other intents → retrieve_documents → generate_response 
             → assess_quality → [conditional]
                ├─ high confidence → format_output → END
                └─ low confidence → escalate_to_human → format_output → END
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_handler: LLMHandler,
        hitl_manager: HITLManager
    ):
        self.retriever = retriever
        self.llm_handler = llm_handler
        self.intent_classifier = IntentClassifier(llm_handler=llm_handler)
        self.hitl_manager = hitl_manager
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the state graph with nodes and conditional edges."""
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("assess_quality", self._assess_quality)
        graph.add_node("escalate_to_human", self._escalate_to_human)
        graph.add_node("format_output", self._format_output)

        # Set entry point
        graph.set_entry_point("classify_intent")

        # Conditional edge after intent classification
        graph.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "retrieve": "retrieve_documents",
                "escalate": "escalate_to_human"
            }
        )

        # Sequential edges: retrieve → generate → assess
        graph.add_edge("retrieve_documents", "generate_response")
        graph.add_edge("generate_response", "assess_quality")

        # Conditional edge after quality assessment
        graph.add_conditional_edges(
            "assess_quality",
            self._route_after_assessment,
            {
                "output": "format_output",
                "escalate": "escalate_to_human"
            }
        )

        # Terminal edges
        graph.add_edge("escalate_to_human", "format_output")
        graph.add_edge("format_output", END)

        return graph.compile()

    # ─── Node Functions ──────────────────────────────────────

    def _classify_intent(self, state: GraphState) -> dict:
        """Node 1: Classify the user's intent."""
        logger.info(f"[Node: classify_intent] Processing query: '{state['query'][:80]}'")

        try:
            result = self.intent_classifier.classify(state["query"])
            return {
                "intent": result.get("intent", "general_inquiry"),
                "intent_confidence": result.get("confidence", 0.5),
            }
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {
                "intent": "general_inquiry",
                "intent_confidence": 0.5,
                "error": str(e),
            }

    def _retrieve_documents(self, state: GraphState) -> dict:
        """Node 2: Retrieve relevant documents from ChromaDB (single call, deduplicated)."""
        logger.info(f"[Node: retrieve_documents] Searching for: '{state['query'][:80]}'")

        try:
            # Single retrieval call that returns both results and formatted context
            results, context = self.retriever.retrieve_with_context(state["query"])
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]

            return {
                "retrieved_docs": docs,
                "retrieval_scores": scores,
                "context": context,
            }
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                "retrieved_docs": [],
                "retrieval_scores": [],
                "context": "Error retrieving documents.",
                "error": str(e),
            }

    def _generate_response(self, state: GraphState) -> dict:
        """Node 3: Generate response using LLM with retrieved context."""
        logger.info("[Node: generate_response] Generating LLM response")

        try:
            response = self.llm_handler.generate_response(
                query=state["query"],
                context=state.get("context", "No context available.")
            )
            return {"response": response}
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return {
                "response": "I apologize, but I'm unable to process your request right now.",
                "error": str(e),
            }

    def _assess_quality(self, state: GraphState) -> dict:
        """Node 4: Assess response quality using hybrid confidence scoring."""
        logger.info("[Node: assess_quality] Evaluating response confidence")

        response = state.get("response", "")
        retrieval_scores = state.get("retrieval_scores", [])
        has_response = bool(response and "unable to process" not in response.lower() and "apologize" not in response.lower())

        try:
            # Get LLM-based confidence assessment
            assessment = self.llm_handler.assess_confidence(
                query=state["query"],
                context=state.get("context", ""),
                response=response
            )
            llm_confidence = assessment.get("confidence", 0.65)
        except Exception as e:
            logger.error(f"LLM confidence assessment error: {e}")
            llm_confidence = 0.65  # Safe default above escalation threshold

        # Compute hybrid confidence from multiple signals
        hybrid_confidence = self.llm_handler.compute_hybrid_confidence(
            retrieval_scores=retrieval_scores,
            llm_confidence=llm_confidence,
            has_response=has_response
        )

        # Check escalation conditions with the hybrid score
        needs_escalation, reason = self.hitl_manager.should_escalate({
            **state,
            "confidence": hybrid_confidence
        })

        return {
            "confidence": hybrid_confidence,
            "needs_escalation": needs_escalation,
            "escalation_reason": reason,
        }

    def _escalate_to_human(self, state: GraphState) -> dict:
        """Node 5: Escalate to human agent and collect response."""
        logger.info(f"[Node: escalate_to_human] Reason: {state.get('escalation_reason', 'Unknown')}")

        context = self.hitl_manager.format_escalation_context(state)
        human_response = self.hitl_manager.request_human_input(context)

        return {
            "human_response": human_response,
            "response": human_response,
            "needs_escalation": False,
            "handled_by": "human",
        }

    def _format_output(self, state: GraphState) -> dict:
        """Node 6: Format the final output with sources and metadata."""
        logger.info("[Node: format_output] Packaging final response")

        source_documents = []
        for doc in state.get("retrieved_docs", []):
            if hasattr(doc, "metadata"):
                source_documents.append({
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "N/A"),
                    "excerpt": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                })

        handled_by = state.get("handled_by", "ai")

        return {
            "source_documents": source_documents,
            "handled_by": handled_by,
        }

    # ─── Routing Functions ───────────────────────────────────

    def _route_after_intent(self, state: GraphState) -> str:
        """Route based on classified intent."""
        if state.get("intent") == "escalation_request":
            logger.info("[Routing] Explicit escalation request → HITL")
            return "escalate"
        logger.info(f"[Routing] Intent '{state.get('intent')}' → Retrieve documents")
        return "retrieve"

    def _route_after_assessment(self, state: GraphState) -> str:
        """Route based on response quality assessment."""
        if state.get("needs_escalation", False):
            logger.info(f"[Routing] Escalation needed: {state.get('escalation_reason')} → HITL")
            return "escalate"
        logger.info(f"[Routing] Confidence {state.get('confidence', 0):.2f} → Output")
        return "output"

    # ─── Public Interface ────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Execute the full RAG workflow for a query.
        
        Args:
            query: User's natural language question.
            
        Returns:
            Final graph state with response, sources, and metadata.
        """
        if not query or not query.strip():
            return {
                "query": query,
                "response": "Please enter a valid question.",
                "confidence": 0.0,
                "intent": "none",
                "handled_by": "system",
                "error": "Empty query",
            }

        # Initialize state
        initial_state = {
            "query": query.strip(),
            "intent": "",
            "intent_confidence": 0.0,
            "retrieved_docs": [],
            "retrieval_scores": [],
            "context": "",
            "response": "",
            "confidence": 0.0,
            "needs_escalation": False,
            "escalation_reason": "",
            "human_response": None,
            "source_documents": [],
            "handled_by": "ai",
            "error": None,
        }

        logger.info(f"Starting RAG workflow for query: '{query[:80]}'")
        result = self.graph.invoke(initial_state)
        logger.info(f"Workflow complete. Handled by: {result.get('handled_by')}, Confidence: {result.get('confidence', 0):.2f}")

        return result
