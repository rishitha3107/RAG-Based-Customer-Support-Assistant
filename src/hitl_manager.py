"""
HITL (Human-in-the-Loop) Manager Module
Handles escalation to human agents and response reintegration.
"""
import logging
from typing import Optional, Tuple

import config

logger = logging.getLogger(__name__)


class HITLManager:
    """Manages Human-in-the-Loop escalation and response collection."""

    ESCALATION_RULES = [
        {
            "name": "explicit_request",
            "description": "User explicitly requested a human agent",
            "priority": "HIGH",
        },
        {
            "name": "low_confidence",
            "description": "AI response confidence is below threshold",
            "priority": "MEDIUM",
        },
        {
            "name": "empty_retrieval",
            "description": "No relevant documents found in knowledge base",
            "priority": "MEDIUM",
        },
        {
            "name": "complaint_handling",
            "description": "Complaint requires human empathy and authority",
            "priority": "HIGH",
        },
        {
            "name": "system_error",
            "description": "System error during processing",
            "priority": "CRITICAL",
        },
    ]

    def __init__(self, mode: str = "cli"):
        """
        Args:
            mode: Interface mode - "cli" for terminal, "streamlit" for web UI.
        """
        self.mode = mode
        self.escalation_history = []

    def should_escalate(self, state: dict) -> Tuple[bool, str]:
        """
        Evaluate the current state and determine if escalation is needed.
        
        CONSERVATIVE escalation — only escalate when truly necessary:
        - Explicit user request for human agent
        - Confidence < 0.35 (truly bad answers only)
        - No documents retrieved AND confidence < 0.5
        - Complaints with very low confidence
        
        Returns:
            Tuple of (should_escalate: bool, reason: str)
        """
        # Rule 1: Explicit escalation request (always honor)
        if state.get("intent") == "escalation_request":
            return True, "Customer requested to speak with a human agent"

        confidence = state.get("confidence", 1.0)
        retrieved_docs = state.get("retrieved_docs", [])
        has_docs = isinstance(retrieved_docs, list) and len(retrieved_docs) > 0

        # Rule 2: Very low confidence — answer is genuinely bad
        threshold = config.CONFIDENCE_THRESHOLD  # 0.35
        if state.get("intent") == "complaint":
            threshold = config.COMPLAINT_CONFIDENCE_THRESHOLD  # 0.5

        if confidence < threshold:
            # Only escalate low confidence if we also lack documents
            if not has_docs:
                return True, f"Low confidence ({confidence:.2f}) with no relevant documents"
            # If we have docs but confidence is extremely low, still escalate
            if confidence < 0.2:
                return True, f"Very low confidence ({confidence:.2f}), answer likely incorrect"
            # For complaints specifically, escalate at the complaint threshold
            if state.get("intent") == "complaint" and confidence < config.COMPLAINT_CONFIDENCE_THRESHOLD:
                return True, f"Complaint with low confidence ({confidence:.2f})"

        # Rule 3: No documents AND response looks like an error message
        if not has_docs:
            response = state.get("response", "")
            if not response or "apologize" in response.lower() or "unable to" in response.lower():
                return True, "No relevant documents found and no valid response generated"

        # Default: DO NOT escalate — show the AI response
        return False, ""

    def format_escalation_context(self, state: dict) -> dict:
        """
        Package the current state into a human-readable escalation report.
        
        Args:
            state: Current graph state.
            
        Returns:
            Dictionary with escalation context for the human agent.
        """
        context = {
            "query": state.get("query", "N/A"),
            "intent": state.get("intent", "N/A"),
            "intent_confidence": state.get("intent_confidence", 0.0),
            "ai_response": state.get("response", "No response generated"),
            "response_confidence": state.get("confidence", 0.0),
            "escalation_reason": state.get("escalation_reason", "Unknown"),
            "retrieved_sources": [],
        }

        # Include retrieved document excerpts
        for doc in state.get("retrieved_docs", []):
            if hasattr(doc, "page_content"):
                context["retrieved_sources"].append({
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "N/A"),
                    "excerpt": doc.page_content[:200] + "..."
                })

        return context

    def request_human_input(self, context: dict) -> str:
        """
        Display escalation context and collect human response.
        
        Args:
            context: Escalation context dictionary.
            
        Returns:
            Human agent's response string.
        """
        if self.mode == "cli":
            return self._cli_input(context)
        elif self.mode == "streamlit":
            # In Streamlit mode, return a placeholder message.
            # Actual human input is collected via the Streamlit UI in app_streamlit.py.
            self.escalation_history.append({
                "query": context.get("query", ""),
                "reason": context.get("escalation_reason", ""),
                "human_response": "[PENDING - Awaiting human agent response]",
            })
            return "This query has been escalated to a human agent. A support representative will respond shortly."
        else:
            logger.error(f"Unknown HITL mode: {self.mode}")
            return "Escalation system unavailable. Please try again later."

    def _cli_input(self, context: dict) -> str:
        """Collect human input via CLI."""
        print("\n" + "=" * 60)
        print("[ESCALATION] ESCALATION TO HUMAN AGENT")
        print("=" * 60)
        print(f"\n[QUERY] Customer Query: {context['query']}")
        print(f"[INTENT] Intent: {context['intent']} (confidence: {context['intent_confidence']:.2f})")
        print(f"[REASON] Reason: {context['escalation_reason']}")

        if context['ai_response'] != "No response generated":
            print(f"\n[AI RESPONSE] AI's Attempted Response:")
            print(f"   {context['ai_response'][:300]}")
            print(f"   Confidence: {context['response_confidence']:.2f}")

        if context['retrieved_sources']:
            print(f"\n[SOURCES] Retrieved Sources:")
            for i, src in enumerate(context['retrieved_sources'], 1):
                print(f"   [{i}] Page {src['page']}: {src['excerpt'][:100]}...")

        print("\n" + "-" * 60)
        human_response = input("[HUMAN AGENT] Response: ").strip()

        if not human_response:
            human_response = "Thank you for reaching out. A support agent will follow up with you shortly via email."

        # Log escalation
        self.escalation_history.append({
            "query": context["query"],
            "reason": context["escalation_reason"],
            "human_response": human_response,
        })

        return human_response

    def get_escalation_stats(self) -> dict:
        """Get statistics about escalation history."""
        return {
            "total_escalations": len(self.escalation_history),
            "history": self.escalation_history[-10:]  # Last 10
        }
