"""
Intent Classifier Module
Classifies user queries into actionable support intents.
"""
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies customer queries into predefined intent categories.
    Uses the LLM handler for classification but provides a clean interface.
    """

    INTENTS = config.INTENT_CATEGORIES

    # Keywords that strongly indicate specific intents (fallback heuristic)
    KEYWORD_HINTS = {
        "escalation_request": [
            "talk to human", "speak to agent", "real person", "manager",
            "supervisor", "human agent", "speak to someone", "talk to someone",
            "representative", "live agent"
        ],
        "complaint": [
            "terrible", "awful", "worst", "unacceptable", "disgusted",
            "furious", "compensation", "sue", "lawyer", "legal"
        ],
        "billing": [
            "charged", "invoice", "payment", "refund", "subscription",
            "cancel", "pricing", "bill", "credit card", "money"
        ],
        "technical_support": [
            "not working", "error", "bug", "crash", "setup",
            "install", "connect", "reset", "update", "firmware"
        ]
    }

    def __init__(self, llm_handler=None):
        """
        Args:
            llm_handler: LLMHandler instance for LLM-based classification.
                        If None, falls back to keyword-based classification.
        """
        self.llm_handler = llm_handler

    def classify(self, query: str) -> dict:
        """
        Classify a query into an intent category.
        
        Uses LLM classification with keyword-based fallback.
        
        Args:
            query: The customer's query string.
            
        Returns:
            Dict with: intent, confidence, reasoning
        """
        if not query or not query.strip():
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "reasoning": "Empty query"
            }

        # Quick keyword check for escalation (highest priority)
        keyword_result = self._keyword_classify(query.lower())
        if keyword_result and keyword_result["intent"] == "escalation_request":
            return keyword_result

        # LLM-based classification
        if self.llm_handler:
            try:
                result = self.llm_handler.classify_intent(query)
                if result.get("intent") and result.get("confidence", 0) > 0.3:
                    return result
            except Exception as e:
                logger.warning(f"LLM classification failed, using keyword fallback: {e}")

        # Fallback to keyword-based
        if keyword_result:
            return keyword_result

        return {
            "intent": "general_inquiry",
            "confidence": 0.6,
            "reasoning": "Default classification — no strong intent signals detected"
        }

    def _keyword_classify(self, query_lower: str) -> Optional[dict]:
        """Simple keyword-based intent detection as fallback."""
        for intent, keywords in self.KEYWORD_HINTS.items():
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                confidence = min(0.9, 0.5 + 0.1 * len(matches))
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "reasoning": f"Keyword match: {', '.join(matches[:3])}"
                }
        return None

    @staticmethod
    def get_intent_description(intent: str) -> str:
        """Get human-readable description of an intent."""
        return config.INTENT_CATEGORIES.get(intent, "Unknown intent category")
