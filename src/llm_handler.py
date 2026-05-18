"""
LLM Handler Module
Manages all interactions with the Google Gemini LLM.
"""
import json
import logging
from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import config

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles LLM interactions: response generation, intent classification, confidence scoring."""

    def __init__(
        self,
        model_name: str = config.LLM_MODEL_NAME,
        api_key: str = config.GOOGLE_API_KEY
    ):
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in your .env file. "
                "Get a free key at https://aistudio.google.com"
            )

        self.generation_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=config.LLM_TEMPERATURE_GENERATION,
            max_output_tokens=config.LLM_MAX_TOKENS_RESPONSE,
        )

        self.classification_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=config.LLM_TEMPERATURE_CLASSIFICATION,
            max_output_tokens=config.LLM_MAX_TOKENS_CLASSIFICATION,
        )

        logger.info(f"LLM initialized: {model_name}")

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a customer support response grounded in retrieved context.
        
        Args:
            query: Customer's question.
            context: Retrieved document context.
            
        Returns:
            Generated response string.
        """
        prompt = config.RESPONSE_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )

        try:
            response = self.generation_llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I'm experiencing a technical issue. Please try again or ask to speak with a human agent."

    def classify_intent(self, query: str) -> dict:
        """
        Classify the user's query into an intent category.
        
        Args:
            query: Customer's question.
            
        Returns:
            Dict with keys: intent, confidence, reasoning.
        """
        prompt = config.INTENT_CLASSIFICATION_PROMPT.format(query=query)

        try:
            response = self.classification_llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)

            # Validate intent is a known category
            if result.get("intent") not in config.INTENT_CATEGORIES:
                logger.warning(f"Unknown intent '{result.get('intent')}', defaulting to general_inquiry")
                result["intent"] = "general_inquiry"

            return result
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "general_inquiry",
                "confidence": 0.7,
                "reasoning": f"Classification failed, defaulting to general inquiry: {e}"
            }

    def assess_confidence(self, query: str, context: str, response: str) -> dict:
        """
        Assess the quality/confidence of a generated response.
        
        Args:
            query: Original customer question.
            context: Retrieved context used for generation.
            response: The generated response to evaluate.
            
        Returns:
            Dict with keys: confidence (float), reasoning (str).
        """
        prompt = config.CONFIDENCE_ASSESSMENT_PROMPT.format(
            query=query,
            context=context,
            response=response
        )

        try:
            result = self.classification_llm.invoke([HumanMessage(content=prompt)])
            parsed = self._parse_json_response(result.content)

            # Ensure confidence is a valid float
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "confidence": confidence,
                "reasoning": parsed.get("reasoning", "No reasoning provided")
            }
        except Exception as e:
            logger.error(f"Confidence assessment failed: {e}")
            return {
                "confidence": 0.65,
                "reasoning": f"Assessment failed, using safe default: {e}"
            }

    def compute_hybrid_confidence(
        self, retrieval_scores: list, llm_confidence: float, has_response: bool
    ) -> float:
        """
        Compute a weighted hybrid confidence score from multiple signals.
        
        Formula: confidence = 0.4 * avg_retrieval + 0.5 * llm_confidence + 0.1 * response_bonus
        
        This prevents a single failing signal (e.g. LLM API error) from
        tanking the overall confidence into escalation territory.
        """
        # Signal 1: Average retrieval relevance (0-1)
        avg_retrieval = sum(retrieval_scores) / max(len(retrieval_scores), 1) if retrieval_scores else 0.0

        # Signal 2: LLM-assessed confidence (0-1) 
        llm_score = max(0.0, min(1.0, llm_confidence))

        # Signal 3: Binary bonus — did we generate a non-empty, non-error response?
        response_bonus = 1.0 if has_response else 0.0

        # Weighted combination
        hybrid = (0.4 * avg_retrieval) + (0.5 * llm_score) + (0.1 * response_bonus)

        logger.info(
            f"Hybrid confidence: {hybrid:.2f} "
            f"(retrieval={avg_retrieval:.2f}, llm={llm_score:.2f}, response={response_bonus})"
        )

        return round(max(0.0, min(1.0, hybrid)), 2)

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code block markers if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
            # Attempt to extract JSON-like content
            import re
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {}
