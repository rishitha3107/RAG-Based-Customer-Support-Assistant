"""
Retriever Module
Handles document retrieval from the ChromaDB vector store.
Includes deduplication to prevent returning near-identical chunks.
"""
import logging
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieves relevant documents from the vector store with scoring and deduplication."""

    def __init__(
        self,
        vectorstore: Chroma,
        top_k: int = config.RETRIEVAL_TOP_K,
        score_threshold: float = config.RETRIEVAL_SCORE_THRESHOLD
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k relevant documents with similarity scores.
        Applies deduplication to remove near-identical chunks.

        Returns:
            List of (Document, score) tuples, deduplicated, filtered by threshold.
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        logger.info(f"Retrieving top-{self.top_k} documents for query: '{query[:80]}'")

        try:
            # Fetch more candidates than top_k so we still have enough after dedup
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=self.top_k + 3
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

        # Filter by score threshold
        filtered = [
            (doc, score) for doc, score in results
            if score >= self.score_threshold
        ]

        # Deduplicate near-identical chunks
        deduped = self._deduplicate(filtered)

        # Limit to top_k after dedup
        deduped = deduped[:self.top_k]

        logger.info(
            f"Retrieved {len(results)} raw, "
            f"{len(filtered)} above threshold, "
            f"{len(deduped)} after dedup"
        )

        return deduped

    def _deduplicate(self, results: List[Tuple[Document, float]], similarity_ratio: float = 0.85) -> List[Tuple[Document, float]]:
        """
        Remove near-duplicate chunks based on text overlap.
        Keeps the highest-scoring version of similar chunks.
        """
        if not results:
            return []

        deduped = []
        seen_texts = []

        for doc, score in results:
            content = doc.page_content.strip()
            is_duplicate = False

            for seen in seen_texts:
                # Simple overlap check: if >85% of the shorter text appears in the longer
                shorter, longer = (content, seen) if len(content) <= len(seen) else (seen, content)
                overlap = sum(1 for c in shorter if c in longer) / max(len(shorter), 1)
                if overlap > similarity_ratio:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append((doc, score))
                seen_texts.append(content)

        return deduped

    def retrieve_with_context(self, query: str) -> Tuple[List[Tuple[Document, float]], str]:
        """
        Retrieve documents AND build context string in a single call.
        This avoids the double-retrieval bug in the original code.

        Returns:
            Tuple of (results_list, context_string)
        """
        results = self.retrieve(query)

        if not results:
            return [], "No relevant information found in the knowledge base."

        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Source {i}: {source}, Page {page} (Relevance: {score:.2f})]\n"
                f"{doc.page_content}"
            )

        context_str = "\n\n---\n\n".join(context_parts)
        return results, context_str

    def retrieve_documents_only(self, query: str) -> List[Document]:
        """Retrieve documents without scores (convenience method)."""
        results = self.retrieve(query)
        return [doc for doc, _ in results]

    def retrieve_as_context(self, query: str) -> str:
        """Retrieve documents and format as a context string for the LLM."""
        _, context = self.retrieve_with_context(query)
        return context

    def get_average_score(self, query: str) -> float:
        """Get the average retrieval score for a query (useful for confidence calculation)."""
        results = self.retrieve(query)
        if not results:
            return 0.0
        return sum(score for _, score in results) / len(results)

    def has_relevant_results(self, query: str) -> bool:
        """Check if there are any relevant results above threshold."""
        results = self.retrieve(query)
        return len(results) > 0
