"""
Embedding Manager Module
Handles embedding generation and ChromaDB vector store operations.
"""
import os
import logging
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding model and ChromaDB vector store lifecycle."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME
    ):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        logger.info(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded successfully")

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new ChromaDB collection from documents.
        Embeds all documents and stores them with metadata.
        
        Args:
            documents: List of Document objects (chunked).
            
        Returns:
            Chroma vectorstore instance.
        """
        if not documents:
            raise ValueError("No documents provided for vectorstore creation")

        logger.info(f"Creating vectorstore with {len(documents)} documents in {self.persist_dir}")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )

        logger.info(f"Vectorstore created with {len(documents)} documents")
        return vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load an existing ChromaDB collection from disk.
        
        Returns:
            Chroma vectorstore instance, or None if not found.
        """
        if not os.path.exists(self.persist_dir):
            logger.warning(f"Vectorstore directory not found: {self.persist_dir}")
            return None

        logger.info(f"Loading vectorstore from {self.persist_dir}")
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        return vectorstore

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current vector store collection.
        
        Returns:
            Dictionary with collection metadata.
        """
        vectorstore = self.load_vectorstore()
        if vectorstore is None:
            return {"status": "not_found", "document_count": 0}

        try:
            collection = vectorstore._collection
            count = collection.count()
            return {
                "status": "active",
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_dir,
                "embedding_model": self.model_name,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"status": "error", "error": str(e)}

    def delete_collection(self):
        """Delete the existing collection (useful for re-ingestion)."""
        import shutil
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            logger.info(f"Deleted vectorstore at {self.persist_dir}")
