"""
Document Processing Module
Handles PDF loading and intelligent text chunking.
"""
import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes PDF documents into retrieval-optimized chunks."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and return raw Document objects.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of Document objects, one per page, with metadata.
            
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a PDF.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError(f"File is not a PDF: {pdf_path}")

        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {pdf_path}")

        # Enrich metadata
        for i, doc in enumerate(documents):
            doc.metadata["source"] = os.path.basename(pdf_path)
            doc.metadata["page"] = i

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into retrieval-optimized chunks.
        
        Args:
            documents: List of Document objects (typically from load_pdf).
            
        Returns:
            List of chunked Document objects with updated metadata.
        """
        logger.info(f"Chunking {len(documents)} documents (size={self.chunk_size}, overlap={self.chunk_overlap})")
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk-level metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def process(self, pdf_path: str) -> List[Document]:
        """
        Full pipeline: Load PDF → Chunk → Return.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of chunked Document objects ready for embedding.
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.chunk_documents(documents)
        return chunks

    def process_directory(self, directory: str) -> List[Document]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Path to directory containing PDF files.
            
        Returns:
            Combined list of chunks from all PDFs.
        """
        all_chunks = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return all_chunks

        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            try:
                chunks = self.process(pdf_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {pdf_file}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")

        logger.info(f"Total chunks from {len(pdf_files)} PDFs: {len(all_chunks)}")
        return all_chunks
