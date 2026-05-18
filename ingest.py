"""
PDF Ingestion Script
Processes PDF files and stores embeddings in ChromaDB.
Run this before using the chatbot to populate the knowledge base.

Usage:
    python ingest.py                           # Ingest from default knowledge_base/ directory
    python ingest.py --pdf path/to/file.pdf    # Ingest a specific PDF
    python ingest.py --reset                   # Clear and re-ingest
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: str, reset: bool = False):
    """Ingest a single PDF file into ChromaDB."""
    print(f"\n{'='*60}")
    print(f"  📄 RAG Knowledge Base Ingestion")
    print(f"{'='*60}")

    # Initialize components
    processor = DocumentProcessor()
    embedding_mgr = EmbeddingManager()

    # Reset if requested
    if reset:
        print("\n🗑️  Clearing existing vector store...")
        embedding_mgr.delete_collection()

    # Process PDF
    print(f"\n📂 Processing: {pdf_path}")
    chunks = processor.process(pdf_path)
    print(f"   ✅ Created {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    # Show sample chunk
    if chunks:
        print(f"\n📝 Sample chunk (#{chunks[0].metadata.get('chunk_id', 0)}):")
        print(f"   \"{chunks[0].page_content[:150]}...\"")

    # Create vector store
    print(f"\n🔢 Generating embeddings with {config.EMBEDDING_MODEL_NAME}...")
    vectorstore = embedding_mgr.create_vectorstore(chunks)

    # Verify
    stats = embedding_mgr.get_collection_stats()
    print(f"\n✅ Ingestion complete!")
    print(f"   📊 Documents in store: {stats.get('document_count', 'N/A')}")
    print(f"   💾 Stored at: {config.CHROMA_PERSIST_DIR}")
    print(f"{'='*60}\n")

    return vectorstore


def ingest_directory(directory: str, reset: bool = False):
    """Ingest all PDFs from a directory."""
    print(f"\n{'='*60}")
    print(f"  📄 RAG Knowledge Base Ingestion (Directory Mode)")
    print(f"{'='*60}")

    processor = DocumentProcessor()
    embedding_mgr = EmbeddingManager()

    if reset:
        print("\n🗑️  Clearing existing vector store...")
        embedding_mgr.delete_collection()

    print(f"\n📂 Scanning directory: {directory}")
    chunks = processor.process_directory(directory)

    if not chunks:
        print("   ⚠️  No PDF files found or no content extracted!")
        return None

    print(f"   ✅ Total chunks: {len(chunks)}")

    print(f"\n🔢 Generating embeddings with {config.EMBEDDING_MODEL_NAME}...")
    vectorstore = embedding_mgr.create_vectorstore(chunks)

    stats = embedding_mgr.get_collection_stats()
    print(f"\n✅ Ingestion complete!")
    print(f"   📊 Documents in store: {stats.get('document_count', 'N/A')}")
    print(f"   💾 Stored at: {config.CHROMA_PERSIST_DIR}")
    print(f"{'='*60}\n")

    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG knowledge base")
    parser.add_argument("--pdf", type=str, help="Path to a specific PDF file to ingest")
    parser.add_argument("--dir", type=str, default=config.PDF_DIRECTORY,
                        help=f"Directory containing PDFs (default: {config.PDF_DIRECTORY})")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing vector store before ingestion")
    args = parser.parse_args()

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"❌ File not found: {args.pdf}")
            sys.exit(1)
        ingest_pdf(args.pdf, reset=args.reset)
    else:
        if not os.path.exists(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            print(f"   Create the '{args.dir}' directory and add PDF files to it.")
            sys.exit(1)
        ingest_directory(args.dir, reset=args.reset)


if __name__ == "__main__":
    main()
